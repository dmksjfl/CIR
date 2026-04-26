import numpy as np
import torch
import argparse
import os
import utils
import random
import math
import time

import algo.sac as SAC
import algo.cir as CIR
from tensorboardX import SummaryWriter
from humanoid import make_humanoid_env


def eval_policy(policy, env_name, seed, eval_episodes=10, action_repeat=2, eval_cnt=None):
    eval_env = make_humanoid_env(env_name, args.seed+100, 1000, action_repeat)

    avg_reward = 0.
    for episode_idx in range(eval_episodes):
        state, info = eval_env.reset()
        done = False
        while not done:
            action = policy.select_action(np.array(state), test=True)
            next_state, reward, terminated, truncated, _ = eval_env.step(action)

            avg_reward += reward
            state = next_state
            if terminated or truncated:
                done = True

    avg_reward /= eval_episodes

    print("[{}] Evaluation over {} episodes: {}".format(eval_cnt, eval_episodes, avg_reward))
    
    return avg_reward


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="./logs")
    parser.add_argument("--policy", default="cir", help='policy to use, support CIR')
    parser.add_argument("--env", default="h1-reach-v0")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--start-steps", default=5e3, type=int, help='Number of steps for the warm-up stage using random policy')
    parser.add_argument("--eval-freq", default=5000, type=int, help='Number of steps per evaluation')
    parser.add_argument("--steps", default=2e6, type=int, help='Maximum number of steps')

    parser.add_argument("--discount", default=0.99, type=float, help='Discount factor')
    parser.add_argument("--tau", default=0.005, help='Target network update rate')
    parser.add_argument("--auto-tune", action="store_true")               
    
    parser.add_argument("--actor-lr", default=3e-4, type=float)     
    parser.add_argument("--critic-lr", default=3e-4, type=float)    
    parser.add_argument("--hidden-sizes", default='512,512', type=str)  
    parser.add_argument("--batch-size", default=256, type=int)      # Batch size for both actor and critic

    parser.add_argument("--save-model", action="store_true")        # Save model and optimizer parameters
    parser.add_argument("--load-model", default="")                 # Model load file name, "" doesn't load, "default" uses file_name

    parser.add_argument("--expl-noise", default=0.1, type=float)                # Std of Gaussian exploration noise
    parser.add_argument("--policy-noise", default=0.2, type=float)              # Noise added to target policy during critic update
    parser.add_argument("--noise-clip", default=0.5, type=float)                # Range to clip target policy noise

    parser.add_argument("--policy-freq", default=2, type=int, help='Frequency of delayed policy updates')
    parser.add_argument("--utd", action="store_true")
    parser.add_argument("--smr", action="store_true")
    parser.add_argument("--ratio", default=1, type=int)   # SMR or UTD ratio
    parser.add_argument("--cuda", default='0', type=str)
    
    args = parser.parse_args()

    print("------------------------------------------------------------")
    print("Policy: {}, Env: {}, Seed: {}".format(args.policy, args.env, args.seed))
    print("------------------------------------------------------------")

    # for humanoid bench: egl device!
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(args.cuda)
    
    outdir = args.dir + '/' + args.policy.lower() + '/' + args.env + '/r' + str(args.seed)
    writer = SummaryWriter('{}/tb'.format(outdir))
    if args.save_model and not os.path.exists("{}/models".format(outdir)):
        os.makedirs("{}/models".format(outdir))

    # following simba, action repeat is set to be 2 regardlesss
    action_repeat = 2

    # Heuristic discount factor
    heuristic_set = ['h1-balance_hard-v0', 'h1-hurdle-v0', 'h1-pole-v0', 'h1-run-v0', 'h1-walk-v0', 'h1-reach-v0', 'h1-sit_simple-v0', 'h1-sit_hard-v0', 'h1-crawl-v0']
    if args.env in heuristic_set:
        args.discount = 0.95
    
    env = make_humanoid_env(args.env, args.seed, 1000, action_repeat)
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0] 
    max_action = float(env.action_space.high[0])
    min_action = -max_action
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": args.discount,
        "tau": args.tau,
        "hidden_sizes": [int(hs) for hs in args.hidden_sizes.split(',')],
        "actor_lr": args.actor_lr,
        "critic_lr": args.critic_lr,
        "device": device,
        "smr": True if args.smr else False,
        "utd": True if args.utd else False,
        "ratio": args.ratio,
    }

    if args.policy.lower() == "sac":
        policy = SAC.SAC(**kwargs)
    elif args.policy.lower() == "cir":
        policy = CIR.CIR(**kwargs)
    else:
        raise NotImplementedError

    if args.load_model != "":
        policy.load("./models/{}".format(args.load_model))
    
    ## write logs to record training parameters
    with open(outdir + 'log.txt','w') as f:
        f.write('\n Policy: {}; Env: {}, seed: {}'.format(args.policy, args.env, args.seed))
        for item in kwargs.items():
            f.write('\n {}'.format(item))

    replay_buffer = utils.ReplayBuffer(state_dim, action_dim, device)

    eval_cnt = 0

    state, info = env.reset()
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0

    for t in range(int(args.steps/action_repeat)):
        episode_timesteps += 1

        # select action randomly or according to policy
        if t < int(args.start_steps/action_repeat):
            action = (max_action - min_action) * np.random.random(env.action_space.shape) + min_action
        else:
            action = policy.select_action(np.array(state), test=False)

        next_state, reward, terminated, truncated, info = env.step(action)
        done_bool = float(truncated or terminated) if episode_timesteps < 1000/int(action_repeat) else 0

        replay_buffer.add(state, action, next_state, reward, done_bool)

        state = next_state
        episode_reward += reward
        if t >= int(args.start_steps/action_repeat):
            policy.train(replay_buffer, args.batch_size, writer)
        
        if terminated or truncated: 
            print("Total T: {} Episode Num: {} Episode T: {} Reward: {}".format(t+1, episode_num+1, episode_timesteps, episode_reward))
            writer.add_scalar('train/train return', episode_reward, global_step = int((t+1) * action_repeat))

            state, info = env.reset()
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1 

        if int((t + 1)*action_repeat) % args.eval_freq == 0:
            eval_return = eval_policy(policy, args.env, args.seed, action_repeat=action_repeat, eval_cnt=eval_cnt)
            writer.add_scalar('test/test return', eval_return, global_step = int((t + 1)*action_repeat))
            eval_cnt += 1

            if args.save_model:
                policy.save('{}/models/model'.format(outdir))
    writer.close()
