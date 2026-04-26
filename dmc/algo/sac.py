import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, TransformedDistribution, constraints

from torch.distributions.transforms import Transform

class TanhTransform(Transform):
    r"""
    Transform via the mapping :math:`y = \tanh(x)`.
    It is equivalent to
    ```
    ComposeTransform([AffineTransform(0., 2.), SigmoidTransform(), AffineTransform(-1., 2.)])
    ```
    However this might not be numerically stable, thus it is recommended to use `TanhTransform`
    instead.
    Note that one should use `cache_size=1` when it comes to `NaN/Inf` values.
    """
    domain = constraints.real
    codomain = constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        # We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
        # one should use `cache_size=1` instead
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        # We use a formula that is more numerically stable, see details in the following link
        # https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/bijectors/tanh.py#L69-L80
        return 2. * (math.log(2.) - x - F.softplus(-2. * x))


class MLPNetwork(nn.Module):
    
    def __init__(self, input_dim, output_dim, hidden_size=256):
        super(MLPNetwork, self).__init__()
        self.network = nn.Sequential(
                        nn.Linear(input_dim, hidden_size),
                        nn.ReLU(),
                        nn.Linear(hidden_size, hidden_size),
                        nn.ReLU(),
                        nn.Linear(hidden_size, output_dim),
                        )
    
    def forward(self, x):
        return self.network(x)


class Policy(nn.Module):

    def __init__(self, state_dim, action_dim, max_action, hidden_size=[256,256]):
        super(Policy, self).__init__()
        self.action_dim = action_dim
        self.max_action = max_action
        self.network = MLPNetwork(state_dim, action_dim * 2, hidden_size[0])

    def forward(self, x, get_logprob=False):
        mu_logstd = self.network(x)
        mu, logstd = mu_logstd.chunk(2, dim=1)
        logstd = torch.clamp(logstd, -20, 2)
        std = logstd.exp()
        dist = Normal(mu, std)
        transforms = [TanhTransform(cache_size=1)]
        dist = TransformedDistribution(dist, transforms)
        action = dist.rsample()
        if get_logprob:
            logprob = dist.log_prob(action).sum(axis=-1, keepdim=True)
        else:
            logprob = None
        mean = torch.tanh(mu)
        return action * self.max_action, logprob, mean * self.max_action


class DoubleQFunc(nn.Module):
    
    def __init__(self, state_dim, action_dim, hidden_size=[256,256]):
        super(DoubleQFunc, self).__init__()
        self.network1 = MLPNetwork(state_dim + action_dim, 1, hidden_size[0])
        self.network2 = MLPNetwork(state_dim + action_dim, 1, hidden_size[0])

    def forward(self, state, action):
        x = torch.cat((state, action), dim=1)
        return self.network1(x), self.network2(x)


class SAC(object):

    def __init__(self,
                 device,
                 state_dim, 
                 action_dim, 
                 max_action,
                 lr=3e-4, 
                 discount=0.99, 
                 tau=5e-3, 
                 actor_lr=3e-4,
                 critic_lr=3e-4,
                 hidden_sizes=[256, 256], 
                 update_interval=1,
                 target_entropy=None,
                 usenorm=False,
                 utd=False,
                 smr=False,
                 ratio=1,):
        self.device = device
        self.discount = discount
        self.tau = tau
        self.target_entropy = target_entropy if target_entropy else -action_dim
        self.update_interval = update_interval
        self.ratio = ratio if smr or utd else 1
        self.utd = utd
        self.smr = smr
        self.max_action = max_action
        self.action_dim = action_dim

        # aka critic
        self.q_funcs = DoubleQFunc(state_dim, action_dim, hidden_size=hidden_sizes).to(self.device)
        self.target_q_funcs = copy.deepcopy(self.q_funcs)
        self.target_q_funcs.eval()
        for p in self.target_q_funcs.parameters():
            p.requires_grad = False

        # aka actor
        self.policy = Policy(state_dim, action_dim, max_action, hidden_size=hidden_sizes).to(self.device)

        # aka temperature
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)

        self.q_optimizer = torch.optim.Adam(self.q_funcs.parameters(), lr=critic_lr)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=actor_lr)
        self.temp_optimizer = torch.optim.Adam([self.log_alpha], lr=actor_lr)
    
    def select_action(self, state, test=False):
        with torch.no_grad():
            action, _, mean = self.policy(torch.Tensor(state).view(1,-1).to(self.device))
        if test:
            return mean.squeeze().cpu().numpy()
        else:
            return action.squeeze().cpu().numpy()

    def update_target(self):
        """moving average update of target networks"""
        with torch.no_grad():
            for target_q_param, q_param in zip(self.target_q_funcs.parameters(), self.q_funcs.parameters()):
                target_q_param.data.copy_(self.tau * q_param.data + (1.0 - self.tau) * target_q_param.data)

    def update_q_functions(self, state_batch, action_batch, reward_batch, nextstate_batch, not_done_batch):
        with torch.no_grad():
            nextaction_batch, logprobs_batch, _ = self.policy(nextstate_batch, get_logprob=True)
            q_t1, q_t2 = self.target_q_funcs(nextstate_batch, nextaction_batch)
            # take min to mitigate positive bias in q-function training
            q_target = torch.min(q_t1, q_t2)
            value_target = reward_batch + not_done_batch * self.discount * (q_target - self.alpha * logprobs_batch)
        q_1, q_2 = self.q_funcs(state_batch, action_batch)
        loss_1 = F.mse_loss(q_1, value_target)
        loss_2 = F.mse_loss(q_2, value_target)
        return loss_1, loss_2

    def update_policy_and_temp(self, state_batch):
        action_batch, logprobs_batch, _ = self.policy(state_batch, get_logprob=True)
        q_b1, q_b2 = self.q_funcs(state_batch, action_batch)
        qval_batch = torch.min(q_b1, q_b2)
        policy_loss = (self.alpha * logprobs_batch - qval_batch).mean()
        temp_loss = -self.alpha * (logprobs_batch.detach() + self.target_entropy).mean()
        return policy_loss, temp_loss

    def train(self, replay_buffer, batch_size=256):
        if self.utd:
            for M in range(self.ratio):
                state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
                # update q-funcs
                q1_loss_step, q2_loss_step = self.update_q_functions(state, action, reward, next_state, not_done)
                q_loss_step = q1_loss_step + q2_loss_step
                self.q_optimizer.zero_grad()
                q_loss_step.backward()
                self.q_optimizer.step()

                self.update_target()

                # update policy and temperature parameter
                for p in self.q_funcs.parameters():
                    p.requires_grad = False
                pi_loss_step, a_loss_step = self.update_policy_and_temp(state)
                self.policy_optimizer.zero_grad()
                pi_loss_step.backward()
                self.policy_optimizer.step()
                self.temp_optimizer.zero_grad()
                a_loss_step.backward()
                self.temp_optimizer.step()
                for p in self.q_funcs.parameters():
                    p.requires_grad = True
        else:
            # whether SMR or not 
            ############################################
            #### Adding Sample Multiple Reuse (SMR) ####
            ############################################
            state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
            
            for M in range(self.ratio):
                # update q-funcs
                q1_loss_step, q2_loss_step = self.update_q_functions(state, action, reward, next_state, not_done)
                q_loss_step = q1_loss_step + q2_loss_step
                self.q_optimizer.zero_grad()
                q_loss_step.backward()
                self.q_optimizer.step()

                self.update_target()

                # update policy and temperature parameter
                for p in self.q_funcs.parameters():
                    p.requires_grad = False
                pi_loss_step, a_loss_step = self.update_policy_and_temp(state)
                self.policy_optimizer.zero_grad()
                pi_loss_step.backward()
                self.policy_optimizer.step()
                self.temp_optimizer.zero_grad()
                a_loss_step.backward()
                self.temp_optimizer.step()
                for p in self.q_funcs.parameters():
                    p.requires_grad = True

    @property
    def alpha(self):
        return self.log_alpha.exp()

