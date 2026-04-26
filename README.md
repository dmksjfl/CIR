# Constrained Initial Representations

This is the code for our novel method, Constrained Initial Representations (CIR).


## Results

We gather CIR logs on all evaluated DMC and HumanoidBench tasks in the `results/cir.csv` file.


## How to run

To reproduce our reported results in the submission, please check the following instructions:

You need to install some necessary packages before running our code, please check the `requirement.txt` file.

**MuJoCo**

```
# Additional environmental variables for headless rendering
export MUJOCO_GL="egl"
export MUJOCO_EGL_DEVICE_ID="0"
export MKL_SERVICE_FORCE_INTEL="0"
```

**HumanoidBench**
```
git clone https://github.com/joonleesky/humanoid-bench
cd humanoid-bench
pip install -e .
```

In the `dmc` file, run `train_dmc.py` by calling:

```
CUDA_VISIBLE_DEVICES=0 python train_dmc.py --env cheetah-run --policy cir --smr --ratio 2 --hidden-sizes 512,512 --seed 2 --dir test
```

Please note that you must run `train_dmc.py` in the `dmc` directory rather than using commands like `python dmc/train_dmc.py`, you may need to
```
cd dmc
ls dmc
CUDA_VISIBLE_DEVICES=0 python train_dmc.py --env cheetah-run --policy cir --smr --ratio 2 --hidden-sizes 512,512 --seed 2 --dir test
```

Similarly, you need to `cd humanoid` to run CIR on HumanoidBench tasks. In the `humanoidbench` file, run `train_humanoid.py` by calling:

```
python train_humanoid.py --env h1-reach-v0 --policy cir --hidden-sizes 512,512 --cuda 0 --smr --ratio 2 --seed 2 --dir test
```

## Citation

If you find our work interesting or use our work in your paper, please consider citing our paper:
```
@article{lyu2026temporal,
  title={Temporal Difference Learning with Constrained Initial Representations},
  author={Lyu, Jiafei and Yang, Jingwen and Qiao, Zhongjian and Liu, Runze and Liu, Zeyuan and Ye, Deheng and Lu, Zongqing and Li, Xiu},
  journal={arXiv preprint arXiv:2602.11800},
  year={2026}
}
```
