import numpy as np
import tensorflow as tf
import gym

"""

Practice implementation of REINFORCE algorithm

Algorithm based on https://arxiv.org/pdf/1604.06778.pdf
"""

def reinforce():
    pass

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="CartPole-v1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--exp_name", type=str, default="reinforce")
    args = parser.parse_args()

    reinforce()
