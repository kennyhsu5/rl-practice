import numpy as np
import tensorflow as tf

"""

Practice implementation of DQN

Algorithm based on https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
"""

class ReplayBuffer:

    def __init__(self, size, obs_space, action_space):
        self.states = np.zeros((size, obs_space.shape), dtype=np.float32)
        self.states_next = np.zeros((size, obs_space.shape), dtype=np.float32)
        self.actions = np.zeros((size, action_space.shape), dtype=np.float32)
        self.rewards = np.zeros((size,), dtype=np.float32)
        self.size = size
        self.count = 0

    def insert(self, s, s_next, a, r):
        i = self.count % self.size
        self.states[i] = s
        self.states_next[i] = s_next
        self.actions[i] = a
        self.rewards[i] = r

        self.count += 1

    def sample(self, size):
        count = min(self.size, self.count)
        idx = np.random.randint(count, size=size)
        return self.states[idx, :], self.states_next[idx, :], self.actions[idx, :], self.rewards[idx, :]


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="InvertedDoublePendulum-v2")
    parser.add_argument("--random", type=bool, default=False)
    parser.add_argument("--search_count", type=int, default=100)
    args = parser.parse_args()
