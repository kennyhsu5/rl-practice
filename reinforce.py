import numpy as np
import tensorflow as tf
import gym

"""

Practice implementation of REINFORCE algorithm

Algorithm based on https://arxiv.org/pdf/1604.06778.pdf
"""

def placeholder(dim=None):
    return tf.placeholder(dtype=tf.float32, shape=(None, dim) if dim else (None,))

def mlp(x, hidden_sizes=(32,), activation_fn=tf.tanh, output_fn=None):
    for h in hidden_sizes[:-1]:
        x = tf.layers.dense(x, h, activation_fn=activation_fn)
    return tf.layers.dense(x, h[-1], activation_fn=output_fn)

def mlp_ac(x, act_dim, hidden_sizes=(128, 64), activation_fn=tf.nn.relu, output_fn=tf.tanh):
    with tf.variable_scope("pi"):
        pi = mlp(x, list(hidden_sizes) + [act_dim], activation_fn, output_fn)
    with tf.variable_scope("b"):
        b = tf.squeeze(mlp(x, list(hidden_sizes) + [1], acivation_fn, None), axis=1)
    return pi, b

def reinforce(env_fn, actor_critic=mlp_ac, ac_kwargs=dict(), max_ep_len=150, seed=0, epochs=100):
    tf.set_random_seed(seed)
    np.random.seed(seed)

    env = env_fn()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    x_ph, a_ph, r_ph = placeholder(obs_dim), placeholder(act_dim), placeholder()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="InvertedDoublePendulum-v2")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--exp_name", type=str, default="reinforce")
    args = parser.parse_args()

    kwargs = dict(
        env_fn=lambda : gym.make(args.env),
        actor_critic=mlp_ac,
        ac_kwargs=dict(hidden_sizes=[128, 128]),
        max_ep_len=150,
        seed=args.seed,
        epochs=10,
    )

    reinforce(**kwargs)
