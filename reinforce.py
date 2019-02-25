import numpy as np
import tensorflow as tf
import gym
from gym import wrappers
from gym.spaces import Discrete

"""

Practice implementation of REINFORCE algorithm

Algorithm based on https://arxiv.org/pdf/1604.06778.pdf
"""

EPS = 1e-8

def placeholder(dim=None):
    return tf.placeholder(dtype=tf.float32, shape=(None, dim) if dim else (None,))

def mlp(x, hidden_sizes=(32,), activation=tf.nn.relu, output=None):
    for h in hidden_sizes[:-1]:
        x = tf.layers.dense(x, h, activation=activation)
    return tf.layers.dense(x, hidden_sizes[-1], activation=output)

def gaussian_likelihood(x, mu, log_std):
    s = -0.5 * (((x-mu)/(tf.exp(log_std)+EPS))**2 + 2*log_std + np.log(2*np.pi))
    return tf.reduce_sum(s, axis=1)

def mlp_gaussian(x, a, action_space, hidden_sizes=(32,), activation=tf.nn.relu, output=None):
    act_dim = a.shape.as_list()[-1]
    mu = mlp(x, list(hidden_sizes)+[act_dim], activation, output)
    log_std = tf.get_variable(name="log_std", initializer=-0.5*np.ones(act_dim, dtype=np.float32))
    std = tf.exp(log_std)
    pi = mu + tf.random_normal(tf.shape(mu)) * std
    log_p = gaussian_likelihood(a, mu, log_std)
    return pi, log_p


def mlp_categorical(x, a, action_space, hidden_sizes=(32,), activation=tf.nn.relu, output=None):
    act_dim = action_space.n
    logits = mlp(x, list(hidden_sizes)+[act_dim], activation, None)
    log_p_all = tf.nn.log_softmax(logits)
    pi = tf.squeeze(tf.multinomial(logits,1), axis=1)
    log_p = tf.reduce_sum(tf.one_hot(a, depth=act_dim) * logp_all, axis=1)
    return pi, log_p

def mlp_ac(x, a, action_space, hidden_sizes=(128, 64), activation=tf.nn.relu, output=None):
    policy = mlp_gaussian
    if isinstance(action_space, Discrete):
        policy = mlp_categorical

    with tf.variable_scope("pi"):
        pi, log_p = policy(x, a, action_space, hidden_sizes, activation, output)
    with tf.variable_scope("b"):
        v = tf.squeeze(mlp(x, list(hidden_sizes) + [1], activation, None), axis=1)
    return pi, log_p, v

def reinforce(env_fn, actor_critic=mlp_ac, ac_kwargs=dict(), epochs=1000,
              episodes_per_epoch=100, max_ep_len=150, seed=0, gamma=0.9):
    tf.set_random_seed(seed)
    np.random.seed(seed)

    env = env_fn()
    action_space = env.action_space
    obs_dim = env.observation_space.shape[0]
    act_dim = action_space.shape[0]

    x_ph, a_ph, v_ph, r_ph = placeholder(obs_dim), placeholder(act_dim), placeholder(), placeholder()
    pi, log_p, _ = mlp_ac(x_ph, a_ph, env.action_space, **ac_kwargs)

    # loss =  tf.reduce_mean(log_p * (r_ph - v_ph))
    loss =  tf.reduce_mean(log_p * r_ph)

    opt = tf.train.AdamOptimizer(learning_rate=0.01)
    train_pi = opt.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    def act(o):
        a = sess.run(pi, feed_dict={x_ph: o.reshape(1,-1)})
        a = np.clip(a, action_space.low, action_space.high)
        # a = env.action_space.sample()
        return a


    for epoch in range(epochs):
        # Gather data
        size, episodes = 0, []
        for j in range(episodes_per_epoch):
            episode = []
            o, r, d, ep_len = env.reset(), 0, False, 0
            while not d:
                a = act(o)
                o, r, d, _ = env.step(a)
                ep_len += 1
                episode.append((o, a, r))
                size += 1
                if ep_len > max_ep_len:
                    break
            episodes.append(episode)

        # Prepare data
        xs = np.zeros([size, obs_dim], dtype=np.float32)
        acts = np.zeros([size, act_dim], dtype=np.float32)
        rs = np.zeros(size, dtype=np.float32)
        i = 0
        total_r = 0
        for ep in episodes:
            ep_r = 0
            for o, a, r in reversed(ep):
                total_r += r
                ep_r = ep_r * gamma + r
                xs[i], acts[i], rs[i] = o, a, ep_r
                i += 1
        print("Epoch {}:".format(epoch))
        print("  Avg Reward: {}".format(total_r / episodes_per_epoch))
        print("  Avg Episode Len: {}".format(size / episodes_per_epoch))

        # Update/train model
        sess.run(train_pi, feed_dict={x_ph: xs, a_ph: acts, r_ph: rs})

    for _ in range(5):
        o, d = env.reset(), False
        while not d:
            env.render()
            a = act(o)
            o, _, d, _ = env.step(a)


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
        seed=args.seed,
        epochs=50,
        episodes_per_epoch = 50,
        max_ep_len=150,
    )

    reinforce(**kwargs)
