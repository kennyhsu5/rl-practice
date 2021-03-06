import numpy as np
import tensorflow as tf
import gym
import os
import random
import time
import shutil
from gym import wrappers
from gym.spaces import Discrete, Box

"""

Practice implementation of REINFORCE algorithm

Algorithm based on https://arxiv.org/pdf/1604.06778.pdf
"""

EPS = 1e-8

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def placeholder(space, name=None):
    if space is None:
        return tf.placeholder(dtype=tf.float32, shape=(None,), name=name)
    if isinstance(space, Box):
        shape = combined_shape
        return tf.placeholder(dtype=tf.float32, shape=combined_shape(None, space.shape))
    if isinstance(space, Discrete):
        return tf.placeholder(dtype=tf.int32, shape=(None,))
    raise NotImplementedError

def mlp(x, hidden_sizes=(32,), activation=tf.nn.relu, output=None):
    for h in hidden_sizes[:-1]:
        x = tf.layers.dense(x, h, kernel_initializer=tf.contrib.layers.variance_scaling_initializer(), activation=activation)
    return tf.layers.dense(x, hidden_sizes[-1], kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),  activation=output)

def gaussian_likelihood(x, mu, log_std):
    s = -0.5 * (((x-mu)/(tf.exp(log_std)+EPS))**2 + 2*log_std + np.log(2*np.pi))
    return tf.reduce_sum(s, axis=1)

def mlp_gaussian(x, a, action_space, hidden_sizes=(32,), activation=tf.nn.relu, output=None):
    act_dim = action_space.shape[0]
    mu = tf.squeeze(mlp(x, list(hidden_sizes)+[act_dim], activation, output))
    log_std = tf.get_variable(name="log_std", initializer=-0.5*np.ones(act_dim, dtype=np.float32))
    pi = mu + tf.random_normal(tf.shape(mu)) * tf.exp(log_std)
    log_liklihood = gaussian_likelihood(a, mu, log_std)
    return pi, log_liklihood

def mlp_categorical(x, a, action_space, hidden_sizes=(32,), activation=tf.nn.relu, output=None):
    act_dim = action_space.n
    logits = mlp(x, list(hidden_sizes)+[act_dim], activation, None)
    dist = tf.contrib.distributions.Categorical(logits=logits)
    pi = dist.sample(1)
    log_liklihood = tf.math.log(tf.nn.softmax(logits))
    return pi, log_liklihood

def mlp_ac(x, a, action_space, hidden_sizes=(128, 64), activation=tf.nn.relu, output=None):
    policy = mlp_gaussian
    if isinstance(action_space, Discrete):
        policy = mlp_categorical

    with tf.variable_scope("pi"):
        pi, log_liklihood = policy(x, a, action_space, hidden_sizes, activation, output)
    with tf.variable_scope("b"):
        v = tf.squeeze(mlp(x, list(hidden_sizes) + [1], activation, None), axis=1)
    return pi, log_liklihood, v

def reinforce(env_fn, actor_critic=mlp_ac, ac_kwargs=dict(), dir="", epochs=1000,
              episodes_per_epoch=100, max_ep_len=150, seed=0, gamma=0.999, random=False,
              pi_lr=0.0003, v_lr=0.0003):
    if os.path.isdir(dir):
        shutil.rmtree(dir)
    os.makedirs(dir)
    file = open("{}/train.txt".format(dir), "w+")

    start = time.time()

    tf.set_random_seed(seed)
    np.random.seed(seed)

    env = env_fn()
    action_type = "Discrete" if isinstance(env.action_space, Discrete) else "Continuous"
    file.write("Environment is {}\n".format(action_type))

    x_ph, a_ph, v_ph, r_ph = placeholder(env.observation_space, "input"), placeholder(env.action_space, "action"), placeholder(None, name="baseline"), placeholder(None, name="reward")
    pi, log_liklihood, v = mlp_ac(x_ph, a_ph, env.action_space, **ac_kwargs)

    adv = r_ph - v_ph
    loss = tf.reduce_mean(log_liklihood * tf.expand_dims(adv, 1))
    v_loss = tf.losses.mean_squared_error(r_ph, v)
    tf.summary.scalar("loss", loss)
    tf.summary.scalar("v_loss", v_loss)
    tf.summary.scalar("avg_ep_length", tf.shape(x_ph)[0] / episodes_per_epoch)

    opt = tf.train.AdamOptimizer(learning_rate=pi_lr)
    train_pi = opt.minimize(-loss)
    v_opt = tf.train.AdamOptimizer(learning_rate=v_lr)
    train_v = v_opt.minimize(v_loss)

    sess = tf.Session()
    train_writer = tf.summary.FileWriter(dir, sess.graph)
    merge = tf.summary.merge_all()
    sess.run(tf.global_variables_initializer())

    def act(o, action_space):
        if random:
            return action_space.sample()
        a = sess.run(pi, feed_dict={x_ph: o.reshape(1,-1)})
        if isinstance(env.action_space, Discrete):
            return a[0][0]
        return np.clip(a, action_space.low, action_space.high)

    for epoch in range(epochs):
        # Gather data
        size, episodes = 0, []
        for j in range(episodes_per_epoch):
            episode = []
            o, r, d, ep_len = env.reset(), 0, False, 0
            while not d:
                a = act(o, env.action_space)
                o, r, d, _ = env.step(a)
                ep_len += 1
                episode.append((o, a, r))
                size += 1
                if ep_len > max_ep_len:
                    break
            episodes.append(episode)

        # Prepare data
        xs = np.zeros(combined_shape(size, env.observation_space.shape), dtype=np.float32)
        acts = np.zeros(combined_shape(size, env.action_space.shape), dtype=np.float32)
        rs = np.zeros(size, dtype=np.float32)
        i = 0
        total_r = 0
        ep_rs = []
        for ep in episodes:
            ep_r = 0
            ep_cumulative_r = 0
            for o, a, r in reversed(ep):
                total_r += r
                ep_r = ep_r * gamma + r
                ep_cumulative_r += r
                xs[i], acts[i], rs[i] = o, a, ep_r
                i += 1
            ep_rs.append(ep_cumulative_r)

        file.write("Epoch {}:\n".format(epoch))
        file.write("  Avg Reward: {}\n".format(total_r / episodes_per_epoch))
        file.write("  Min Reward: {}\n".format(min(ep_rs)))
        file.write("  Max Reward: {}\n".format(max(ep_rs)))
        file.write("  Std Reward: {}\n".format(np.std(np.array(ep_rs))))
        file.write("  Avg Episode Len: {}\n".format(size / episodes_per_epoch))

        # Update/train model
        v_val = sess.run(v, feed_dict={x_ph: xs})
        _, _, summary = sess.run([train_pi, train_v, merge], feed_dict={x_ph: xs, a_ph: acts, r_ph: rs, v_ph: v_val})
        train_writer.add_summary(summary, epoch)
    train_writer.close()

    file.write("\nTraining time: {}".format(time.time() - start))

    file.write("\n\nTesting...\n")
    print("  Tests:")
    for t in range(5):
        o, d, ep_r = env.reset(), False, 0
        while not d:
            a = act(o, env.action_space)
            o, r, d, _ = env.step(a)
            ep_r += r
        print("    {}: {}".format(t, ep_r))
        file.write("Reward: {}\n".format(ep_r))
    file.close()

def hyp_hash(hidden_sizes, seed, gamma, episodes_per_epoch, max_ep_len, random, v_lr, pi_lr):
    return "{}_{}_{}_{}_{}_{}_{}_{}".format(seed, gamma, episodes_per_epoch, max_ep_len, random, v_lr, pi_lr, hidden_sizes)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="InvertedDoublePendulum-v2")
    parser.add_argument("--random", type=bool, default=False)
    parser.add_argument("--search_count", type=int, default=50)
    args = parser.parse_args()

    algo = "reinforce"
    total_episodes = 50000

    hidden_sizes = [[32, 32], [64, 32], [64, 64], [128, 64], [128, 128]]

    print("Environment: {}".format(args.env))
    print("Algotirhm: {}\n".format(algo))

    for s in range(args.search_count):
        tf.reset_default_graph()
        seed = random.randint(0, 1000)
        gamma = np.random.uniform(0.99, 1,)
        episodes_per_epoch = random.randrange(25, 100, 5)
        max_ep_len = random.randrange(150, 250, 10)
        lr = np.random.uniform(0.00001, 0.001)
        hidden_size = random.choice(hidden_sizes)
        epochs = int(total_episodes / episodes_per_epoch)

        dir = "tf_logs/{}/{}/{}".format(algo, args.env, hyp_hash(hidden_size, seed, gamma, episodes_per_epoch, max_ep_len, args.random, lr, lr))

        print("Search {}: ".format(s))
        print("  seed: {}".format(seed))
        print("  epoch: {}".format(epochs))
        print("  gamma: {}".format(gamma))
        print("  episodes_per_epoch: {}".format(episodes_per_epoch))
        print("  max_ep_len: {}".format(max_ep_len))
        print("  lr: {}".format(lr))
        print("  hidden_size: {}".format(hidden_size))
        print("  dir: {}".format(dir))

        kwargs = dict(
            env_fn=lambda : gym.make(args.env),
            dir=dir,
            actor_critic=mlp_ac,
            ac_kwargs=dict(hidden_sizes=hidden_size),
            seed=seed,
            gamma=gamma,
            epochs=epochs,
            episodes_per_epoch=episodes_per_epoch,
            max_ep_len=max_ep_len,
            pi_lr=lr,
            v_lr=lr,
            random=args.random,
        )
        reinforce(**kwargs)
