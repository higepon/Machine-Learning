#!/usr/bin/env python

import gym
import numpy as np
import tensorflow as tf

image_size = 80 * 80
gamma = 0.99
num_batches = 40000
num_episodes_per_batch = 3


class PolicyGradientAgent:
    def __init__(self, parameters, sess):
        self._sess = sess

        self._observation, self._sample, self._acts, self._advantages, self._loss_summary, self._train, self._global_step = self._build_graph(
            parameters)

    @staticmethod
    def _build_graph(parameters):
        global_step = tf.get_variable("global_step", shape=[],
                                      initializer=tf.constant_initializer(0, dtype=tf.int64), dtype=tf.int64,
                                      trainable=False)

        observation = tf.placeholder(tf.float32, shape=[None, parameters['input_size']], name="observation")

        hidden1 = tf.contrib.layers.fully_connected(
            inputs=observation,
            num_outputs=parameters['hidden_size'],
            activation_fn=tf.nn.relu,
            weights_initializer=tf.random_normal_initializer())

        logits = tf.contrib.layers.fully_connected(
            inputs=hidden1,
            num_outputs=parameters['num_actions'],
            activation_fn=None)

        sample = tf.reshape(tf.multinomial(logits, 1), [])

        log_prob = tf.log(tf.nn.softmax(logits))

        actions = tf.placeholder(tf.int32)
        advantages = tf.placeholder(tf.float32)

        indices = tf.range(0, tf.shape(log_prob)[0]) * tf.shape(log_prob)[1] + actions
        action_prob = tf.gather(tf.reshape(log_prob, [-1]), indices)

        loss = -tf.reduce_sum(tf.multiply(action_prob, advantages))

        # log loss
        loss_summary = tf.summary.scalar("loss", loss)

        optimizer = tf.train.RMSPropOptimizer(parameters['learning_rate'])
        train = optimizer.minimize(loss, global_step=global_step)
        return observation, sample, actions, advantages, loss_summary, train, global_step

    def act(self, observation):
        return self._sess.run(self._sample, feed_dict={self._observation: observation})

    def train_step(self, obs, acts, advantages):
        batch_feed = {self._observation: obs,
                      self._acts: acts,
                      self._advantages: advantages}
        return self._sess.run([self._train, self._loss_summary], feed_dict=batch_feed)

    def global_step(self):
        return self._global_step


# Borrowed from http://karpathy.github.io/2016/05/31/rl/.
# Written by Andrej Karpathy.
def process_observation(I):
    I = I[35:195]  # crop
    I = I[::2, ::2, 0]  # down sample by factor of 2
    I[I == 144] = 0  # erase background (background type 1)
    I[I == 109] = 0  # erase background (background type 2)
    I[I != 0] = 1  # everything else (paddles, ball) just set to 1
    return I.astype(np.float).ravel().reshape(1, image_size)


def play_episodes(env, agent, num_episodes):
    batch_observations = []
    batch_actions = []
    batch_rewards = []
    batch_my_score = 0
    batch_com_score = 0
    for _ in range(num_episodes):
        observations, actions, rewards, my_score, com_score = play_episode(env, agent)

        batch_my_score = batch_my_score + my_score
        batch_com_score = batch_com_score + com_score

        batch_observations.extend(observations)
        batch_actions.extend(actions)

        advantages = process_rewards(rewards)
        batch_rewards.extend(advantages)
    return batch_observations, batch_actions, batch_rewards, batch_my_score / num_episodes, batch_com_score / num_episodes


def play_episode(env, agent):
    observation, reward, done = env.reset(), 0, False
    observations, actions, rewards = [], [], []

    prev_processed_observation = None
    my_score = 0
    com_score = 0

    while not done:
        #        env.render()
        processed_observation = process_observation(observation)

        simplified_observation = processed_observation - prev_processed_observation if prev_processed_observation is not None else np.zeros_like(
            processed_observation)
        prev_processed_observation = processed_observation

        observations.append(simplified_observation)

        action = agent.act(simplified_observation)

        # 2 is UP and 3 is DOWN. Pong-v0 specific.
        observation, reward, done, _ = env.step(action + 2)

        if reward == 1.0:
            my_score = my_score + 1
        elif reward == -1.0:
            com_score = com_score + 1

        # note that we don't pass action + 2, as this is input for the model.
        actions.append(action)
        rewards.append(reward)

    return observations, actions, rewards, my_score, com_score


# this is specific to pong
def process_rewards(rewards):
    # In one episode, there are some 1 (The COM loses the ball) and -1 (You loses the ball).
    # The actions led to the result get some rewards based on gamma.
    processed_rewards = np.zeros_like(rewards)
    current_reward = 0
    for t in reversed(range(0, len(rewards))):
        # this is boundary
        if rewards[t] != 0:
            current_reward = 0
        current_reward = current_reward * gamma + rewards[t]
        processed_rewards[t] = current_reward
    return processed_rewards


def main():
    parameters = {
        'input_size': image_size,
        'hidden_size': 200,
        'num_actions': 2,
        'learning_rate': 1e-4
    }

    env = gym.make("Pong-v0")

    with tf.Session() as sess:

        # Set up model before we load checkpoint.
        agent = PolicyGradientAgent(parameters, sess)

        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state('./model')
        if ckpt:
            last_model = ckpt.model_checkpoint_path
            print("loading " + last_model)
            saver.restore(sess, last_model)
        else:
            print("Starting fresh model")
            sess.run(tf.global_variables_initializer())

        train_writer = tf.summary.FileWriter('train', sess.graph)

        for batch in range(num_batches):
            print("=========")
            print("Step {}".format(sess.run(agent.global_step())))

            batch_observations, batch_actions, batch_rewards, batch_my_score, batch_com_score = play_episodes(env,
                                                                                                              agent,
                                                                                                              num_episodes_per_batch)

            print(" my score {:.2f} : com_score {:.2f}".format(batch_my_score, batch_com_score))

            # normalize rewards
            epsilon = 1e-10
            batch_rewards = (batch_rewards - np.mean(batch_rewards)) / (np.std(batch_rewards) + epsilon)

            _, summary = agent.train_step(np.vstack(batch_observations), batch_actions, batch_rewards)
            train_writer.add_summary(summary, sess.run(agent.global_step()))

            if batch % 5 == 0:
                saver.save(sess, './model/model.ckpt', global_step=agent.global_step())


if __name__ == "__main__":
    main()
