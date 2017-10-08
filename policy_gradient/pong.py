import gym
import numpy as np
import tensorflow as tf

num_hidden_layer_size = 200
image_height = 80
image_width = 80
image_size = 80 * 80
gamma = 0.99

# Tensorflow: 1.10
# gym: 0.9.3
# numpy: 1.13.1
class PolicyGradientAgent:
    def __init__(self, sess):
        with tf.name_scope("PolicyGradientAgent"):
            self._sess = sess
            self._X, self._log_probs, self._actions, self._rewards, self._loss, self._loss_summary, self._global_step, self._train = self._build_graph()

    @staticmethod
    def _build_graph():
        global_step = tf.get_variable("global_step", shape=[], initializer=tf.constant_initializer(0, dtype=tf.int64), dtype=tf.int64, trainable=False)
        X = tf.placeholder(tf.float32, [image_size, None], name="X")
        weight_init = tf.random_normal_initializer(mean=0.0, stddev=0.05)

        # activation of hidden layer
        W1 = tf.get_variable(name="W1", shape=[num_hidden_layer_size, image_size], initializer=weight_init)
        Z1 = tf.matmul(W1, X)
        tf.assert_equal(tf.shape(Z1)[0], num_hidden_layer_size)
        A1 = tf.nn.relu(Z1)
        tf.assert_equal(tf.shape(A1)[0], num_hidden_layer_size)

        # activation of output layer
        W2 = tf.get_variable(name="W2", shape=[1, num_hidden_layer_size], initializer=weight_init)
        Z2 = tf.matmul(W2, A1)
        tf.assert_equal(tf.shape(Z2)[0], 1)
        A2 = tf.nn.sigmoid(Z2)
        tf.assert_equal(tf.shape(A2)[0], 1)

        # we had log(0) here, so not using log. And also the original script is not using log for some reason.
        log_probs = A2

        # reward for each action take for input
        # shape=[1, batch_size]
        rewards = tf.placeholder(tf.float32, shape=[1, None], name="rewards")

        # to use subtract actions should have taken
        actions = tf.placeholder(tf.float32, shape=[1, None], name="actions")
        fake_labels = tf.to_float(tf.equal(actions, 2), name="fake_labels")

        # We chose action based on sampled probability p(y|x) and random dice.
        # When we chose to take opposite action to p(y|x), we should take into account that.
        addjusted_logprob = fake_labels - log_probs

        # We negate objective function, because higher reward is better.
        cost_function = -addjusted_logprob * rewards
        loss = tf.reduce_sum(cost_function)
        loss_summary = tf.summary.scalar("loss", loss)

        optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
        train = optimizer.minimize(loss, global_step=global_step)
        return X, log_probs, actions, rewards, loss, loss_summary, global_step, train

    def global_step(self):
        return self._global_step

    def act(self, observation):
        sampled_prob = self._sess.run(self._log_probs, {self._X: observation})
        # todo breaking abstraction here :)
        action = 2 if np.random.uniform() < sampled_prob[0][0] else 3  # roll the dice!
        return action

    def train(self, observations, actions, rewards):
        # todo normalize shouldn't be here
        rewards -= np.mean(rewards)
        rewards /= np.std(rewards)

        feed_dict = {
            self._X: observations,
            self._rewards: rewards,
            self._actions: actions,
        }
        return self._sess.run([self._train, self._loss_summary, self._loss, self._log_probs], feed_dict=feed_dict)


## copyright
def prepro(I):
  """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
  I = I[35:195] # crop
  I = I[::2,::2,0] # downsample by factor of 2
  I[I == 144] = 0 # erase background (background type 1)
  I[I == 109] = 0 # erase background (background type 2)
  I[I != 0] = 1 # everything else (paddles, ball) just set to 1
  return I.astype(np.float).ravel().reshape(image_size, 1)


def play_episode(agent, env):
    observation, reward, done = env.reset(), 0, False

    observations = []
    rewards = []
    actions = []
    prev_x = None
    while not done:
#s        env.render()
        cur_x = prepro(observation)
        x = cur_x - prev_x if prev_x is not None else np.zeros([image_size, 1])
        prev_x = cur_x

        action = agent.act(x)
        # todo refactor this reshape
        observations.append(cur_x.reshape(-1, 6400))
        rewards.append(reward)
        actions.append(action)

        observation, reward, done, _ = env.step(action)

    print(reward, "lost" if reward == 0.0 or reward == -1.0 else "win")
    return observations, rewards, actions


# this is specific to pong
def process_rewards(rewards):
    # In one episode, there are some 1 (The COM loses the ball) and -1 (You loses the ball).
    # The actions led to the result get some rewards based on gamma
    processed_rewards = np.zeros_like(rewards)
    current_reward = 0
    win_count = 0
    lose_count = 0
    for t in reversed(range(0, len(rewards))):
        # this is boundary
        if rewards[t] == 1.0:
            win_count = win_count + 1
        elif rewards[t] == -1.0:
            lose_count = lose_count + 1
        if rewards[t] != 0:
            current_reward = 0
        current_reward = current_reward * gamma + rewards[t]
        processed_rewards[t] = current_reward
    print("win:lose", win_count, ":", lose_count)
    return processed_rewards


def main():
    env = gym.make("Pong-v0")

    num_episodes_per_batch = 50

    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state('./model')
        model = PolicyGradientAgent(sess)
        saver = tf.train.Saver()
        if ckpt:
            last_model = ckpt.model_checkpoint_path
            print("load " + last_model)
            saver.restore(sess, last_model)
        else:
            sess.run(tf.global_variables_initializer())

        train_writer = tf.summary.FileWriter('train', sess.graph)

        for j in range(0, 100000):
            batch_observations = []
            batch_rewards = []
            batch_actions = []
            for i in range(num_episodes_per_batch):
                print("batch:", j, "episode:", i)
                observations, rewards, actions = play_episode(model, env)
                processed_rewards = process_rewards(rewards)
                batch_observations.extend(observations)
                batch_rewards.extend(processed_rewards)
                batch_actions.extend(actions)
            _, summary, loss, _ = model.train(np.vstack(batch_observations).T, np.vstack(actions).T, np.vstack(batch_rewards).T)

            if j % 5 == 0:
                print("saved model")
                saver.save(sess, './model/model.ckpt', global_step=model.global_step())
            print("current_step", sess.run(model.global_step()))
#            print(logprogs)
            print(loss)
#            print(hoge)
#            print(probs)

            train_writer.add_summary(summary, j)

    #with tf.Session() as sess:
#    sess.run(tf.global_variables_initializer())
#    sess.run(tf.local_variables_initializer())
    #sess.run(tf.initialize_all_variables())

#    observation = env.reset()
#    print(env.env.action_space)
#    observations = []
#    rewards = []


    # do it until done?




#  print('x=', x.shape)
#  h = np.dot(model['W1'], x)
#  print('h=', h.shape)
#  h[h<0] = 0 # ReLU nonlinearity
#  logp = np.dot(model['W2'], h)
#  print('logp=', logp.shape)
#  p = sigmoid(logp)


#print("x.shape", x.shape, "W.shape", W.shape)

#b = tf.Variable(tf.zeros([10]))
main()