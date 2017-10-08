import gym
import numpy as np
import tensorflow as tf

num_hidden_layer_size = 200
image_height = 80
image_width = 80
image_size = 80 * 80
gamma = 0.99


## todo
## use get_variable with proper scope
## have sess as ivar
## rename this to agent
class PolicyGradientModel:
    def __init__(self):
        self.__X = tf.placeholder(tf.float32, [image_size, None], name="X")

        self.W1 = tf.Variable(tf.random_normal([num_hidden_layer_size, image_size]), name="W11")
        W2 = tf.Variable(tf.random_normal([1, num_hidden_layer_size]), name="W2A")

        # activation of hidden layer
        Z1 = tf.matmul(self.W1, self.__X)
        tf.assert_equal(tf.shape(Z1)[0], num_hidden_layer_size)

        A1 = tf.nn.relu(Z1)
        tf.assert_equal(tf.shape(A1)[0], num_hidden_layer_size)

        # activation of output layer
        Z2 = tf.matmul(W2, A1)
        tf.assert_equal(tf.shape(Z2)[0], 1)

        A2 = tf.nn.sigmoid(Z2)
        tf.assert_equal(tf.shape(A2)[0], 1)
        self.__probs = A2

        # Core of policy gradient

        ## log_prob for each input
        ## shape=[1, batch_size]
        ## we had log(0) here, so not using log. And also the original script is not using log for some reason.
        self.__log_probs = A2 # tf.log(A2)

        ## reward for each action take for input
        ## shape=[1, batch_size]
        self.__advantages = tf.placeholder(tf.float32, shape=[1, None], name="advantages")

        ## to use subtract actions should have taken
        self.__actions = tf.placeholder(tf.float32, shape=[1, None], name="actions")
        fake_labels = self.__actions == 2

        # We chose action based on sampled probability p(y|x) and random dice.
        # When we chose to take opposite action to p(y|x), we should take into account that.
        addjusted_logprob = fake_labels - self.__log_probs

        # We negate objective function as higher reward is better
        self.__hoge = -addjusted_logprob * self.__advantages

        # todo: confirm if reduce_sum is right way here.
        self.__loss = tf.reduce_sum(self.__hoge)

        self.__loss_summary = tf.summary.scalar("loss", self.__loss)

        optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
        self.__train = optimizer.minimize(self.__loss)

    def act(self, sess, observation):
        sampled_prob = sess.run(self.__probs, {self.__X: observation})
        ## todo breaking abstraction here :)
        action = 2 if np.random.uniform() < sampled_prob[0][0] else 3  # roll the dice!
        return action

    def train(self, sess, observations, actions, rewards):
        ## todo normalize shouldn't be here
        rewards -= np.mean(rewards)
        rewards /= np.std(rewards)

        feed_dict = { self.__X: observations,
                self.__advantages: rewards,
                      self.__actions: actions}
        return sess.run([self.__train, self.__loss_summary, self.__loss, self.__log_probs, self.__hoge, self.__probs, tf.shape(self.__advantages), tf.shape(self.__log_probs), tf.shape(self.__hoge)], feed_dict=feed_dict)




## todo have tensorflow, python, gym_version

## copyright
def prepro(I):
  """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
  I = I[35:195] # crop
  I = I[::2,::2,0] # downsample by factor of 2
  I[I == 144] = 0 # erase background (background type 1)
  I[I == 109] = 0 # erase background (background type 2)
  I[I != 0] = 1 # everything else (paddles, ball) just set to 1
  return I.astype(np.float).ravel().reshape(image_size, 1)


def play_episode(sess, agent, env):
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

        action = agent.act(sess, x)
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

    num_episodes_per_batch = 1
    model = PolicyGradientModel()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state('./model')
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
                observations, rewards, actions = play_episode(sess, model, env)
                processed_rewards = process_rewards(rewards)
                batch_observations.extend(observations)
                batch_rewards.extend(processed_rewards)
                batch_actions.extend(actions)
            _, summary, loss, logprogs, hoge, probs, a_shape, p_shape, h_shape = model.train(sess, np.vstack(batch_observations).T, np.vstack(actions).T, np.vstack(batch_rewards).T)
            print(a_shape, p_shape, h_shape)
            if j % 5 == 0:
                print("saved model")
                saver.save(sess, './model/model.ckpt', global_step=j)
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