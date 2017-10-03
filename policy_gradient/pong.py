import gym
import numpy as np
import tensorflow as tf

num_hidden_layer_size = 200
image_height = 80
image_width = 80
image_size = 80 * 80


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
        log_probs = tf.log(A2)

        ## reward for each action take for input
        ## shape=[1, batch_size]
        self.__advantages = tf.placeholder(tf.float32, shape=[None], name="advantages")

        ## this is negative of expected total reward
        ## todo: confirm if reduce_sum is right way
        loss = tf.reduce_sum(-log_probs * self.__advantages)

        optimizer = tf.train.AdamOptimizer(learning_rate=0.005)
        self.__train = optimizer.minimize(loss)

    def act(self, sess, observation):
        sampled_prob = sess.run(self.__probs, {self.__X: observation})
        ## todo breaking abstraction here :)
        action = 2 if np.random.uniform() < sampled_prob else 3  # roll the dice!
        return action

    def train(self, sess, observations, actions, rewards):
        feed_dict = { self.__X: observations,
                self.__advantages: rewards }
        return sess.run(self.__train, feed_dict=feed_dict)



env = gym.make("Pong-v0")
observation = env.reset()
print(tf.__version__)
#print(observation)


observation = env.reset()

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
    prev_x = None
    while not done:
        cur_x = prepro(observation)
        x = cur_x - prev_x if prev_x is not None else np.zeros([image_size, 1])
        prev_x = cur_x

        action = agent.act(sess, x)
        ## todo
        print(action)
        observations.append(cur_x.reshape(-1, 6400))
        rewards.append(reward)

        observation, reward, done, _ = env.step(action)
    return observations, rewards


with tf.Session() as sess:
    model = PolicyGradientModel()
    sess.run(tf.global_variables_initializer())

    print(model)

    #with tf.Session() as sess:
#    sess.run(tf.global_variables_initializer())
#    sess.run(tf.local_variables_initializer())
    #sess.run(tf.initialize_all_variables())

#    observation = env.reset()
#    print(env.env.action_space)
#    observations = []
#    rewards = []
    observations, rewards = play_episode(sess, model, env)
    print(model.train(sess, np.vstack(observations).T, [], rewards))


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