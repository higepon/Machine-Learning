import gym
import numpy as np
import tensorflow as tf

num_hidden_layer_size = 200
image_height = 80
image_width = 80
image_size = 80 * 80


class PolicyGradientModel:
    def __init__(self):
        self.__X = tf.placeholder(tf.float32, [image_size, None])

        W1 = tf.Variable(tf.random_normal([num_hidden_layer_size, image_size]), name="W11")
        W2 = tf.Variable(tf.random_normal([1, num_hidden_layer_size]), name="W2A")

        # activation of hidden layer
        Z1 = tf.matmul(W1, self.__X)
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
        advantages = tf.placeholder(tf.float32, shape=[None])

        ## this is negative of expected total reward
        ## todo: confirm if reduce_sum is right way
        loss = tf.reduce_sum(-log_probs * advantages)

        optimizer = tf.train.AdamOptimizer(learning_rate=0.005)
        minimize = optimizer.minimize(loss)

    def sample(self, sess2, x):
        return sess2.run(self.__probs, {self.__X: x})



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


prev_x = None
with tf.Session() as sess:
    model = PolicyGradientModel()
    sess.run(tf.global_variables_initializer())

    print(model)

    #with tf.Session() as sess:
#    sess.run(tf.global_variables_initializer())
#    sess.run(tf.local_variables_initializer())
    #sess.run(tf.initialize_all_variables())


    while True:
        # todo initializer, w1 random
        # get input
        observation = env.reset()
        cur_x = prepro(observation)
        x = cur_x - prev_x if prev_x is not None else np.zeros([image_size, 1])
        prev_x = cur_x
        print(model.sample(sess, x))
        # sample prob

        # decide action

    # accumulate reward

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