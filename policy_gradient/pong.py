import gym

env = gym.make("Pong-v0")
observation = env.reset()
print(observation)