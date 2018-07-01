import pickle
import gym
import numpy as np 

env = gym.make("Taxi-v2")


with open("policy", "rb") as fp:
	policy = pickle.load(fp)

s = env.reset()

for t in range(200):


	a = np.argmax(policy[s])

	env.render()

	new_s , r, done, info = env.step(a)

	print(r)

	s = new_s

	if(done):
		break