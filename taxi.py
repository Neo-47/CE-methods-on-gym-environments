import gym
import numpy as np 
from itertools import groupby
import pickle


class Cross_Entropy:

	def __init__(self, env):

		self.env = env
		self.n_states = env.observation_space.n
		self.n_actions = env.action_space.n


	def generate_session(self, policy, t_steps = 10**4):

		states, actions = [], []
		total_reward = 0.0

		s = self.env.reset()

		for t in range(t_steps):

			#sample action from policy
			a = np.random.choice(self.n_actions, 1, p = policy[s])[0]

			#apply that action
			new_s, r, done, info = self.env.step(a)

			states.append(s)
			actions.append(a)
			total_reward += r

			s = new_s

			if(done):
				break

		return states, actions, total_reward



	def select_elites(self, states_batch, actions_batch, rewards_batch, percentile = 50):


		#pth percent of data are less than reward_threshold
		reward_threshold = np.percentile(rewards_batch, percentile)

		#get states and associated action with rewards >= reward threshold
		elite_states = [s for i in range(len(states_batch)) if rewards_batch[i] >= reward_threshold for s in states_batch[i]]
		elite_actions = [a for i in range(len(actions_batch)) if rewards_batch[i] >= reward_threshold for a in actions_batch[i]]
		
	
		return elite_states, elite_actions	


	def update_policy(self, elite_states, elite_actions):

		new_policy = np.zeros([self.n_states, self.n_actions])

		for s in range(self.n_states):

			if(s in elite_states):

				num_visits = elite_states.count(s)
				list_actions = [elite_actions[i] for i in range(len(elite_actions)) if elite_states[i] == s]

				list_actions.sort()

				set_of_actions = set(list_actions)
				freq = [len(list(group)) for key, group in groupby(list_actions)]

				for i in range(len(set_of_actions)):
					new_policy[s][list(set_of_actions)[i]] = freq[i] / num_visits


			else:
				new_policy[s] = 1 / self.n_actions

		return new_policy


	def loop(self, n_sessions, percentile, learning_rate):

		policy = np.ones([self.n_states, self.n_actions]) / self.n_actions

		for i in range(100):

			sessions = [self.generate_session(policy) for i in range(n_sessions)]

			states_batch, actions_batch, rewards_batch = zip(*sessions)

			elite_states, elite_actions = self.select_elites(states_batch, actions_batch, rewards_batch, percentile)

			new_policy = self.update_policy(elite_states, elite_actions)

			policy = learning_rate * new_policy + (1 - learning_rate) * policy

			print(i)

		return policy


if __name__ == "__main__":

	
	n_sessions = 250
	percentile = 50
	learning_rate = 0.5

	#create environment
	env = gym.make("Taxi-v2")

	CEM = Cross_Entropy(env)
	policy = CEM.loop(n_sessions, percentile, learning_rate)


	print("TRY THE NEW POLICY")


	#save policy
	with open("policy", "wb") as fb:
		pickle.dump(policy, fb)

	
	#try out the policy
	s = env.reset()

	for t in range(200):


		a = np.argmax(policy[s])

		env.render()

		new_s , r, done, info = env.step(a)

		print(r)

		s = new_s

		if(done):
			break





