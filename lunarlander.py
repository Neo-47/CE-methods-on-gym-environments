import gym
import numpy as np
import pickle
import gym.wrappers
from sklearn.neural_network import MLPClassifier



class Cross_Entropy:

    def __init__(self, env):

        self.env = env
        self.n_actions = env.action_space.n

        self.agent = MLPClassifier(hidden_layer_sizes = (30, 30),
                            activation = 'tanh',
                            warm_start = True,
                            max_iter = 5)

        #initialize agent to dimension of states and actions
        self.agent.fit([env.reset()] * self.n_actions, list(range(self.n_actions)))


    def generate_session(self, t_steps = 10**4):

        states, actions = [], []
        total_reward = 0.0

        s = self.env.reset()

        for t in range(t_steps):

            #A vector of action probabilities in current state
            probs = self.agent.predict_proba([s])[0]


            #sample action from policy
            a = np.random.choice(self.n_actions, p = probs)
            
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
        elite_states = [s for i in range(len(states_batch)) if rewards_batch[i] > reward_threshold for s in states_batch[i]]
        elite_actions = [a for i in range(len(actions_batch)) if rewards_batch[i] > reward_threshold for a in actions_batch[i]]
        
    
        return elite_states, elite_actions  



    def loop(self, n_sessions, percentile):

        for i in range(200):

            sessions = [self.generate_session() for _ in range(n_sessions)]

            batch_states,batch_actions,batch_rewards = map(np.array, zip(*sessions))
            

            elite_states, elite_actions = self.select_elites(batch_states, batch_actions, batch_rewards, percentile)
    
            self.agent.fit(elite_states, elite_actions)

            print(i, np.mean(batch_rewards))



            if(np.mean(batch_rewards) > 50):

                print("YOU WIN")
                return self.agent




if __name__ == "__main__" :

    
    env = gym.make("LunarLander-v2") 


    
    n_sessions = 100
    percentile = 70

    cem = Cross_Entropy(env)
    agent = cem.loop(n_sessions, percentile)

    print("TRY THE AGENT")

    
    #save the agent
    with open("lunarlander agent", "wb") as fb:
        pickle.dump(agent, fb)

    
    #to recor a video
    env = gym.wrappers.Monitor(gym.make("LunarLander-v2"), directory = "videos", force = True)

    s = env.reset()

    #Try out the agent
    for i in range(200):

        probs = agent.predict_proba([s])[0]

        a = np.argmax(probs)

        env.render()

        new_s , r, done, info = env.step(a)


        s = new_s

        if(done):
            break
    
    
    