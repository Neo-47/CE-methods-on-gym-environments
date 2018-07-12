import gym
import numpy as np
import pickle
from pong import make_pong
import gym.wrappers
from keras_CNN_model import model
from keras.utils import np_utils
from keras.models import model_from_json



class Cross_Entropy:

    def __init__(self, env):

        self.env = env
        self.n_actions = env.action_space.n

        self.s = self.env.reset()
        #initialize agent to dimension of states and actions
        self.agent = model(self.s.shape)


    def generate_session(self, t_steps = 10**4):

        states, actions = [], []
        total_reward = 0.0

        s = self.env.reset()

        s = np.reshape(s, (1, 4, 42, 42))

        for t in range(t_steps):

            #A vector of action probabilities in current state
            probs = self.agent.predict(s)[0]


            #sample action from policy
            a = np.random.choice(self.n_actions, p = probs)
            
 
            #apply that action
            new_s, r, done, info = self.env.step(a)

            states.append(s)
            actions.append(a)
            total_reward += r

            s = new_s

            s = np.reshape(s, (1, 4, 42, 42))

            if(done):
                break

        states = np.array(states)

        states = np.reshape(states, (states.shape[0], 4, 42, 42))


        return states, np.array(actions), total_reward



    def select_elites(self, states_batch, actions_batch, rewards_batch, percentile = 80):


        #pth percent of data are less than reward_threshold
        reward_threshold = np.percentile(rewards_batch, percentile)

        #get states and associated action with rewards >= reward threshold
        elite_states = [s for i in range(len(states_batch)) if rewards_batch[i] > reward_threshold for s in states_batch[i]]
        elite_actions = [a for i in range(len(actions_batch)) if rewards_batch[i] > reward_threshold for a in actions_batch[i]]
        
    
        return elite_states, elite_actions  



    def loop(self, n_sessions, percentile):

        for i in range(200):

            sessions = [self.generate_session() for _ in range(n_sessions)]

            print("after sessions")
            batch_states,batch_actions,batch_rewards = map(np.array, zip(*sessions))
            
            print("after map")
            elite_states, elite_actions = self.select_elites(batch_states, batch_actions, batch_rewards, percentile)
        
            print("after elites")

            elite_states = np.array(elite_states)

            elite_actions = np.array(elite_actions)

            if(elite_states.size == 0):
                return self.agent

            print(i, np.mean(batch_rewards))

            elite_actions = np_utils.to_categorical(elite_actions)

            self.agent.fit(elite_states, elite_actions, batch_size=200, verbose=2, epochs = 10)

            
            if(np.mean(batch_rewards) > 70):

                print("YOU WIN")
                return self.agent




if __name__ == "__main__" :

    
    env = make_pong()


    n_sessions = 10
    percentile = 30

    cem = Cross_Entropy(env)
    agent = cem.loop(n_sessions, percentile)

    # to save model
    model_json = agent.to_json()

    with open("agent.json", "w") as json_file:
        json_file.write(model_json)

    # serialize weights to HDF5
    agent.save_weights("model.h5")

    print("Saved model to disk")

    # to save weights
    json_file = open('agent.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()

    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")

    print("Loaded model from disk")

    agent = loaded_model


    #to record a video

    env = gym.wrappers.Monitor(make_pong(), directory = "videos", force = True)

    s = env.reset()

    s = np.reshape(s, (1, 4, 42, 42))

    #Try out the agent
    for i in range(2000):

        probs = agent.predict(s)[0]

        a = np.argmax(probs)

        env.render()

        new_s , r, done, info = env.step(a)

        s = new_s

        s = np.reshape(s, (1, 4, 42, 42))

        if(done):
            break

    