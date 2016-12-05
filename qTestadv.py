import gym
import reinforcenet
import neuralnet
from gym import envs
import random
import numpy as np

gamma = .7
learning_rate = 0.1
numActions = 2

#potato

def run():
    #env = gym.make('LunarLander-v2')
    env = gym.make('CartPole-v1')

    batch_size = 500 #number of memories to learn from
    buffer_size = 5000 #how many memories are stored in one batch
    startE = 1  # Starting chance of random action
    endE = 0#.1  # Final chance of random action
    anneling_steps = 10000.  # How many steps of training to reduce startE to endE.
    num_episodes = 10000  # How many episodes of game environment to train network with.
    pre_train_steps = 10000  # How many steps of random actions before training begins.
    update_freq = 5000 #train the network after this many episodes

    e = startE #chance of random action
    stepDrop = (startE - endE) / anneling_steps #updated every episode


    """
    brain = reinforcenet.neuralnet(0.1,[5,1])

    print("potato")
    print(brain.feedforward([9,9,9,9] ))
    print("potato")
    """

    #print(env.action_space)
    #print(env.observation_space)

    filename = 'results/cartpole-experiment-5'

    brain = neuralnet.initialize_network(4,4,1)


    mybuffer = experience_buffer(buffer_size) #creates a place to store memories


    env.monitor.start(filename, force=True)
    total = 0
    for i_episode in range(100000):
        observation = env.reset()

        if(i_episode % 1000 == 0):
            print(i_episode)
            print("****\ntotal reward is %s\n****" % total)

        total = 0
        for t in range(10000):
         #   env.render()
        #    print(observation)


            if np.random.rand(1) < e:#chooses a random example or finds the action with the best Q1
                action = env.action_space.sample()
            else:
                action, Calcreward = maxQ(observation[:],numActions,brain)


            observation2, reward, done, info = env.step(action) #takes a step


            temp = [0,0,0,0,0]
            temp[0] = observation
            temp[1] = action
            temp[2] = reward
            temp[3] = observation2 #the current environment after taking a step
            temp[4] = done #true if the game ended (pole fell over)


            mybuffer.add(temp) #stores the memory



            """
           tmpAcion, max =  maxQ(observation2[:],numActions,brain)
           QValue = reward + gamma*max

           total = total + reward

           state = observation[:]
           state = np.append(state, 9)
           state[len(state) - 1] = action

           neuralnet.forward_propagate(brain, state)

           if(done):
               neuralnet.backward_propagate_error(brain, [-1000])
               neuralnet.update_weights(brain,state,learning_rate)
           else:
               neuralnet.backward_propagate_error(brain, [QValue])
               neuralnet.update_weights(brain, state, learning_rate)


           """

            observation = observation2

       #     print("action is %s" %action)

            if done:
                if e > endE:
                    e = e - stepDrop
              #  print("Episode finished after {} timesteps".format(t+1))

                break


        if(i_episode% update_freq == 0 and len(mybuffer.buffer) == buffer_size):
            train_on_batch(mybuffer,brain,batch_size)


    print("****\ntotal reward is %s\n****" % total)
    env.monitor.close()

    #gym.upload('/home/pedro/Desktop/ML/results/cartpole-experiment-5',api_key='sk_g4guzFpcSQSxIv1VsL8Xsw')

#trains the network on a set of instances all at once
def train_on_batch(buffer, brain, batch_size):
    memories = buffer.sample(batch_size)
    print('TRAINING')
    for memory in memories:
        observation = memory[0]
        action = memory[1]
        reward = memory[2]
        observation2 = memory [3]
        done = memory[4]

        state = observation[:]
        state = np.append(state, 9)
        state[len(state) - 1] = action

        neuralnet.forward_propagate(brain, state)

        if (done): #if this move lead to a termination of the episode
            neuralnet.backward_propagate_error(brain, [reward])
            neuralnet.update_weights(brain, state, learning_rate)
        else:
            tmpAcion, max = maxQ(observation2[:], numActions, brain)
            QValue = reward + gamma * max
            neuralnet.backward_propagate_error(brain, [QValue])
            neuralnet.update_weights(brain, state, learning_rate)


#retures the action with the best predicted reward
def maxQ(state,numActions,brain):
    bestR = 0
    bestAction = 0
    state = np.append(state,9)
    for ai in range(0,numActions):
        state[len(state)-1] = ai
        Result = neuralnet.forward_propagate(brain,state[:])[0]
        if(Result > bestR):
            bestR = Result
            bestAction = ai

    return bestAction, bestR


# [Observations1, action, reward, observations2, terminal]
#stores the memories and replaces old ones
class experience_buffer():

    def __init__(self, buffer_size = 1000):
        self.buffer = []
        self.buffer_size = buffer_size
        self.current = 0

    def add(self, experience):
        if len(self.buffer)  >= self.buffer_size:
            if(self.current == self.buffer_size-1):
                self.current = 0
            self.buffer[self.current] = experience
            self.current += 1
        else:
            self.buffer.append(experience)

    def sample(self, size):
        return random.sample(self.buffer, size)



def main():
    run()



main()

"""

"""
"""

"""