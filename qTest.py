import gym
import reinforcenet
import neuralnet
from gym import envs
import random
import numpy as np

def main():
    #env = gym.make('LunarLander-v2')
    env = gym.make('CartPole-v1')
    gamma = .7
    numActions = 2
    probOfRandom = 30
    startE = 1  # Starting chance of random action
    endE = 0.1  # Final chance of random action
    anneling_steps = 10000.  # How many steps of training to reduce startE to endE.
    num_episodes = 10000  # How many episodes of game environment to train network with.
    pre_train_steps = 10000  # How many steps of random actions before training begins.

    e = startE
    stepDrop = (startE - endE) / anneling_steps


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


    env.monitor.start(filename, force=True)
    total = 0
    for i_episode in range(1):
        observation = env.reset()


        if(i_episode % 1000 == 0):
            print(i_episode)
            print("****\ntotal reward is %s\n****" % total)

        total = 0
        for t in range(1):
         #   env.render()
        #    print(observation)

           # action = env.action_space.sample()


            if np.random.rand(1) < e:
                action = np.random.randint(0,numActions)
            else:
                action, Calcreward = maxQ(observation,numActions,brain)



            observation2, reward, done, info = env.step(action)

            tmpAcion, max =  maxQ(observation2,numActions,brain)
            QValue = reward + gamma*max



            total = total + reward

            neuralnet.forward_propagate(brain, observation)

            if(done):
                neuralnet.backward_propagate_error(brain, [0])
            else:
                neuralnet.backward_propagate_error(brain, [QValue])

       #     print("action is %s" %action)

            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break


    print("****\ntotal reward is %s\n****" % total)
    env.monitor.close()

    gym.upload('/home/pedro/Desktop/ML/results/cartpole-experiment-5',api_key='sk_g4guzFpcSQSxIv1VsL8Xsw')

def maxQ(state,numActions,brain):
    bestR = 0
    bestAction = 0
    state.insert(len(state), 'TEMP')
    for ai in range(0,numActions):
        state[len(state)-1] = ai
        Result = neuralnet.forward_propagate(brain,state[:])[0]
        if(Result > bestR):
            bestR = Result
            bestAction = ai

    return bestAction, bestR

class experience_buffer():

    def __init__(self, buffer_size = 1000):
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self, experience):
        if len(self.buffer) + len(experience) >= self.buffer_size:
            self.buffer[0:(len(experience) + len(self.buffer)) - self.buffer_size] = []
        self.buffer.extend(experience)

    def sample(self, size):
        return random.sample(self.buffer, size)



"""

"""
"""

"""