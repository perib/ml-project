import gym
import reinforcenet
import neuralnet
from gym import envs
import random
import numpy as np
import copy

gamma = .99
learning_rate = 0.1
numActions = 2

#potato

def run():
   # env = gym.make('LunarLander-v2')
    env = gym.make('CartPole-v1')

#old values that got some learns
#    batch_size = 32 #number of memories to learn from
#    buffer_size = 1000 #how many memories are stored in one batch
#    startE = 1  # Starting chance of random action
#    endE = 0#.1  # Final chance of random action
#    anneling_steps = 10000.  # How many steps of training to reduce startE to endE.
#    num_episodes = 100000  # How many episodes of game environment to train network with.
#    pre_train_steps = 10000  # How many steps of random actions before training begins.
#    update_freq = 5000 #train the network after this many episodes



    batch_size = 30 #number of memories to learn from
    buffer_size = 50000 #how many memories are stored in one batch
    startE = 1  # Starting chance of random action
    endE = 0.01 #.1  # Final chance of random action
    anneling_steps = 10000.  # How many steps of training to reduce startE to endE.
    num_episodes = 30000  # How many episodes of game environment to train network with.
    pre_train_steps = 10000  # How many steps of random actions before training begins.
    update_freq = 4 #train the network after this many episodes

    currentBest = 0
    preStepcount = 0
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

    brain = reinforcenet.neuralnet(0.1, [5,10,1], random.randint(0, 20000))


    mybuffer = experience_buffer(buffer_size) #creates a place to store memories


    env.monitor.start(filename, force=True)
    total = 0
    for i_episode in range(num_episodes):
        observation = env.reset()

        if(i_episode % 500 == 0):
            print(i_episode)
            print("****\ntotal reward is %s\n****" % total)
            currentBest = 0

        total = 0
        for t in range(10000):
         #   env.render()
        #    print(observation)


            if np.random.rand(1) < e or preStepcount< pre_train_steps:#chooses a random example or finds the action with the best Q1
                action = env.action_space.sample()
                preStepcount=+1
            else:
                action, Calcreward = maxQ(copy.deepcopy(observation),numActions,brain)


            observation2, reward, done, info = env.step(action) #takes a step

            total = total + reward
            if(total>currentBest):
                currentBest = total

            temp = [0,0,0,0,0]
            temp[0] = copy.deepcopy(observation)
            temp[1] = action
            temp[2] = reward
            temp[3] = copy.deepcopy(observation2) #the current environment after taking a step
            temp[4] = done #true if the game ended (pole fell over)


            mybuffer.add(temp) #stores the memory

            observation = copy.deepcopy(observation2)

       #     print("action is %s" %action)

            if done:
                if e > endE and preStepcount >= pre_train_steps:
                    e = e - stepDrop
              #  print("Episode finished after {} timesteps".format(t+1))

                break

        if(i_episode% update_freq == 0 and len(mybuffer.buffer) > batch_size):
            train_on_batch(mybuffer,brain,batch_size)


    print("****\ntotal reward is %s\n****" % total)
    env.monitor.close()

    gym.upload('/home/pedro/Desktop/ML/results/cartpole-experiment-5',api_key='sk_g4guzFpcSQSxIv1VsL8Xsw')

#trains the network on a set of instances all at once
def train_on_batch(buffer, brain, batch_size):
    memories = buffer.sample(batch_size)
   # print('TRAINING')
    for memory in memories:
        observation = memory[0]
        action = memory[1]
        reward = memory[2]
        observation2 = memory [3]
        done = memory[4]

        state = copy.deepcopy(observation)
        state = np.append(state, 9)
        state[len(state) - 1] = action



        if (done): #if this move lead to a termination of the episode
            brain.feedforward(copy.deepcopy(state))
            brain.backpropagate(-1)
        else:
            tmpAcion, max = maxQ(copy.deepcopy(observation2), numActions, brain)
            QValue = reward + gamma * max
            brain.feedforward(copy.deepcopy(state))
            brain.backpropagate(QValue)



#retures the action with the best predicted reward
def maxQ(state2,numActions,brain):
    bestR = 0
    bestAction = 0
    state = copy.deepcopy(state2)
    state = np.append(state,9)
    for ai in range(0,numActions):
        state[len(state)-1] = ai

        Result = brain.feedforward(copy.deepcopy(state))[0]

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
    #tests()

def tests():
    env = gym.make('CartPole-v1')
    brain = reinforcenet.neuralnet(0.1, [5, 1], random.randint(0, 20000))
    observation = env.reset()

    state = observation[:]
    state = np.append(state, 9)
    state[len(state) - 1] = 1

    print(state)


    Result = brain.feedforward(copy.deepcopy(state))

    print(Result)

    Result = brain.feedforward(copy.deepcopy(state))

    print(Result)

    brain.backpropagate(1)

    Result = brain.feedforward(copy.deepcopy(state))

    print(Result)

    brain.backpropagate(1)

    Result = brain.feedforward(copy.deepcopy(state))

    print(Result)

    brain.backpropagate(1)

    Result = brain.feedforward(copy.deepcopy(state))

    print(Result)

    brain.backpropagate(1)

    Result = brain.feedforward(copy.deepcopy(state))

    print(Result)

    brain.backpropagate(1)

    Result = brain.feedforward(copy.deepcopy(state))

    print(Result)

    print(maxQ(copy.deepcopy(observation),2,brain))




main()

"""

"""
"""

"""