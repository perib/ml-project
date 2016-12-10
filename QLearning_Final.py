import gym
import reinforcenet
from gym import envs
import random
import numpy as np
import copy
import matplotlib.pyplot as plt
#Pedro Ribeiro
#Uses QLearning to solve the cartpole environment


gamma = .9
learning_rate = 0.1
numActions = 2


def run(): #Runs the neuralnetwork using q-learning on the cartpole environment.
    #env = gym.make('LunarLander-v2')
    # env = gym.make('MountainCar-v0')
    env = gym.make('CartPole-v1')

    #print(env.action_space)
    #print(env.observation_space)

    if(input("use default values? (y/n) ") == "y"):
        batch_size = 40 #number of memories to learn from
        buffer_size = 50000 #how many memories are stored in one batch
        startE = 1  # Starting chance of random action
        endE = 0.01 #.1  # Final chance of random action
        anneling_steps = 1500  # How many steps of training to reduce startE to endE.
        num_episodes = 5000  # How many episodes of game environment to train network with.
        pre_train_steps = 10000  # How many steps of random actions before training begins.
        update_freq = 1 #train the network after this many episodes
        numHiddens = 64 #Number of nodes in the hidden layer
        filename = 'results/cartpolewinner10' #where to save the environment monitor data
    else:
        batch_size = eval(input("Batch size?(default 35) ")) #number of memories to learn from
        buffer_size = eval(input("Buffer size?(default 50000) ")) #how many memories are stored in one batch
        startE = eval(input("Starting e, the probability of random action when beginning learning? (default 1) "))  # Starting chance of random action
        endE = eval(input("final e, the probability of random action after anneling steps?(default .0001) ")) #.1  # Final chance of random action
        anneling_steps = eval(input("How many episodes to reduce starting e to final e? (default 500) ")) # How many steps of training to reduce startE to endE.
        num_episodes = eval(input("Number of episodes?(default 2000) "))  # How many episodes of game environment to train network with.
        pre_train_steps = eval(input("How many pre train steps to take?(default 10000) "))   # How many steps of random actions before training begins.
        update_freq = eval(input("Update frequency, update after how many episodes? (default 1) "))  #train the network after this many episodes
        numHiddens = eval(input("How many nodes in the hidden layer? (default 32) "))#Number of nodes in the hidden layer
        filename = input("Where to save the environment monitor data? (recommended: /results) ")#where to save the environment monitor data

    savedTotalR = [] #saved total reward per episode for printing a pretty chart later on.
    preStepcount = 0 #keeps track of steps needed before training
    e = startE #chance of random action
    stepDrop = (startE - endE) / anneling_steps #updated every episode

    brain = reinforcenet.neuralnet(learning_rate, [5,numHiddens,1], random.randint(0, 20000)) #sets up our neural network
    mybuffer = experience_buffer(buffer_size) #creates a place to store memories
    env.monitor.start(filename, force=True)
    total = 0 #total reward


    for i_episode in range(num_episodes):
        observation = env.reset()
        savedTotalR.append(total)

      #  plt.plot(savedTotalR) #for plotting in real time
     #   plt.pause(0.00000001)

        if(i_episode % 100 == 0):
            print(i_episode)
            print("****\ntotal reward is %s\n****" % total)


        total = 0
        for t in range(1000):
         #   env.render()
        #    print(observation)

            if np.random.rand(1) < e or preStepcount< pre_train_steps:#chooses a random example or finds the action with the best Q1
                action = env.action_space.sample()
                preStepcount = preStepcount + 1
            else:
                action, Calcreward = maxQ(copy.deepcopy(observation),numActions,brain)

            observation2, reward, done, info = env.step(action) #takes a step

            total = total + reward


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
   # env.monitor.close()

    plt.plot(savedTotalR)


    plt.ylabel('Reward over time')
    plt.show()

    #gym.upload('/home/pedro/Desktop/ML/results/cartpole-experiment-5',api_key='sk_g4guzFpcSQSxIv1VsL8Xsw')

#trains the network on a set of instances all at once
def train_on_batch(buffer, brain, batch_size):
    memories = buffer.sample(batch_size) #get batch_size number of memories
   # print('TRAINING')
    for memory in memories: #iterate through memories
        observation = memory[0] #get the values out of memories
        action = memory[1]
        reward = memory[2]
        observation2 = memory [3]
        done = memory[4]

        state = copy.deepcopy(observation) #so that we do not alter the original memory
        state = np.append(state, action) #set the last index equal to the action
     #   state[len(state) - 1] = action

        if (done): #if this move lead to a termination of the episode, only give reward. No predicted future reward cause its over
            brain.feedforward(copy.deepcopy(state)) #Correct the predicted reward for this state
            brain.backpropagate([reward])
        else:
            tmpAcion, max = maxQ(copy.deepcopy(observation2), numActions, brain) #get the best predicted reward
            QValue = reward + gamma * max
            #print()
            #print(reward)
            #print(max)
            #print(QValue)
            if(QValue > 5000): #neural network has an overflow bug, this keeps values low to prevent that. (max reward should only be 500 any way for cartpole)
                QValue = 5000;
            if(QValue < -5000):
                QValue = -5000;
            brain.feedforward(copy.deepcopy(state)) #Correct the predicted reward for this state
            brain.backpropagate([QValue])

#retures the action with the best predicted reward
def maxQ(state2,numActions,brain):
    bestR = float("-inf") #best reward found
    bestAction = float("-inf") #best action found
    state = copy.deepcopy(state2) #so that we don't change the original memory
    state = np.append(state,9) #place holder value
    for ai in range(0,numActions): #iterate through possible actions 0,1,2...
        state[len(state)-1] = ai #add this action to the state
        Result = brain.feedforward(copy.deepcopy(state))[0] #predict reward based on current state and possible action
        if(Result > bestR):
            bestR = Result #save best reward
            bestAction = ai #save the best action
    return bestAction, bestR


# [Observations1, action, reward, observations2, terminal]
#stores the new memories and replaces old ones
class experience_buffer():

    def __init__(self, buffer_size = 1000):
        self.buffer = [] #stores objects in an arrray of length buffer_size
        self.buffer_size = buffer_size #max size of array
        self.current = 0 #current index to be replaced

    def add(self, experience):
        if len(self.buffer)  >= self.buffer_size: #if the buffer is full
            if(self.current == self.buffer_size-1): #if we are at the end of the list, replace the first item
                self.current = 0
            self.buffer[self.current] = experience #replace item
            self.current += 1 #set to replace the following number
        else:
            self.buffer.append(experience) #if not full, just add new item

    def sample(self, size): #randomly sample size number of instances
        return random.sample(self.buffer, size)



def main():
    run()
    #tests()
    #gym.upload('/home/pedro/Desktop/ML/results/cartpolewinner10', api_key ='sk_KOZ7tcB2TCCPWyOI5DGckQ')

def tests(): #misc debugging
    env = gym.make('CartPole-v1')
    brain = reinforcenet.neuralnet(0.1, [5,100, 1], random.randint(0, 20000))
    observation = env.reset()

    state = observation[:]
    state = np.append(state, 9)
    state[len(state) - 1] = 1
    state2 = [9,-34,9,999,0]
    num = 500
    print(state)
    Result = brain.feedforward(copy.deepcopy(state))
    print(Result)
    brain.backpropagate(-100)
    Result = brain.feedforward(copy.deepcopy(state))
    print(Result)
    brain.backpropagate(100)
    Result = brain.feedforward(copy.deepcopy(state))
    print(Result)

   # for i in range(2000):
   #     brain.feedforward(copy.deepcopy(state2))
    #    print(brain.feedforward(copy.deepcopy(state)))
     #   brain.backpropagate([num])

    Result = brain.feedforward(copy.deepcopy(state))
    print(Result)

main()

"""

"""
"""

"""