import gym
import reinforcenet
import neuralnet
from gym import envs
import random


#env = gym.make('LunarLander-v2')
env = gym.make('CartPole-v1')



"""
brain = reinforcenet.neuralnet(0.1,[5,1])

print("potato")
print(brain.feedforward([9,9,9,9] ))
print("potato")
"""

print(env.action_space)
#print(env.observation_space)

filename = 'results/cartpole-experiment-5'

brain = reinforcenet.neuralnet(0.1,[4,4,4,4,2], random.randint(0,20000))


env.monitor.start(filename, force=True)
total = 0
for i_episode in range(200000):
    observation = env.reset()


    if(i_episode % 1000 == 0):
        print(i_episode)
        print("****\ntotal reward is %s\n****" % total)

    total = 0
    for t in range(10000):
     #   env.render()
    #    print(observation)

       # action = env.action_space.sample()

        action = brain.feedforward(observation)



        observation, reward, done, info = env.step(action)

        total = total + reward

        if(done):
            brain.backpropagate(0)
        else:
            brain.backpropagate(reward)

   #     print("action is %s" %action)

        if done:
           # print("Episode finished after {} timesteps".format(t+1))
            break


print("****\ntotal reward is %s\n****" % total)
env.monitor.close()

#gym.upload('/home/pedro/Desktop/ML/results/cartpole-experiment-5',api_key='sk_g4guzFpcSQSxIv1VsL8Xsw')

"""

"""
"""

"""