# ReinforceNet.py
# Aviva Blonder
# A neural network model that learns through reinforcement learning.

import random
import math


# the overarching neural network class
## interacts with the pool class so you don't have to
## organizes and runs all of the neural network functions
class neuralnet:
    # constructor that creates intputpool and outputpool and sets lrate given...
    ## lrate - learning rate parameter
    ## topography - a list of the number of nodes of each pool in order from the input to output
    ## seed - a number to use to seed random
    def __init__(self, lrate, topography, seed=None):
        # set learning rate parameter
        self.lrate = lrate
        # seed random
        random.seed(seed)
        # use the pool class constructor to set the input pool and create the network
        ## this pool will recieve the state of the environment as input
        self.inputpool = pool(0, topography)
        # access the output pool that was just created
        self.outputpool = self.inputpool.getLast()

    # runs the neural network on a state of the environment
    ## state - a list of floats corresponding to some state of the environment
    ## returns the index of the node with the highest activation in the output layer
    def feedforward(self, state):
        # sets the activation of the nodes in the input layer to the state of the environment
        self.inputpool.activation = state
        # feeds activation forward from the input pool and returns the index of the chosen action
        self.action = self.inputpool.feedforward()
        returnme = self.action
        self.action = self.action.index(max(self.action))
        return returnme

    # backpropagates on the outcome of the action
    ## reward - actual outcome of taking the action
    def backpropagate(self, reward):
        # backpropagate starting from the output layer
        self.outputpool.backprop(self.action, reward, self.lrate)


# the structure of the neural network defined recursively
## this will handle all of the neural network opperations
## ie. feedforward activaion and backpropagation of error
class pool:
    # constructor that creates this pool and initializes all of the instance variables given...
    ## line - the index of topology designating the number of nodes in this pool
    ## topography - a list of the number of nodes in each pool in the network
    def __init__(self, line, topography):
        # set up all o fthe instance variables for use
        self.activation = []  # list of the activation of each node in pool
        self.prevactivation = []  # list of the previous activation of each node for use in backpropagation
        self.blame = []  # list of all of the blame attributed to each node in pool
        self.bias = []  # list of bias of each node in pool
        self.nextpool = None  # pool projected to
        self.outweights = []  # a list of lists of weights from each node in this pool too each node in nextpool
        self.prevpool = None  # pool that this is recieving projections from
        self.inweights = []  # a list of lists of weights from each node in prevpool to each node in this pool

        # initialize the values of all of the instance variables for each node
        for node in range(0, topography[line]):
            # set this node's activation to 0
            self.activation.append(0)
            # set this node's previous activation to 0
            self.prevactivation.append(0)
            # set this node's blame to 0
            self.blame.append(0)
            # start this node with random bias
            self.bias.append(random.random())
            # if this is not the output pool, add a row of outweights from this node
            if line < len(topography) - 1:
                # initialize the row of weights
                w = []
                # loop through all of the nodes in the next layer and create a random weight to each
                for n in range(0, topography[line + 1]):
                    # create a random weight between node and n
                    w.append(random.random())
                # add the new row of weights from this node to outweights
                self.outweights.append(w)
        # if this isn't the output pool, create the next pool using the next line of topography
        if line < len(topography) - 1:
            self.nextpool = pool(line + 1, topography)
            # set the next pool's prevpool to this
            self.nextpool.prevpool = self
            # set the next pool's incoming weights to this pool's outgoing weights
            self.nextpool.inweights = self.outweights

    # returns the output pool for the superclass
    def getLast(self):
        # if this is the output pool, return this pool
        if self.nextpool == None:
            return self
        # otherwise, try the next pool
        return self.nextpool.getLast()

    # feeds activation forward from this pool
    ## returns the index of the most strongly activated node in the outputpool
    def feedforward(self):
        for n in range(0, len(self.activation)):
            # if this is not the input pool
            if self.prevpool != None:
                # add bias to the activation of each node in the pool and take the sigmoid
                self.prevactivation[n] = 1 / (1 + math.exp(-(self.activation[n] + self.bias[n])))
            # if this is the input pool, just set prevactivation to activation
            else:
                self.prevactivation[n] = self.activation[n]
            self.activation[n] = 0
        # if this is the output pool return the activations of all of the nodes for Q-learning
        if self.nextpool == None:
            return self.prevactivation
            # chooses the most strongly activated node
        ##            best = self.prevactivation[0]
        ##            bestn = 0
        ##            for n in range(1 , len(self.activation)):
        ##                if self.prevactivation[n] > best:
        ##                    best = self.prevactivation[n]
        ##                    bestn = n
        ##            return bestn

        # otherwise use the activation of each node to contribute to the activation of the next pool
        else:
            # loop through each node in this pool and use it to activate the next pool
            for n in range(0, len(self.activation)):
                # loop through each node in next pool and add this node's weighted activation to it
                for nextn in range(0, len(self.nextpool.activation)):
                    self.nextpool.activation[nextn] += self.prevactivation[n] * self.outweights[n][nextn]
            # feedforward from the next pool and return the results
            return self.nextpool.feedforward()

    # calculates the probability of each action given the environment using the softmax algorithm
    ## activation - an array of the activations of each node in the output pool
    ## returns the index of the action chosen randomly using the probabilities
    def softmax(activation):
        pA = []  # list of probabilities of each action being the best
        total = 0  # sum of all of the expected reward for each action
        # sum up e^expected reward for each action
        for n in range(0, len(activation)):
            # add e^activation of this node to pA to be turned into a probability
            pA.append(math.exp(-activation[n]))
            # also add it to total
            total += pA[n]
        # if the total activation is zero, choose an action at random
        if total == 0:
            return random.randint(0, len(activation) - 1)
        # choose a random number between 0 and 1 and use it to choose an action
        choice = random.random()
        # the sum of the probabilities so far
        t = 0
        # loop through the possible actions again and calculate the probabilities
        ## then use those probabilities to choose an action
        for n in range(0, len(pA)):
            # calculate the probability
            pA[n] = pA[n] / total
            # add it to the total
            t += pA[n]
            # if the random choice is within the range of this action, choose it
            if choice <= t:
                return n


    # provides feedback backwards from the chosen action in the output pool
    ## node - the index of the chosen action
    ## error - the squared difference between the expected and acual reward
    def backpropNode(self, node, error, lrate):
        # adjust bias
        self.bias[node] += lrate * error
        # loop through each node in the previous pool to calculate the error on that weight and ascribe blame
        for prevn in range(0, len(self.prevpool.prevactivation)):
            # attribute blame to that node in accordance with the weight to this one
            self.prevpool.blame[prevn] += self.inweights[prevn][node] * error
            # adjust the weight from that node to this one
            dW = lrate * self.prevpool.prevactivation[prevn] * error
            ##            print(dW)
            self.inweights[prevn][node] += dW


    # general purpose backpropagation function that backpropagates from all nodes in this pool
    def backpropagate(self, lrate):
        # backpropagate on each node in the pool
        for node in range(len(self.activation)):
            # if this is not the input pool, backpropagate
            if self.prevpool != None:
                # calculate the overall feedback to this node
                feedback = self.prevactivation[node] * (1 - self.prevactivation[node]) * self.blame[node]
                # backpropagate on this node
                self.backpropNode(node, feedback, lrate)
            # either way reset activation and blame for the next run
            self.blame[node] = 0
        # if this isn't the input pool backpropagate on the previous pool
        if self.prevpool != None:
            self.prevpool.backpropagate(lrate)


    # backpropagate on the output layer using the reward provided
    def backprop(self, action, reward, lrate):
        # calculate error based on the difference between the expected and actual reward
        error = reward - self.prevactivation[action]
        # loop through all the nodes in this pool to backpropagate from each of them
        for node in range(0, len(self.activation)):
            # if this is the action that was chosen, use error
            if node == action:
                feedback = self.prevactivation[node] * (1 - self.prevactivation[node]) * error
                # otherwise use -error to settle on one action
                ##            else:
                ##                feedback = -1*self.prevactivation[node]*(1-self.prevactivation[node])*error
                # backpropagate!
                self.backpropNode(node, feedback, lrate)
        # backpropagate on the next pool
        self.prevpool.backpropagate(lrate)


# what I've been using to test the neural network and see what it's doing
def main():
    nn = neuralnet(.05, [2, 2])
    for i in range(0, 10):
        print()
        act = nn.feedforward([1, 0])
        print(act)
        if act == 0:
            nn.backpropagate(1)
        else:
            nn.backpropagate(-1)
        act = nn.feedforward([0, 1])
        print(act)
        if act == 1:
            nn.backpropagate(1)
        else:
            nn.backpropagate(-1)
            ##        act = nn.feedforward([1, 1])
            ##        print(act)
            ##        if act == 0:
            ##            nn.backpropagate(1)
            ##        else:
            ##            nn.backpropagate(0)
            ##        act = nn.feedforward([0, 0])
            ##        print(act)
            ##        if act == 0:
            ##            nn.backpropagate(1)
            ##        else:
            ##            nn.backpropagate(0)