# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 14:39:00 2016

@author: Kuldeep
"""
import numpy as np
import random
import matplotlib.pyplot as plt


n=10
arms=np.random.rand(n)

av = np.ones(n) #initialize array
counts=np.zeros(n) #stores counts of how many times we've taken a particular action
av_softmax=np.zeros(n)
av_softmax[:]=0.1 #initialize each action to have equal probability

def reward(prob):
    total=0
    for i in range(10):
        if random.random()<prob:
            total+=1
    return total
    
tau=1.12 #tau was selected

def softmax(av):
    probs=np.zeros(n)
    for i in range(n):
        softm = ( np.exp(av[i] / tau) / np.sum( np.exp(av[:] / tau) ) )
        probs[i]=softm
    return probs
    
plt.xlabel("Plays")
plt.ylabel("Mean Reward")

for i in range(500):
    #select random arm using weighted probability distribution
    choice = np.where(arms == np.random.choice(arms, p=av_softmax))[0][0]
    counts[choice] += 1
    k = counts[choice]
    rwd =  reward(arms[choice])
    old_avg = av[choice]
    new_avg = old_avg + (1/k)*(rwd - old_avg)
    av[choice] = new_avg
    av_softmax = softmax(av) #update softmax probabilities for next play
    runningMean = np.average(av, weights=np.array([counts[i]/np.sum(counts) for i in range(len(counts))]))
    plt.scatter(i,runningMean)



