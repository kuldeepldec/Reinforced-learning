# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 10:18:22 2016

@author: dimira
"""

import numpy as np
from scipy import stats
import random
import matplotlib.pyplot as plt

n=10  #slot machine
arms=np.random.rand(n) #generating random number
eps=0.5

print eps
def reward(prob): #With an arm probability of 0.7, the average reward of doing this to infinity would be 7
    reward=0
    for i in range(10):
        if random.random()<prob: #reward if a random float is less than the arm's probability
            reward +=1
    return reward
    
#initialize memory array; has 1 row defaulted to random action index
av=np.array([np.random.randint(0,(n+1)),0]).reshape(1,2)

def bestArm(a):
    bestArm=0
    bestMean=0
    for u in a:
        avg=np.mean(a[np.where(a[:,0]==u[0])][:,1]) #calc mean reward for each action
        if bestMean<avg:
            bestMean=avg
            bestArm=u[0]
    return bestArm
    
plt.xlabel("Plays")
plt.ylabel("Avg Reward")

for i in range(500):
    if random.random()>eps:
        if random.random() > eps:
            choice=bestArm(av)
            thisAV = np.array([[choice, reward(arms[choice])]])
            av = np.concatenate((av, thisAV), axis=0)
            
    else:
        choice = np.where(arms == np.random.choice(arms))[0][0]
        thisAV = np.array([[choice, reward(arms[choice])]]) #choice, reward
        av = np.concatenate((av, thisAV), axis=0) #add to our action-value memory array
        
    percCorrect = 100*(len(av[np.where(av[:,0] == np.argmax(arms))])/len(av))
    runningMean = np.mean(av[:,1])
    plt.scatter(i, runningMean)
