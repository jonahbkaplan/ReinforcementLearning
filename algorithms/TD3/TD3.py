import os, sys
parent = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(parent)

from algorithms.Agent import Agent


import gymnasium
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import math
import copy


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#replay buffer - stores and smaples past experience tuples (s, a, s', r, done)
class ReplayBuffer(object):
    def __init__(self, stateDim, actionDim, maxSize = int(1000000)):
        self.maxSize = maxSize
        self.pointer = 0
        self.size = 0
        
        #preallocate memory for effiency
        self.state = np.zeros((maxSize, stateDim))
        self.action = np.zeros((maxSize, actionDim))
        self.nextState = np.zeros((maxSize, stateDim))
        self.reward = np.zeros((maxSize, 1))
        self.done = np.zeros((maxSize, 1))
        
    #adds a single transition to the buffer
    def add(self, state, action, nextState, reward, done):
        self.state[self.pointer] = state
        self.action[self.pointer] = action
        self.nextState[self.pointer] = nextState
        self.reward[self.pointer] = reward
        self.done[self.pointer] = done 
        
        #circular buffer update
        self.pointer = (self.pointer + 1) % self.maxSize
        self.size = min(self.size + 1, self.maxSize)
      
    #randomly sample a batch of transitions and return them  
    def sample(self, batchSize):
        index = np.random.randint(0, self.size, size = batchSize)
        
        state = self.state[index]
        action = self.action[index]
        nextState = self.nextState[index]
        reward = self.reward[index]
        done = self.done[index]
        
        return state, action, nextState, reward, done
      
#actor network - maps states to continuous action
class Actor(nn.Module):
    def __init__(self, stateDim, actionDim, maxAction):
        super(Actor, self).__init__()
        
        self.f1 = nn.Linear(stateDim, 256)
        self.f2 = nn.Linear(256, 256)
        self.f3 = nn.Linear(256, actionDim)
        
        self.maxAction = maxAction
        
    #forward pass through policy network
    def forward(self, state):
        a = F.relu(self.f1(state))
        a = F.relu(self.f2(a))
        #bounds action between [-1,1]
        a = torch.tanh(self.f3(a))

        #scales action to environment limits
        return self.maxAction * a
  
#critic network - twin Q-networks  
class Critic(nn.Module):
    def __init__(self, stateDim, actionDim):
        super(Critic, self).__init__()
        
        #Q1 network
        self.f1 = nn.Linear(stateDim + actionDim, 256)
        self.f2 = nn.Linear(256, 256)
        self.f3 = nn.Linear(256, 1)
        
        #Q2 network
        self.f4 = nn.Linear(stateDim + actionDim, 256)
        self.f5 = nn.Linear(256, 256)
        self.f6 = nn.Linear(256, 1)
        
    def forward(self, state, action):
        #concatenates state and action
        sa = torch.cat([state, action], 1)
        
        #Q1 computation
        q1 = F.relu(self.f1(sa))
        q1 = F.relu(self.f2(q1))
        q1 = self.f3(q1)
        
        #Q2 computation
        q2 = F.relu(self.f4(sa))
        q2 = F.relu(self.f5(q2))
        q2 = self.f6(q2)
    
        return q1, q2
    
    #returns only Q1 - used for actor update
    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)
        
        q1 = F.relu(self.f1(sa))
        q1 = F.relu(self.f2(q1))
        q1 = self.f3(q1)
        
        return q1
    
#Twin Delayed Deep Deterministic Policy Gradient (TD3)
class TD3(Agent):
    def __init__(self, env, gamma=0.99, tau=0.005, policyNoise=0.2, noiseClip=0.5, policyFreq=2):
        super().__init__(env)
        
        #establish hyperparameters
        
        #discount factor
        self.gamma = gamma
        #target network update rate
        self.tau = tau
        #target policy smoothing noise
        self.policyNoise = policyNoise
        self.noiseClip = noiseClip
        #delays actor updates
        self.policyFreq = policyFreq
        
        #handles highway environment's state dimensions
        if isinstance(env.unwrapped.observation_space,gymnasium.spaces.dict.Dict):
            self.stateDim = sum(math.prod(obs.shape) for obs in env.unwrapped.observation_space.values())
        else:
            self.stateDim = math.prod(env.unwrapped.observation_space.shape)
         
        #set up action space
        actionSpace = env.action_space 
        self.actionDim = actionSpace.shape[0]
        self.maxAction = float(actionSpace.high[0])
        
        self.totalIterations = 0
        
        #establish actor network and target actor network with same parameters
        self.actor = Actor(self.stateDim, self.actionDim, self.maxAction).to(device)
        self.actorTarget = copy.deepcopy(self.actor)
        self.actorOptimiser = torch.optim.Adam(self.actor.parameters(), lr = 3e-4)
        
        #establish critic network and target critic network with same parameters
        self.critic = Critic(self.stateDim, self.actionDim). to(device)
        self.criticTarget = copy.deepcopy(self.critic)
        self.criticOptimiser = torch.optim.Adam(self.critic.parameters(), lr = 3e-4)
        
        #set up relay buffer
        self.replayBuffer = ReplayBuffer(self.stateDim, self.actionDim)
        self.batchSize = 256
        
    #use actor network to select an action
    def selectAction(self, state, explore=True):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        action = self.actor(state).cpu().data.numpy().flatten()
    
        #add exploration noise during training
        if explore:
            noise = np.random.normal(0, self.policyNoise, size = self.actionDim)
            action = action + noise

        return np.clip(action, -self.maxAction, self.maxAction)
    
    #perform one TD3 update step
    def learn(self, S, A, R, SNext):
        
        done, nextState = SNext
        self.replayBuffer.add(S, A, nextState, R, done)
        
        #wait until replay buffer is sufficiently populated with past experiences
        if self.replayBuffer.size < 256:
            return
        self.totalIterations += 1
        
        #sample batch from replay buffer
        state, action, nextState, reward, done = self.replayBuffer.sample(self.batchSize)
        
        #convert to tensors
        state = torch.FloatTensor(state).to(device)
        action = torch.FloatTensor(action).to(device)
        nextState = torch.FloatTensor(nextState).to(device)
        reward = torch.FloatTensor(reward).to(device)
        done = torch.FloatTensor(done).to(device)
        
        #critic update
        with torch.no_grad():
            
            #target policy smoothing
            standardNoise = torch.randn_like(action)
            standardNoise *= self.policyNoise
            standardNoise = torch.clip(standardNoise, -self.noiseClip, self.noiseClip)
            
            nextAction = self.actorTarget(nextState) + standardNoise
            nextAction = nextAction.clip(-self.maxAction, self.maxAction)
            
            #target Q-value using minimum of twin critics
            targetQ1, targetQ2 = self.criticTarget(nextState, nextAction)
            targetQ = torch.min(targetQ1, targetQ2)
            targetQ = reward + self.gamma * (1 - done) * targetQ
            
        currentQ1, currentQ2 = self.critic(state, action)
        criticLoss = F.mse_loss(currentQ1, targetQ) + F.mse_loss(currentQ2, targetQ)
        
        self.criticOptimiser.zero_grad()
        criticLoss.backward()
        self.criticOptimiser.step()
        
        #delayed actor update
        if self.totalIterations % self.policyFreq == 0:
            
            actorLoss = -self.critic.Q1(state, self.actor(state)).mean()
            self.actorOptimiser.zero_grad()
            actorLoss.backward()
            self.actorOptimiser.step()
            
            criticDict = self.critic.state_dict()
            criticTargetDict = self.criticTarget.state_dict()
            
            #update target networks
            for key in criticTargetDict:
                criticTargetDict[key] = self.tau * criticDict[key] + (1 - self.tau) * criticTargetDict[key] 
            self.criticTarget.load_state_dict(criticTargetDict)
                
            actorDict = self.actor.state_dict()
            actorTargetDict = self.actorTarget.state_dict()
            
            for key in actorTargetDict:
                actorTargetDict[key] = self.tau * actorDict[key] + (1 - self.tau) * actorTargetDict[key] 
            self.actorTarget.load_state_dict(actorTargetDict)
                  
    #save model
    def save(self, fileName):
        torch.save(self.critic.state_dict(), fileName + "_critic")
        torch.save(self.criticOptimiser.state_dict(), fileName + "_criticOptimiser")
        
        torch.save(self.actor.state_dict(), fileName + "_actor")
        torch.save(self.actorOptimiser.state_dict(), fileName + "_actorOptimiser")
        
    #load model
    def load(self, fileName):
        self.critic.load_state_dict(torch.load(fileName + "_critic"))
        self.criticOptimiser.load_state_dict(torch.load(fileName + "_criticOptimiser"))
        self.criticTarget = copy.deepcopy(self.critic)
        
        self.actor.load_state_dict(torch.load(fileName + "_actor"))
        self.actorOptimiser.load_state_dict(torch.load(fileName + "_actorOptimiser"))
        self.actorTarget = copy.deepcopy(self.actor)
        


    