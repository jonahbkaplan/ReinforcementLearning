import random

import torch
from algorithms.Agent import Agent
import numpy as np
from torch import nn,optim
import math


class ActionValueNN(nn.Module):
    def __init__(self, state_dim, action_dim) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            
        )
        self.value_head  = nn.Linear(128, 1)
        self.advantage_head = nn.Linear(128, action_dim)

    def forward(self, x):
        h = self.backbone(x)
        V = self.value_head(h)
        A = self.advantage_head(h)
        return V + (A - A.mean(dim=1, keepdim=True))  
   

class RDQN(Agent):
    # Rainbow DQN
    # Discrete
    # Jonah

    def __init__(self, env,batch_size = 16, gamma = 0.99, C = 1000):
        super().__init__(env)
        self.batch_size = batch_size
        self.gamma = gamma
        self.C = C
        self.D = []
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.replay_capacity = 1000
        
        self.state_dim = math.prod(env.unwrapped.observation_space.shape)
        self.action_dim = int(env.unwrapped.action_type.space().n)
        
        self.q_1 = ActionValueNN(self.state_dim,self.action_dim)
        self.q_2 = ActionValueNN(self.state_dim,self.action_dim)
        self.q_2.load_state_dict(self.q_1.state_dict())
        self.optimizer = optim.Adam(self.q_1.parameters(), lr=1e-3)


        self.counter = 0
        
        

    def predict(self, obs):
        if random.random() < self.epsilon:              # explore
            action = random.randrange(self.action_dim)
        else:                                      # exploit
            with torch.no_grad():
                obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                q_values = self.q_1(obs_t)
                action =  q_values.argmax(dim=1).item()
        
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return action
       
    def learn(self, S, A, R, S_next):
    # Store transition
        self.D.append((S, A, R, S_next))
        if len(self.D) > self.replay_capacity:
            self.D.pop(0)

        # Not enough samples yet
        if len(self.D) < self.batch_size:
            return

        # Sample batch
        batch = random.sample(self.D, self.batch_size)

        states, actions, rewards, next_states, dones = [], [], [], [], []

        for S_j, A_j, R_j, S_next_j in batch:
            done, obs_next = S_next_j
            states.append(S_j)
            actions.append(A_j)
            rewards.append(R_j)
            next_states.append(obs_next)
            dones.append(done)

        # Convert to tensors
        states      = torch.tensor(states, dtype=torch.float32)
        actions     = torch.tensor(actions, dtype=torch.long)
        rewards     = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones       = torch.tensor(dones, dtype=torch.float32)

        # Q(s,a) from online network
        q_values = self.q_1(states)                       # [B, action_dim]
        q_pred = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # max_a Q_target(s', a)
        with torch.no_grad():
            a = self.q_1(next_states).argmax(dim=1)

            q1_eval = self.q_1(next_states).gather(1, a.unsqueeze(1)).squeeze(1)
            q2_eval = self.q_2(next_states).gather(1, a.unsqueeze(1)).squeeze(1)
            q_next = torch.min(q1_eval,q2_eval)

            y_target = rewards + self.gamma * q_next * (1 - dones)

        # Loss
        loss = nn.functional.huber_loss(q_pred, y_target)

        # Update
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        self.counter += 1
        if self.counter % self.C == 0:
            self.q_2.load_state_dict(self.q_1.state_dict())