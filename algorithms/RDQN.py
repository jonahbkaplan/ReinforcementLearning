import random

import torch
from algorithms.Agent import Agent
import numpy as np
from torch import nn,optim
import math
from torchrl import modules

from collections import deque
import random

class NStepReplayBuffer:
    def __init__(self, capacity, n_step, gamma):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity) # Use deque for main buffer too
        self.n_step = n_step
        self.gamma = gamma
        self.n_queue = deque()

    def push(self, S, A, R, S_next, done):
        # Add transition to local queue
        self.n_queue.append((S, A, R, S_next, done))

        # 1. Normal case: Queue is full, pop one and store
        if len(self.n_queue) >= self.n_step:
            self._process_queue_item()

        # 2. Terminal case: Flush the queue
        if done:
            while len(self.n_queue) > 0:
                self._process_queue_item()

    def _process_queue_item(self):
        # We process the oldest item in the queue
        S0, A0, _, _, _ = self.n_queue[0]
        
        # Calculate n-step discounted reward
        R_n = 0
        gamma_power = 0
        done_n = False
        S_n = self.n_queue[-1][3] # Default to S_next of newest item

        for i, (_, _, r_i, s_next_i, done_i) in enumerate(self.n_queue):
            R_n += (self.gamma ** gamma_power) * r_i
            gamma_power += 1
            if done_i:
                done_n = True
                S_n = s_next_i # The state where it actually ended
                break
        
        self.buffer.append((S0, A0, R_n, S_n, done_n))
        self.n_queue.popleft()

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class ActionValueNN(nn.Module):
    def __init__(self, state_dim, action_dim) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            modules.NoisyLinear(state_dim, 64),
            nn.ReLU(),
            modules.NoisyLinear(64, 64),
            nn.ReLU(),
            
        )
        self.value_head  = nn.Linear(64, 1)
        self.advantage_head = nn.Linear(64, action_dim)

    def forward(self, x):
        h = self.backbone(x)
        V = self.value_head(h)
        A = self.advantage_head(h)
        return V + (A - A.mean(dim=1, keepdim=True))  
   


class RDQN(Agent):
    def __init__(self, env, batch_size=64, gamma=0.99, C=1000, buffer_capacity=10000, n_step=3):
        super().__init__(env)
        self.batch_size = batch_size 
        self.gamma = gamma
        self.n_step = n_step
        self.C = C
        self.D = NStepReplayBuffer(buffer_capacity, self.n_step, self.gamma)
        
        self.state_dim = math.prod(env.unwrapped.observation_space.shape)
        self.action_dim = int(env.unwrapped.action_space.n) # Fixed attribute name

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.q_1 = ActionValueNN(self.state_dim, self.action_dim).to(self.device)
        self.q_2 = ActionValueNN(self.state_dim, self.action_dim).to(self.device)
        self.q_2.load_state_dict(self.q_1.state_dict())
        self.optimizer = optim.Adam(self.q_1.parameters(), lr=1e-3)
        self.counter = 0

    def predict(self, obs):
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.q_1(obs_t)
            action = q_values.argmax(dim=1).item()
        return action
       
    def learn(self, S, A, R, S_next):
        # FIX: Store raw data (numpy/floats), not GPU tensors, to save VRAM
        done, obs_next = S_next
        self.D.push(S, A, R, obs_next, done)

        if len(self.D) < self.batch_size:
            return

        batch = self.D.sample(self.batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)
        
        states      = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        actions     = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards     = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device)
        dones       = torch.tensor(dones, dtype=torch.float32, device=self.device)

        # Q(s,a) from online network
        q_values = self.q_1(states)
        q_pred = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Double DQN Target Logic
        with torch.no_grad():
            next_actions = self.q_1(next_states).argmax(dim=1, keepdim=True)
            q_next = self.q_2(next_states).gather(1, next_actions).squeeze(1)
            y_target = rewards + (self.gamma ** self.n_step) * (1 - dones) * q_next

        loss = nn.functional.huber_loss(q_pred, y_target)

        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients for stability
        torch.nn.utils.clip_grad_norm_(self.q_1.parameters(), 10.0)
        self.optimizer.step()

        self.counter += 1
        if self.counter % self.C == 0:
            self.q_2.load_state_dict(self.q_1.state_dict())