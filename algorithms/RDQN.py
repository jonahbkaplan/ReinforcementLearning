from pathlib import Path
import random

import gymnasium
import torch
from algorithms.Agent import Agent
import numpy as np
from torch import nn, optim
import torch.nn.functional as F
import math
from torchrl import modules

from collections import deque
import random


class NStepPriorityReplayBuffer:
    def __init__(self, capacity, n_step, total_training_steps, gamma, beta=0.4):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.n_step = n_step
        self.gamma = gamma
        self.beta = beta
        self.beta_increment = (1.0 - self.beta) / total_training_steps
        self.n_queue = deque()
        self.priorities = []

    def push(self, S, A, R, S_next, done):
        self.n_queue.append((S, A, R, S_next, done))

        if len(self.n_queue) >= self.n_step:
            self._process_queue_item()

        if done:
            while len(self.n_queue) > 0:
                self._process_queue_item()

    def _process_queue_item(self):

        S0, A0, _, _, _ = self.n_queue[0]

        R_n = 0
        gamma_power = 0
        done_n = False
        S_n = self.n_queue[-1][3]

        for i, (_, _, r_i, s_next_i, done_i) in enumerate(self.n_queue):
            R_n += (self.gamma**gamma_power) * r_i
            gamma_power += 1
            if done_i:
                done_n = True
                S_n = s_next_i
                break

        self.buffer.append((S0, A0, R_n, S_n, done_n))
        max_priority = 1 if len(self.priorities) == 0 else max(self.priorities)
        if len(self.priorities) < self.capacity:
            self.priorities.append(max_priority)
        else:
            self.priorities.pop(0)
            self.priorities.append(max_priority)

        self.n_queue.popleft()

    def sample(self, batch_size):
        indices = random.choices(
            population=range(len(self.buffer)), weights=self.priorities, k=batch_size
        )

        transitions = [self.buffer[i] for i in indices]

        full_p = np.array(self.priorities) / np.sum(self.priorities)

        weights = (1.0 / (len(self.buffer) * full_p[indices])) ** self.beta
        weights /= weights.max()

        self.beta = min(1.0, self.beta + self.beta_increment)

        return indices, weights, transitions

    def __len__(self):
        return len(self.buffer)

    def update_priorities(self, idxs, priorites):
        for idx, priority in zip(idxs, priorites):
            self.priorities[idx] = priority


class DistributionalVisualNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, v_min, v_max, n_atoms):
        super().__init__()
        self.n_atoms = n_atoms
        self.action_dim = action_dim

        self.register_buffer("support", torch.linspace(v_min, v_max, n_atoms))

        self.backbone = nn.Sequential(
            modules.NoisyLinear(state_dim, 128, std_init=0.5),
            nn.ReLU(),
            modules.NoisyLinear(128, 128, std_init=0.5),
            nn.ReLU(),
        )

        self.fc_q = modules.NoisyLinear(128, action_dim * n_atoms, std_init=0.5)

        self.aux_head = nn.Sequential(
            nn.Linear(128 + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, state_dim)
        )

    def forward(self, states, actions=None):
        batch_size = states.size(0)

        h = self.backbone(states)
        q_logits = self.fc_q(h)

        q_view = q_logits.view(batch_size, self.action_dim, self.n_atoms)

        q_probs = torch.softmax(q_view, dim=2)

        q_log_probs = torch.log_softmax(q_view, dim=2)

        if actions is not None:
            a_onehot = F.one_hot(actions, num_classes=self.action_dim).float()
            aux_input = torch.cat([h, a_onehot], dim=1)
            next_state_pred = self.aux_head(aux_input)

            return q_probs, q_log_probs,next_state_pred

        return q_probs, q_log_probs

    def get_q_value(self, x):
        q_probs, _ = self.forward(x)
        weights = q_probs * self.support
        return weights.sum(dim=2)



class RDQN(Agent):
    def __init__(
        self,
        env,
        batch_size=64,
        gamma=0.85,
        C=100,
        buffer_capacity=10000,
        n_step=3,
        zeta=0.95,
        v_min=0,
        v_max=7,
        total_training_steps=1000,
        lambda_=0.8
    ):
        super().__init__(env)
        self.type = type
        self.batch_size = batch_size
        self.gamma = gamma
        self.n_step = n_step
        self.C = C
        self.zeta = zeta
        self.lambda_ = lambda_

        self.v_min = v_min
        self.v_max = v_max
        self.n_atoms = 51
        self.delta_z = (self.v_max - self.v_min) / (self.n_atoms - 1)

        self.D = NStepPriorityReplayBuffer(
            buffer_capacity, self.n_step, total_training_steps, self.gamma
        )

        if isinstance(env.unwrapped.observation_space, gymnasium.spaces.dict.Dict):
            self.state_dim = sum(
                math.prod(obs.shape) for obs in env.unwrapped.observation_space.values()
            )
        else:
            self.state_dim = math.prod(env.unwrapped.observation_space.shape)
        self.action_dim = int(env.unwrapped.action_space.n)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.q_1 = DistributionalVisualNetwork(
            self.state_dim, self.action_dim, self.v_min, self.v_max, self.n_atoms
        ).to(self.device)
        self.q_2 = DistributionalVisualNetwork(
            self.state_dim, self.action_dim, self.v_min, self.v_max, self.n_atoms
        ).to(self.device)
        self.q_2.load_state_dict(self.q_1.state_dict())

        self.optimizer = optim.Adam(
            self.q_1.parameters(), lr=1e-3, eps=1e-5
        )  
        self.counter = 0

    def eval(self):
        self.q_1.eval()
        self.q_2.eval()

    def train(self):
        self.q_1.train()
        self.q_2.train()

    def save_model(self, path):
        torch.save(
            {
                "q_1": self.q_1.state_dict(),
                "q_2": self.q_2.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            path
        )

    def load_model(self, path):
        checkpoint = torch.load(path, weights_only=True, map_location=self.device)
        self.q_1.load_state_dict(checkpoint["q_1"])
        self.q_2.load_state_dict(checkpoint["q_2"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])

    def predict(self, obs):
        with torch.no_grad():
            obs_t = torch.tensor(
                obs, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            q_scalar = self.q_1.get_q_value(obs_t)
            action = q_scalar.argmax(dim=1).item()
        return action

    def learn(self, S, A, R, S_next):
        done, obs_next = S_next
        self.D.push(S, A, R, obs_next, done)

        if len(self.D) < self.batch_size:
            return

        idxs, weights, batch = self.D.sample(self.batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards = torch.tensor(
            rewards, dtype=torch.float32, device=self.device
        ).unsqueeze(
            1
        )  
        next_states = torch.tensor(
            np.array(next_states), dtype=torch.float32, device=self.device
        )
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(
            1
        ) 
        weights = torch.tensor(weights, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            next_action_idx = self.q_1.get_q_value(next_states).argmax(dim=1)
            p_next, _ = self.q_2(next_states) 

            p_next_best = p_next[range(self.batch_size), next_action_idx]

            Tz_1 = rewards + (1 - dones) * self.gamma * self.q_1.support
            Tz_1 = Tz_1.clamp(self.v_min, self.v_max)

            Tz_n = rewards + (1 - dones) * (self.gamma**self.n_step) * self.q_1.support
            Tz_n = Tz_n.clamp(min=self.v_min, max=self.v_max)
            Tz = (1 - self.lambda_) * Tz_1 + self.lambda_ * Tz_n

            b = (Tz - self.v_min) / self.delta_z
            l = b.floor().long().clamp(0, self.n_atoms - 1)
            u = b.ceil().long().clamp(0, self.n_atoms - 1)

            m = torch.zeros(self.batch_size, self.n_atoms, device=self.device)

            offset = (
                torch.linspace(
                    0,
                    (self.batch_size - 1) * self.n_atoms,
                    self.batch_size,
                    device=self.device,
                )
                .long()
                .unsqueeze(1)
                .expand(self.batch_size, self.n_atoms)
            )

            m.view(-1).index_add_(
                0, (l + offset).view(-1), (p_next_best * (u.float() - b)).view(-1)
            )
            m.view(-1).index_add_(
                0, (u + offset).view(-1), (p_next_best * (b - l.float())).view(-1)
            )

        _, log_p_pred,next_state_pred = self.q_1(states,actions)

        log_p_action = log_p_pred[range(self.batch_size), actions]

        loss_elementwise = -torch.sum(m * log_p_action, dim=1)
 
        q_loss = (weights * loss_elementwise).mean()
        aux_loss = F.mse_loss(next_state_pred, next_states)
        loss = q_loss + 0.1 * aux_loss

        new_priorities = loss_elementwise.detach().cpu().numpy() + 1e-6
        self.D.update_priorities(idxs, new_priorities)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_1.parameters(), 10.0)
        self.optimizer.step()

        self.counter += 1
        if self.counter % self.C == 0:
            self.q_2.load_state_dict(self.q_1.state_dict())
