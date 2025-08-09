#!/usr/bin/env python3
"""
Comprehensive Reinforcement Learning Model Implementations

This module implements all specified RL algorithms with advanced features
for benchmarking between Python and Rust implementations.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque, namedtuple
import random
from typing import Dict, Any, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

# Experience replay buffer
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


class DQN(nn.Module):
    """Deep Q-Network implementation."""
    
    def __init__(self, state_size: int, action_size: int, hidden_sizes: List[int] = [64, 64]):
        super(DQN, self).__init__()
        
        layers = []
        input_size = state_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            input_size = hidden_size
        
        layers.append(nn.Linear(input_size, action_size))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class DDQN(nn.Module):
    """Double Deep Q-Network implementation."""
    
    def __init__(self, state_size: int, action_size: int, hidden_sizes: List[int] = [64, 64]):
        super(DDQN, self).__init__()
        
        # Value stream
        value_layers = []
        input_size = state_size
        
        for hidden_size in hidden_sizes:
            value_layers.extend([
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            input_size = hidden_size
        
        value_layers.append(nn.Linear(input_size, 1))
        self.value_stream = nn.Sequential(*value_layers)
        
        # Advantage stream
        advantage_layers = []
        input_size = state_size
        
        for hidden_size in hidden_sizes:
            advantage_layers.extend([
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            input_size = hidden_size
        
        advantage_layers.append(nn.Linear(input_size, action_size))
        self.advantage_stream = nn.Sequential(*advantage_layers)
    
    def forward(self, x):
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        
        # Combine value and advantage
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values


class DuelingDQN(nn.Module):
    """Dueling Deep Q-Network implementation."""
    
    def __init__(self, state_size: int, action_size: int, hidden_sizes: List[int] = [64, 64]):
        super(DuelingDQN, self).__init__()
        
        # Shared layers
        shared_layers = []
        input_size = state_size
        
        for hidden_size in hidden_sizes[:-1]:
            shared_layers.extend([
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            input_size = hidden_size
        
        self.shared_layers = nn.Sequential(*shared_layers)
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[-1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[-1], 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[-1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[-1], action_size)
        )
    
    def forward(self, x):
        shared = self.shared_layers(x)
        value = self.value_stream(shared)
        advantage = self.advantage_stream(shared)
        
        # Combine value and advantage
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values


class PrioritizedReplayBuffer:
    """Prioritized Experience Replay Buffer."""
    
    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0
    
    def add(self, experience: Experience, priority: float = None):
        """Add experience to buffer with priority."""
        if priority is None:
            priority = max(self.priorities) if self.buffer else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        
        self.priorities[self.position] = priority ** self.alpha
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int) -> Tuple[List[Experience], List[int], np.ndarray]:
        """Sample experiences with priorities."""
        if self.size < batch_size:
            indices = np.random.randint(0, self.size, size=self.size)
            experiences = [self.buffer[i] for i in indices]
            weights = np.ones(self.size)
            return experiences, indices, weights
        
        # Calculate sampling probabilities
        priorities = self.priorities[:self.size]
        probabilities = priorities / priorities.sum()
        
        # Sample indices
        indices = np.random.choice(self.size, batch_size, p=probabilities)
        experiences = [self.buffer[i] for i in indices]
        
        # Calculate importance sampling weights
        weights = (self.size * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()
        
        return experiences, indices, weights
    
    def update_priorities(self, indices: List[int], priorities: List[float]):
        """Update priorities for sampled experiences."""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority ** self.alpha


class PrioritizedDQN(nn.Module):
    """Prioritized Deep Q-Network implementation."""
    
    def __init__(self, state_size: int, action_size: int, hidden_sizes: List[int] = [64, 64]):
        super(PrioritizedDQN, self).__init__()
        
        layers = []
        input_size = state_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            input_size = hidden_size
        
        layers.append(nn.Linear(input_size, action_size))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class RainbowDQN(nn.Module):
    """Rainbow DQN implementation combining multiple improvements."""
    
    def __init__(self, state_size: int, action_size: int, hidden_sizes: List[int] = [64, 64],
                 num_atoms: int = 51, v_min: float = -10.0, v_max: float = 10.0):
        super(RainbowDQN, self).__init__()
        
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.support = torch.linspace(v_min, v_max, num_atoms)
        
        # Shared layers
        shared_layers = []
        input_size = state_size
        
        for hidden_size in hidden_sizes[:-1]:
            shared_layers.extend([
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            input_size = hidden_size
        
        self.shared_layers = nn.Sequential(*shared_layers)
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[-1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[-1], num_atoms)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[-1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[-1], action_size * num_atoms)
        )
    
    def forward(self, x):
        shared = self.shared_layers(x)
        
        value = self.value_stream(shared).view(-1, 1, self.num_atoms)
        advantage = self.advantage_stream(shared).view(-1, -1, self.num_atoms)
        
        # Combine value and advantage
        q_dist = value + (advantage - advantage.mean(dim=1, keepdim=True))
        q_dist = F.softmax(q_dist, dim=-1)
        
        return q_dist


class RLAgent:
    """Base RL Agent class."""
    
    def __init__(self, state_size: int, action_size: int, **kwargs):
        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Hyperparameters
        self.learning_rate = kwargs.get("learning_rate", 0.001)
        self.gamma = kwargs.get("gamma", 0.99)
        self.epsilon = kwargs.get("epsilon", 1.0)
        self.epsilon_min = kwargs.get("epsilon_min", 0.01)
        self.epsilon_decay = kwargs.get("epsilon_decay", 0.995)
        self.batch_size = kwargs.get("batch_size", 32)
        self.memory_size = kwargs.get("memory_size", 100000)
        self.target_update_freq = kwargs.get("target_update_freq", 1000)
        self.update_freq = kwargs.get("update_freq", 4)
        
        # Networks
        self.q_network = None
        self.target_network = None
        self.optimizer = None
        
        # Memory
        self.memory = deque(maxlen=self.memory_size)
        self.step_count = 0
    
    def act(self, state: np.ndarray, training: bool = True) -> int:
        """Select action using epsilon-greedy policy."""
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
    
    def remember(self, state: np.ndarray, action: int, reward: float, 
                next_state: np.ndarray, done: bool):
        """Store experience in memory."""
        self.memory.append(Experience(state, action, reward, next_state, done))
    
    def replay(self) -> Optional[float]:
        """Train the network on a batch of experiences."""
        if len(self.memory) < self.batch_size:
            return None
        
        # Sample batch
        batch = random.sample(self.memory, self.batch_size)
        
        states = torch.FloatTensor([e.state for e in batch]).to(self.device)
        actions = torch.LongTensor([e.action for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e.reward for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e.next_state for e in batch]).to(self.device)
        dones = torch.BoolTensor([e.done for e in batch]).to(self.device)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss.item()
    
    def update_target_network(self):
        """Update target network."""
        self.target_network.load_state_dict(self.q_network.state_dict())


class DQNAgent(RLAgent):
    """DQN Agent implementation."""
    
    def __init__(self, state_size: int, action_size: int, **kwargs):
        super().__init__(state_size, action_size, **kwargs)
        
        self.q_network = DQN(state_size, action_size).to(self.device)
        self.target_network = DQN(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        
        self.update_target_network()


class DDQNAgent(RLAgent):
    """Double DQN Agent implementation."""
    
    def __init__(self, state_size: int, action_size: int, **kwargs):
        super().__init__(state_size, action_size, **kwargs)
        
        self.q_network = DDQN(state_size, action_size).to(self.device)
        self.target_network = DDQN(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        
        self.update_target_network()
    
    def replay(self) -> Optional[float]:
        """Train the network using Double DQN."""
        if len(self.memory) < self.batch_size:
            return None
        
        # Sample batch
        batch = random.sample(self.memory, self.batch_size)
        
        states = torch.FloatTensor([e.state for e in batch]).to(self.device)
        actions = torch.LongTensor([e.action for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e.reward for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e.next_state for e in batch]).to(self.device)
        dones = torch.BoolTensor([e.done for e in batch]).to(self.device)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values using Double DQN
        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(1)
            next_q_values = self.target_network(next_states).gather(1, next_actions.unsqueeze(1))
            target_q_values = rewards + (self.gamma * next_q_values.squeeze() * ~dones)
        
        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss.item()


class DuelingDQNAgent(RLAgent):
    """Dueling DQN Agent implementation."""
    
    def __init__(self, state_size: int, action_size: int, **kwargs):
        super().__init__(state_size, action_size, **kwargs)
        
        self.q_network = DuelingDQN(state_size, action_size).to(self.device)
        self.target_network = DuelingDQN(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        
        self.update_target_network()


class PrioritizedDQNAgent(RLAgent):
    """Prioritized DQN Agent implementation."""
    
    def __init__(self, state_size: int, action_size: int, **kwargs):
        super().__init__(state_size, action_size, **kwargs)
        
        self.q_network = PrioritizedDQN(state_size, action_size).to(self.device)
        self.target_network = PrioritizedDQN(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        
        # Prioritized replay buffer
        self.memory = PrioritizedReplayBuffer(
            capacity=self.memory_size,
            alpha=kwargs.get("alpha", 0.6),
            beta=kwargs.get("beta", 0.4)
        )
        
        self.update_target_network()
    
    def remember(self, state: np.ndarray, action: int, reward: float, 
                next_state: np.ndarray, done: bool, priority: float = None):
        """Store experience in prioritized memory."""
        experience = Experience(state, action, reward, next_state, done)
        self.memory.add(experience, priority)
    
    def replay(self) -> Optional[float]:
        """Train the network using prioritized experience replay."""
        if self.memory.size < self.batch_size:
            return None
        
        # Sample batch with priorities
        experiences, indices, weights = self.memory.sample(self.batch_size)
        
        states = torch.FloatTensor([e.state for e in experiences]).to(self.device)
        actions = torch.LongTensor([e.action for e in experiences]).to(self.device)
        rewards = torch.FloatTensor([e.reward for e in experiences]).to(self.device)
        next_states = torch.FloatTensor([e.next_state for e in experiences]).to(self.device)
        dones = torch.BoolTensor([e.done for e in experiences]).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Compute loss with importance sampling weights
        td_errors = torch.abs(current_q_values.squeeze() - target_q_values)
        loss = (td_errors * weights).mean()
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update priorities
        priorities = td_errors.detach().cpu().numpy() + 1e-6
        self.memory.update_priorities(indices, priorities)
        
        # Update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss.item()


class RainbowDQNAgent(RLAgent):
    """Rainbow DQN Agent implementation."""
    
    def __init__(self, state_size: int, action_size: int, **kwargs):
        super().__init__(state_size, action_size, **kwargs)
        
        self.q_network = RainbowDQN(state_size, action_size).to(self.device)
        self.target_network = RainbowDQN(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        
        self.update_target_network()


def create_rl_agent(algorithm: str, state_size: int, action_size: int, **kwargs) -> RLAgent:
    """Factory function to create RL agents."""
    
    agent_configs = {
        "dqn": {
            "class": DQNAgent,
            "default_params": {}
        },
        "ddqn": {
            "class": DDQNAgent,
            "default_params": {}
        },
        "dueling_dqn": {
            "class": DuelingDQNAgent,
            "default_params": {}
        },
        "prioritized_dqn": {
            "class": PrioritizedDQNAgent,
            "default_params": {"alpha": 0.6, "beta": 0.4}
        },
        "rainbow_dqn": {
            "class": RainbowDQNAgent,
            "default_params": {}
        }
    }
    
    if algorithm not in agent_configs:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    config = agent_configs[algorithm]
    agent_class = config["class"]
    default_params = config["default_params"].copy()
    
    # Update with provided parameters
    default_params.update(kwargs)
    
    return agent_class(state_size, action_size, **default_params)


def get_agent_info(agent: RLAgent) -> Dict[str, Any]:
    """Get comprehensive information about an agent."""
    
    total_params = sum(p.numel() for p in agent.q_network.parameters())
    trainable_params = sum(p.numel() for p in agent.q_network.parameters() if p.requires_grad)
    
    # Calculate model size in MB
    param_size = 0
    buffer_size = 0
    
    for param in agent.q_network.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in agent.q_network.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024 / 1024
    
    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "model_size_mb": size_mb,
        "algorithm": agent.__class__.__name__,
        "epsilon": agent.epsilon,
        "memory_size": len(agent.memory)
    } 