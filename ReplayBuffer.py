import random
import torch
import numpy as np
from collections import deque

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Experience():
    
    def __init__(self, state, action, reward ,next_state, done):
        
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done

class Buffer():
    
    def __init__(self, batch_size, maxlen):
        
        self.memory = deque(maxlen=maxlen)
        self.batch_size = batch_size
        
    def add(self, state, action, reward ,next_state, done):
        
        e = Experience(state, action, reward ,next_state, done)
        self.memory.append(e)
        
    def sample(self):
        
        batch = random.sample(self.memory, k = self.batch_size)
        
        states = torch.from_numpy(np.vstack([e.state for e in batch])).float().to(DEVICE)
        actions = torch.from_numpy(np.vstack([e.action for e in batch])).float().to(DEVICE)
        rewards = torch.from_numpy(np.vstack([e.reward for e in batch])).float().to(DEVICE)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in batch])).float().to(DEVICE)
        dones = torch.from_numpy(np.vstack([e.done for e in batch ]).astype(np.uint8)).float().to(DEVICE)
        
        return (states, actions, rewards, next_states, dones)
    
    def count(self):
        
        return len(self.memory)