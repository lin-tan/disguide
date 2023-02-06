from collections import deque

from my_utils import perm
import torch
import numpy as np


class ReplayMemory(object):
    """Experience replay class interface"""
    def __init__(self):
        """Initialize experience replay"""
        pass

    def update(self, fake, t_logit):
        """Update experience replay with batch of features and labels"""
        pass

    def sample(self):
        """Return batch of features and labels"""
        pass

    def new_epoch(self):
        """Update epoch. This is only relevant for experience replay that has some concept of aging"""
        pass


class NoReplayMemory(ReplayMemory):
    def __init__(self):
        pass


class ClassicalMemory(ReplayMemory):
    """Circular FIFO buffer based experiment replay. Returns uniform random samples when queried."""
    def __init__(self, device, length, batch_size):
        """Initialize replay memory"""
        self.device = device
        self.max_length = length
        self.fakes = None
        self.logits = None
        self.batch_size = batch_size
        self.size = 0
        self.head = 0

    def update(self, fake, t_logit):
        """Update memory with batch of features and labels"""
        fake = fake.cpu()
        t_logit = t_logit.cpu()
        if self.fakes is None:  # Initialize circular buffer if it hasn't already been.
            self.fakes = torch.zeros((self.max_length, fake.shape[1], fake.shape[2], fake.shape[3]))
            self.logits = torch.zeros((self.max_length, t_logit.shape[1]))
        tail = self.head + fake.shape[0]
        if tail <= self.max_length:  # Buffer is not full
            self.fakes[self.head:tail] = fake
            self.logits[self.head:tail] = t_logit
            if self.size < self.max_length:
                self.size = tail
        else:  # Buffer is full
            n = self.max_length - self.head
            self.fakes[self.head:tail] = fake[:n]
            self.logits[self.head:tail] = t_logit[:n]
            tail = tail % self.max_length
            self.fakes[:tail] = fake[n:]
            self.logits[:tail] = t_logit[n:]
            self.size = self.max_length
        self.head = tail % self.max_length

    def sample(self):
        """Return samples uniformly at random from memory"""
        assert self.fakes is not None  # Only sample after having stored samples
        assert self.size >= self.batch_size  # Only sample if we have stored a full batch of samples
        idx = perm(self.size, self.batch_size, self.device).cpu()
        return self.fakes[idx].to(self.device), self.logits[idx].to(self.device)

    def new_epoch(self):
        """Does nothing for this type of experience replay"""
        pass

    def __len__(self):
        """Returns number of stored samples"""
        return self.size


def init_replay_memory(args):
    if args.replay == "Off":
        return NoReplayMemory()
    if args.replay == "Classic":
        return ClassicalMemory(args.device, args.replay_size, args.batch_size)
    raise ValueError(f"Unknown replay parameter {args.replay}")