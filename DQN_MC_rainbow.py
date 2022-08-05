import os
import sys
import gym
import time
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision import transforms
from PIL import Image
import torch.nn.init as init
import torch.nn.utils.prune as prune
import torch.backends.cudnn as cudnn
from functools import reduce
import operator
import VTT_RL

# ------------------------< SEED FIX >---------------------------------
custom_seed_val = 0
torch.manual_seed(custom_seed_val)
torch.cuda.manual_seed(custom_seed_val)
torch.cuda.manual_seed_all(custom_seed_val)
cudnn.benchmark = False
cudnn.deterministic = True
random.seed(custom_seed_val)

input_stack_size = 163

torch.cuda.memory_summary(device=None, abbreviated=False)

cur_path = os.path.dirname(__file__)
aim_agent_version = "agent_rainbow_DQN_v1.01.pth"

env = gym.make('VTT-v0').unwrapped

'''------------------< INPUT SHAPE DESCRIPTION >--------------------+
 |                                                                  |
 |inputs shape = [base part position, quertonion, vel, ang_vel (13)]|
 |              +[joint_state(12)]                                  |
 |              +[joint_vel(12)]                                    |
 |              +[joint_reaction force(72)]                         |
 |              +[member pos, ori(42)]                              |
 |              +[action(12)]                 = 163                 |
 +----------------------------------------------------------------'''

# inputs = 7 + 12 + 12 + 12 --old inputs 26016
inputs = 13 + 12 + 12 + 72 + 42 + 12

output = 12

max_episode_len = 2 ** 10
episode_persent_step = max_episode_len // 100

plt.ion()

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
# Transition = namedtuple('Transition',
#                         ('state', 'action', 'next_state', 'reward'))

# Transition = namedtuple('Transition',
#                         ('state', 'action', 'next_state', 'reward', 'weights', 'indices'))


#
class ReplayBuffer:
    """
    ## Buffer for Prioritized Experience Replay
    [Prioritized experience replay](https://papers.labml.ai/paper/1511.05952)
     samples important transitions more frequently.
    The transitions are prioritized by the Temporal Difference error (td error), $\delta$.
    We sample transition $i$ with probability,
    $$P(i) = \frac{p_i^\alpha}{\sum_k p_k^\alpha}$$
    where $\alpha$ is a hyper-parameter that determines how much
    prioritization is used, with $\alpha = 0$ corresponding to uniform case.
    $p_i$ is the priority.
    We use proportional prioritization $p_i = |\delta_i| + \epsilon$ where
    $\delta_i$ is the temporal difference for transition $i$.
    We correct the bias introduced by prioritized replay using
     importance-sampling (IS) weights
    $$w_i = \bigg(\frac{1}{N} \frac{1}{P(i)}\bigg)^\beta$$ in the loss function.
    This fully compensates when $\beta = 1$.
    We normalize weights by $\frac{1}{\max_i w_i}$ for stability.
    Unbiased nature is most important towards the convergence at end of training.
    Therefore we increase $\beta$ towards end of training.
    ### Binary Segment Tree
    We use a binary segment tree to efficiently calculate
    $\sum_k^i p_k^\alpha$, the cumulative probability,
    which is needed to sample.
    We also use a binary segment tree to find $\min p_i^\alpha$,
    which is needed for $\frac{1}{\max_i w_i}$.
    We can also use a min-heap for this.
    Binary Segment Tree lets us calculate these in $\mathcal{O}(\log n)$
    time, which is way more efficient that the naive $\mathcal{O}(n)$
    approach.
    This is how a binary segment tree works for sum;
    it is similar for minimum.
    Let $x_i$ be the list of $N$ values we want to represent.
    Let $b_{i,j}$ be the $j^{\mathop{th}}$ node of the $i^{\mathop{th}}$ row
     in the binary tree.
    That is two children of node $b_{i,j}$ are $b_{i+1,2j}$ and $b_{i+1,2j + 1}$.
    The leaf nodes on row $D = \left\lceil {1 + \log_2 N} \right\rceil$
     will have values of $x$.
    Every node keeps the sum of the two child nodes.
    That is, the root node keeps the sum of the entire array of values.
    The left and right children of the root node keep
     the sum of the first half of the array and
     the sum of the second half of the array, respectively.
    And so on...
    $$b_{i,j} = \sum_{k = (j -1) * 2^{D - i} + 1}^{j * 2^{D - i}} x_k$$
    Number of nodes in row $i$,
    $$N_i = \left\lceil{\frac{N}{D - i + 1}} \right\rceil$$
    This is equal to the sum of nodes in all rows above $i$.
    So we can use a single array $a$ to store the tree, where,
    $$b_{i,j} \rightarrow a_{N_i + j}$$
    Then child nodes of $a_i$ are $a_{2i}$ and $a_{2i + 1}$.
    That is,
    $$a_i = a_{2i} + a_{2i + 1}$$
    This way of maintaining binary trees is very easy to program.
    *Note that we are indexing starting from 1*.
    We use the same structure to compute the minimum.
    """

    def __init__(self, capacity, alpha):
        """
        ### Initialize
        """
        # We use a power of $2$ for capacity because it simplifies the code and debugging
        self.capacity = capacity
        # $\alpha$
        self.alpha = alpha

        # Maintain segment binary trees to take sum and find minimum over a range
        self.priority_sum = [0 for _ in range(2 * self.capacity)]
        self.priority_min = [float('inf') for _ in range(2 * self.capacity)]

        # Current max priority, $p$, to be assigned to new transitions
        self.max_priority = 1.

        # Arrays for buffer
        self.data = {
            'obs':      np.zeros(shape=(capacity, 1, 163, 163), dtype=np.uint8  ),
            'action':   np.zeros(shape=(capacity, 12),          dtype=np.int32  ),
            'reward':   np.zeros(shape=(capacity),              dtype=np.float32),
            'next_obs': np.zeros(shape=(capacity, 1, 163, 163), dtype=np.uint8  ),
        }
        # We use cyclic buffers to store data, and `next_idx` keeps the index of the next empty
        # slot
        self.next_idx = 0

        # Size of the buffer
        self.size = 0

    def push(self, obs, action, reward, next_obs):
        """
        ### Add sample to queue
        """

        # Get next available slot
        idx = self.next_idx

        # store in the queue
        self.data['obs'][idx] = obs
        self.data['action'][idx] = action
        self.data['reward'][idx] = reward
        self.data['next_obs'][idx] = next_obs

        # Increment next available slot
        self.next_idx = (idx + 1) % self.capacity
        # Calculate the size
        self.size = min(self.capacity, self.size + 1)

        # $p_i^\alpha$, new samples get `max_priority`
        priority_alpha = self.max_priority ** self.alpha
        # Update the two segment trees for sum and minimum
        self._set_priority_min(idx, priority_alpha)
        self._set_priority_sum(idx, priority_alpha)

    def _set_priority_min(self, idx, priority_alpha):
        """
        #### Set priority in binary segment tree for minimum
        """

        # Leaf of the binary tree
        idx += self.capacity
        self.priority_min[idx] = priority_alpha

        # Update tree, by traversing along ancestors.
        # Continue until the root of the tree.
        while idx >= 2:
            # Get the index of the parent node
            idx //= 2
            # Value of the parent node is the minimum of it's two children
            self.priority_min[idx] = min(self.priority_min[2 * idx], self.priority_min[2 * idx + 1])

    def _set_priority_sum(self, idx, priority):
        """
        #### Set priority in binary segment tree for sum
        """

        # Leaf of the binary tree
        idx += self.capacity
        # Set the priority at the leaf
        self.priority_sum[idx] = priority

        # Update tree, by traversing along ancestors.
        # Continue until the root of the tree.
        while idx >= 2:
            # Get the index of the parent node
            idx //= 2
            # Value of the parent node is the sum of it's two children
            self.priority_sum[idx] = self.priority_sum[2 * idx] + self.priority_sum[2 * idx + 1]

    def _sum(self):
        """
        #### $\sum_k p_k^\alpha$
        """

        # The root node keeps the sum of all values
        return self.priority_sum[1]

    def _min(self):
        """
        #### $\min_k p_k^\alpha$
        """

        # The root node keeps the minimum of all values
        return self.priority_min[1]

    def find_prefix_sum_idx(self, prefix_sum):
        """
        #### Find largest $i$ such that $\sum_{k=1}^{i} p_k^\alpha  \le P$
        """

        # Start from the root
        idx = 1
        while idx < self.capacity:
            # If the sum of the left branch is higher than required sum
            if self.priority_sum[idx * 2] > prefix_sum:
                # Go to left branch of the tree
                idx = 2 * idx
            else:
                # Otherwise go to right branch and reduce the sum of left
                #  branch from required sum
                prefix_sum -= self.priority_sum[idx * 2]
                idx = 2 * idx + 1

        # We are at the leaf node. Subtract the capacity by the index in the tree
        # to get the index of actual value
        return idx - self.capacity

    def sample(self, batch_size, beta):
        """
        ### Sample from buffer
        """

        # Initialize samples
        samples = {
            'weights': np.zeros(shape=batch_size, dtype=np.float32),
            'indexes': np.zeros(shape=batch_size, dtype=np.int32)
        }

        # Get sample indexes
        for i in range(batch_size):
            p = random.random() * self._sum()
            idx = self.find_prefix_sum_idx(p)
            samples['indexes'][i] = idx

        # $\min_i P(i) = \frac{\min_i p_i^\alpha}{\sum_k p_k^\alpha}$
        prob_min = self._min() / self._sum()
        # $\max_i w_i = \bigg(\frac{1}{N} \frac{1}{\min_i P(i)}\bigg)^\beta$
        max_weight = (prob_min * self.size) ** (-beta)

        for i in range(batch_size):
            idx = samples['indexes'][i]
            # $P(i) = \frac{p_i^\alpha}{\sum_k p_k^\alpha}$
            prob = self.priority_sum[idx + self.capacity] / self._sum()
            # $w_i = \bigg(\frac{1}{N} \frac{1}{P(i)}\bigg)^\beta$
            weight = (prob * self.size) ** (-beta)
            # Normalize by $\frac{1}{\max_i w_i}$,
            #  which also cancels off the $\frac{1}{N}$ term
            samples['weights'][i] = weight / max_weight

        # Get samples data
        for k, v in self.data.items():
            samples[k] = v[samples['indexes']]

        return samples

    def update_priorities(self, indexes, priorities):
        """
        ### Update priorities
        """

        # for idx, priority in zip(indexes, priorities):
        #     # Set current max priority
        #     self.max_priority = max(self.max_priority, priority)

        #     # Calculate $p_i^\alpha$
        #     priority_alpha = priority ** self.alpha
        #     # Update the trees
        #     self._set_priority_min(idx, priority_alpha)
        #     self._set_priority_sum(idx, priority_alpha)
        
        self.max_priority = max(self.max_priority, priorities)

        # Calculate $p_i^\alpha$
        priority_alpha = priorities ** self.alpha
        # Update the trees
        self._set_priority_min(indexes, priority_alpha)
        self._set_priority_sum(indexes, priority_alpha)

    def is_full(self):
        """
        ### Whether the buffer is full
        """
        return self.capacity == self.size

### Rainbow V1
# class SegmentTree(object):
#     def __init__(self, capacity, operation, neutral_element):
#         """Build a Segment Tree data structure.
#         https://en.wikipedia.org/wiki/Segment_tree
#         Can be used as regular array, but with two
#         important differences:
#             a) setting item's value is slightly slower.
#                It is O(lg capacity) instead of O(1).
#             b) user has access to an efficient `reduce`
#                operation which reduces `operation` over
#                a contiguous subsequence of items in the
#                array.
#         Paramters
#         ---------
#         capacity: int
#             Total size of the array - must be a power of two.
#         operation: lambda obj, obj -> obj
#             and operation for combining elements (eg. sum, max)
#             must for a mathematical group together with the set of
#             possible values for array elements.
#         neutral_element: obj
#             neutral element for the operation above. eg. float('-inf')
#             for max and 0 for sum.
#         """
#         assert capacity > 0 and capacity & (capacity - 1) == 0, "capacity must be positive and a power of 2."
#         self._capacity = capacity
#         self._value = [neutral_element for _ in range(2 * capacity)]
#         self._operation = operation
#
#     def _reduce_helper(self, start, end, node, node_start, node_end):
#         if start == node_start and end == node_end:
#             return self._value[node]
#         mid = (node_start + node_end) // 2
#         if end <= mid:
#             return self._reduce_helper(start, end, 2 * node, node_start, mid)
#         else:
#             if mid + 1 <= start:
#                 return self._reduce_helper(start, end, 2 * node + 1, mid + 1, node_end)
#             else:
#                 return self._operation(
#                     self._reduce_helper(start, mid, 2 * node, node_start, mid),
#                     self._reduce_helper(mid + 1, end, 2 * node + 1, mid + 1, node_end)
#                 )
#
#     def reduce(self, start=0, end=None):
#         """Returns result of applying `self.operation`
#         to a contiguous subsequence of the array.
#             self.operation(arr[start], operation(arr[start+1], operation(... arr[end])))
#         Parameters
#         ----------
#         start: int
#             beginning of the subsequence
#         end: int
#             end of the subsequences
#         Returns
#         -------
#         reduced: obj
#             result of reducing self.operation over the specified range of array elements.
#         """
#         if end is None:
#             end = self._capacity
#         if end < 0:
#             end += self._capacity
#         end -= 1
#         return self._reduce_helper(start, end, 1, 0, self._capacity - 1)
#
#     def __setitem__(self, idx, val):
#         # index of the leaf
#         idx += self._capacity
#         self._value[idx] = val
#         idx //= 2
#         while idx >= 1:
#             self._value[idx] = self._operation(
#                 self._value[2 * idx],
#                 self._value[2 * idx + 1]
#             )
#             idx //= 2
#
#     def __getitem__(self, idx):
#         assert 0 <= idx < self._capacity
#         return self._value[self._capacity + idx]
#
#
# class SumSegmentTree(SegmentTree):
#     def __init__(self, capacity):
#         super(SumSegmentTree, self).__init__(
#             capacity=capacity,
#             operation=operator.add,
#             neutral_element=0.0
#         )
#
#     def sum(self, start=0, end=None):
#         """Returns arr[start] + ... + arr[end]"""
#         return super(SumSegmentTree, self).reduce(start, end)
#
#     def find_prefixsum_idx(self, prefixsum):
#         """Find the highest index `i` in the array such that
#             sum(arr[0] + arr[1] + ... + arr[i - i]) <= prefixsum
#         if array values are probabilities, this function
#         allows to sample indexes according to the discrete
#         probability efficiently.
#         Parameters
#         ----------
#         perfixsum: float
#             upperbound on the sum of array prefix
#         Returns
#         -------
#         idx: int
#             highest index satisfying the prefixsum constraint
#         """
#         assert 0 <= prefixsum <= self.sum() + 1e-5
#         idx = 1
#         while idx < self._capacity:  # while non-leaf
#             if self._value[2 * idx] > prefixsum:
#                 idx = 2 * idx
#             else:
#                 prefixsum -= self._value[2 * idx]
#                 idx = 2 * idx + 1
#         return idx - self._capacity
#
#
# class MinSegmentTree(SegmentTree):
#     def __init__(self, capacity):
#         super(MinSegmentTree, self).__init__(
#             capacity=capacity,
#             operation=min,
#             neutral_element=float('inf')
#         )
#
#     def min(self, start=0, end=None):
#         """Returns min(arr[start], ...,  arr[end])"""
#
#         return super(MinSegmentTree, self).reduce(start, end)
#
#
# class ReplayBuffer(object):
#     def __init__(self, size):
#         """Create Replay buffer.
#         Parameters
#         ----------
#         size: int
#             Max number of transitions to store in the buffer. When the buffer
#             overflows the old memories are dropped.
#         """
#         self._storage = []
#         self._maxsize = size
#         self._next_idx = 0
#
#     def __len__(self):
#         return len(self._storage)
#
#     def push(self, state, action, reward, next_state):
#         data = (state, action, reward, next_state)
#
#         if self._next_idx >= len(self._storage):
#             self._storage.append(data)
#         else:
#             self._storage[self._next_idx] = data
#         self._next_idx = (self._next_idx + 1) % self._maxsize
#
#     def _encode_sample(self, idxes):
#         obses_t, actions, rewards, obses_tp1 = [], [], [], []
#         for i in idxes:
#             data = self._storage[i]
#             obs_t, action, reward, obs_tp1 = data
#             obses_t.append(np.array(obs_t, copy=False))
#             actions.append(np.array(action, copy=False))
#             rewards.append(reward)
#             obses_tp1.append(np.array(obs_tp1, copy=False))
#
#         return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1)
#
#     def sample(self, batch_size):
#         """Sample a batch of experiences.
#         Parameters
#         ----------
#         batch_size: int
#             How many transitions to sample.
#         Returns
#         -------
#         obs_batch: np.array
#             batch of observations
#         act_batch: np.array
#             batch of actions executed given obs_batch
#         rew_batch: np.array
#             rewards received as results of executing act_batch
#         next_obs_batch: np.array
#             next set of observations seen after executing act_batch
#         done_mask: np.array
#             done_mask[i] = 1 if executing act_batch[i] resulted in
#             the end of an episode and 0 otherwise.
#         """
#         idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
#         return self._encode_sample(idxes)
#
#
# class PrioritizedReplayBuffer(ReplayBuffer):
#     def __init__(self, size, alpha):
#         """Create Prioritized Replay buffer.
#         Parameters
#         ----------
#         size: int
#             Max number of transitions to store in the buffer. When the buffer
#             overflows the old memories are dropped.
#         alpha: float
#             how much prioritization is used
#             (0 - no prioritization, 1 - full prioritization)
#         See Also
#         --------
#         ReplayBuffer.__init__
#         """
#         super(PrioritizedReplayBuffer, self).__init__(size)
#         assert alpha > 0
#         self._alpha = alpha
#
#         it_capacity = 1
#         while it_capacity < size:
#             it_capacity *= 2
#
#         self._it_sum = SumSegmentTree(it_capacity)
#         self._it_min = MinSegmentTree(it_capacity)
#         self._max_priority = 1.0
#
#     def push(self, *args, **kwargs):
#         """See ReplayBuffer.store_effect"""
#         idx = self._next_idx
#         super(PrioritizedReplayBuffer, self).push(*args, **kwargs)
#         self._it_sum[idx] = self._max_priority ** self._alpha
#         self._it_min[idx] = self._max_priority ** self._alpha
#
#     def _sample_proportional(self, batch_size):
#         res = []
#         for _ in range(batch_size):
#             # TODO(szymon): should we ensure no repeats?
#             mass = random.random() * self._it_sum.sum(0, len(self._storage) - 1)
#             idx = self._it_sum.find_prefixsum_idx(mass)
#             res.append(idx)
#         return res
#
#     def sample(self, batch_size, beta):
#         """Sample a batch of experiences.
#         compared to ReplayBuffer.sample
#         it also returns importance weights and idxes
#         of sampled experiences.
#         Parameters
#         ----------
#         batch_size: int
#             How many transitions to sample.
#         beta: float
#             To what degree to use importance weights
#             (0 - no corrections, 1 - full correction)
#         Returns
#         -------
#         obs_batch: np.array
#             batch of observations
#         act_batch: np.array
#             batch of actions executed given obs_batch
#         rew_batch: np.array
#             rewards received as results of executing act_batch
#         next_obs_batch: np.array
#             next set of observations seen after executing act_batch
#         done_mask: np.array
#             done_mask[i] = 1 if executing act_batch[i] resulted in
#             the end of an episode and 0 otherwise.
#         weights: np.array
#             Array of shape (batch_size,) and dtype np.float32
#             denoting importance weight of each sampled transition
#         idxes: np.array
#             Array of shape (batch_size,) and dtype np.int32
#             idexes in buffer of sampled experiences
#         """
#         assert beta > 0
#
#         idxes = self._sample_proportional(batch_size)
#
#         weights = []
#         p_min = self._it_min.min() / self._it_sum.sum()
#         max_weight = (p_min * len(self._storage)) ** (-beta)
#
#         for idx in idxes:
#             p_sample = self._it_sum[idx] / self._it_sum.sum()
#             weight = (p_sample * len(self._storage)) ** (-beta)
#             weights.append(weight / max_weight)
#         weights = np.array(weights)
#         encoded_sample = self._encode_sample(idxes)
#         return tuple(list(encoded_sample) + [weights, idxes])
#
#     def update_priorities(self, idxes, priorities):
#         """Update priorities of sampled transitions.
#         sets priority of transition at index idxes[i] in buffer
#         to priorities[i].
#         Parameters
#         ----------
#         idxes: [int]
#             List of idxes of sampled transitions
#         priorities: [float]
#             List of updated priorities corresponding to
#             transitions at the sampled idxes denoted by
#             variable `idxes`.
#         """
#         assert len(idxes) == len(priorities)
#         for idx, priority in zip(idxes, priorities):
#             assert priority > 0
#             assert 0 <= idx < len(self._storage)
#             self._it_sum[idx] = priority ** self._alpha
#             self._it_min[idx] = priority ** self._alpha
#
#             self._max_priority = max(self._max_priority, priority)
#


class DQN(nn.Module):

    def __init__(self, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, kernel_size=11, stride=5)
        self.bn1 = nn.BatchNorm2d(4)
        self.conv2 = nn.Conv2d(4, 6, kernel_size=5, stride=1)
        self.bn2 = nn.BatchNorm2d(6)
        self.conv3 = nn.Conv2d(6, 16, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(16)
        self.fc1 = nn.Linear(10000, 2000)
        self.fc2 = nn.Linear(2000, 400)
        self.fc3 = nn.Linear(400, outputs)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = x.to(device)
        x = self.bn1(self.conv1(x))
        x = self.bn2(self.conv2(x))
        x = self.bn3(self.conv3(x))
        x = x.view(-1, reduce(operator.mul, x.shape, 1))
        # x = x.view(-1, math.prod(x.shape))
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        out = self.fc3(x)

        return out


BATCH_SIZE = 1  # 256 64 1
GAMMA = 0.999
EPS_START = 0.5
EPS_END = 0.01
EPS_DECAY = 200
TARGET_UPDATE = 100
LEARNING_RATE = 0.001  # 0.001 0.01

env.reset()

# Get number of actions from gym action space
n_actions = env.action_space.n

policy_net = DQN(env.action_space.n * output).to(device)  # cnn2d prune
target_net = DQN(env.action_space.n * output).to(device)  # cnn2d prune

# Global prune
parameters_to_prune = (
    (policy_net.conv1, 'weight'),
    (policy_net.conv2, 'weight'),
    (policy_net.conv3, 'weight'),
    (policy_net.fc1, 'weight'),
    (policy_net.fc2, 'weight'),
    (policy_net.fc3, 'weight'),
)

prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured,
    amount=0.2,
)

parameters_to_prune = (
    (target_net.conv1, 'weight'),
    (target_net.conv2, 'weight'),
    (target_net.conv3, 'weight'),
    (target_net.fc1, 'weight'),
    (target_net.fc2, 'weight'),
    (target_net.fc3, 'weight'),
)

prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured,
    amount=0.2,
)

total1 = sum(p.numel() for p in policy_net.parameters())
trainpa1 = sum(p.numel() for p in policy_net.parameters() if p.requires_grad)
total2 = sum(p.numel() for p in target_net.parameters())
trainpa2 = sum(p.numel() for p in target_net.parameters() if p.requires_grad)

print("Params of policy net:", total1,
      "Trainable params of policy net:", trainpa1)
print("Params of target net:", total2,
      "Trainable params of target net:", trainpa2)

target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
# criterion = torch.jit.script(nn.MSELoss())
criterion = torch.jit.script(nn.SmoothL1Loss())
# memory = ReplayMemory(10000)
memory = ReplayBuffer(100000, 0.4)
# memory = PrioritizedReplayBuffer(100000, 0.4)


steps_done = 0


def select_action(state):  # steps_done,
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                    math.exp(-0.1 * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # return policy_net(state).max(1)[1].view(1, 1)
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).view(output, n_actions).max(-1)[1]
    else:
        # return torch.randn([output], device=device, dtype=torch.float32)
        return torch.tensor([random.randrange(n_actions) for _ in range(output)], device=device, dtype=torch.float32)


reward_hisroty = []
loss_hisroty = []

fig = plt.figure(1)
plt.clf()

ax_one = plt.gca()
ax_loss = ax_one.twinx()

plt.title('Training...')
ax_one.set_xlabel("Episode")
ax_one.set_ylabel("Reward_avg/life")
ax_loss.set_ylabel("Loss_avg/life")

p_reward, = ax_one.plot([], color='deepskyblue', label="Reward_avg/life")
p_mean, = ax_one.plot([], color='navy', label="Reward_filter_100")
p_loss, = ax_loss.plot([], color='tomato', label="Loss_avg/life")

ax_one.grid(True)
ax_loss.legend(handles=[p_reward, p_mean, p_loss], loc='upper left')


def plot_durations():
    objective_val = torch.tensor(reward_hisroty, dtype=torch.float)

    ax_one.plot(objective_val.numpy(), color='deepskyblue')

    # Take 100 episode averages and plot them too
    if len(objective_val) >= 100:
        means = objective_val.unfold(0, 100, 1).mean(1).view(-1)
        loss_val = torch.tensor(loss_hisroty, dtype=torch.float)
        means = torch.cat((torch.zeros(99), means))
        ax_one.plot(means.numpy(), color='navy')

    if len(loss_hisroty) > 0:
        loss_val = torch.tensor(loss_hisroty, dtype=torch.float)
        ax_loss.plot(loss_val.numpy(), color='tomato')

    plt.pause(0.001)


def beta_scheduler():
    def function(frame_idx):
        return min(1.0, 0.4 + frame_idx * (1.0 - 0.4) / 100000)

    return function


beta_by_frame = beta_scheduler()
multi_step = 1

### Rainbow V1
# def optimize_model(beta=None):
#     if len(memory) < BATCH_SIZE:
#         return
#     # transitions = memory.sample(BATCH_SIZE)
#     # transitions = memory.sample(BATCH_SIZE, beta)
#     # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
#     # detailed explanation). This linerts batch-array of Transitions
#     # to Transition of batch-arrays.
#     # batch = Transition(*zip(*transitions))
#
#     state, next_state, action, reward, weights, indices = memory.sample(BATCH_SIZE, beta)
#     state = torch.FloatTensor(np.float32(state)).to(device)
#     next_state = torch.FloatTensor(np.float32(next_state)).to(device)
#     action = torch.LongTensor(action).to(device)
#     reward = torch.FloatTensor(reward).to(device)
#     weights = torch.FloatTensor(weights).to(device)
#
#     q_values = policy_net(state)
#     target_next_q_values = target_net(next_state)
#
#     q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
#
#     next_q_values = policy_net(next_state)
#     next_actions = next_q_values.max(1)[1].unsqueeze(1)
#     next_q_value = target_next_q_values.gather(1, next_actions).squeeze(1)
#
#     expected_q_value = reward + (GAMMA ** multi_step) * next_q_value * (1 - done)
#
#
#     # Compute a mask of non-final states and concatenate the batch elements
#     # (a final state would've been the one after which simulation ended)
#     # non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
#     #                                         batch.next_state)), device=device, dtype=torch.bool)
#     # non_final_next_states = torch.cat([s for s in batch.next_state
#     #                                    if s is not None])
#     # state_batch = torch.cat(batch.state)
#     # action_batch = torch.cat(batch.action)
#     # reward_batch = torch.cat(batch.reward)
#     # weights_batch = torch.cat(batch.weights)
#     # indices_batch = torch.cat(batch.indices)
#
#     # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
#     # columns of actions taken. These are the actions which would've been taken
#     # for each batch state according to policy_net
#     # state_thr_policy_net = policy_net(
#     #     state_batch).view(BATCH_SIZE, output, n_actions)
#     # state_action_values = state_thr_policy_net.gather(
#     #     2, action_batch.view(BATCH_SIZE, output, 1).to(dtype=torch.int64))
#     # state_action_values = policy_net(state_batch).gather(1, action_batch)
#
#     # Compute V(s_{t+1}) for all next states.
#     # Expected values of actions for non_final_next_states are computed based
#     # on the "older" target_net; selecting their best reward with max(1)[0].
#     # This is merged based on the mask, such that we'll have either the expected
#     # state value or 0 in case the state was final.
#
#
#
#     # next_state_values = torch.zeros((BATCH_SIZE, output), device=device)
#     # # next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
#     # next_state_thr_policy_net = target_net(
#     #     non_final_next_states).view(BATCH_SIZE, output, n_actions)
#     # next_state_values[non_final_mask, :] = next_state_thr_policy_net.max(2)[
#     #     0].detach()
#
#     # Compute the expected Q values
#     # expected_state_action_values = (next_state_values * GAMMA) + reward_batch.unsqueeze(1).repeat(1, output)
#
#     # Compute Huber loss
#     # criterion = nn.SmoothL1Loss()
#     # loss = criterion(state_action_values.view(1, -1).float(), expected_state_action_values.view(1, -1).float())
#
#     loss = criterion(q_value, expected_q_value.detach(), reduction='none')
#     prios = torch.abs(loss) + 1e-5
#     loss = (loss * weights).mean()
#
#     # Optimize the model
#     optimizer.zero_grad()
#     loss.backward()
#
#     # memory.update_priorities(samples['indexes'], prios)
#
#     memory.update_priorities(indices, prios.data.cpu().numpy())
#
#     optimizer.step()
#
#     return loss

def optimize_model(beta=None):
    # if len(memory) < BATCH_SIZE:
    #    return

    samples = memory.sample(BATCH_SIZE, beta)
    state = torch.FloatTensor(np.float32(samples['obs'])).to(device)
    next_state = torch.FloatTensor(np.float32(samples['next_obs'])).to(device)
    action = torch.LongTensor(samples['action']).to(device)
    reward = torch.FloatTensor(samples['reward']).to(device)
    weights = torch.FloatTensor(samples['weights']).to(device)

    q_values = policy_net(state)
    target_next_q_values = target_net(next_state)

    q_value = q_values.view(12,3).gather(1, action).squeeze(1)

    next_q_values = policy_net(next_state)
    next_actions = next_q_values.max(1)[1].unsqueeze(1)
    next_q_value = target_next_q_values.gather(1, next_actions).squeeze(1)

    expected_q_value = reward.repeat(12) + (GAMMA ** multi_step) * next_q_value * (1 - done)

    # loss = criterion(q_value.squeeze(0), expected_q_value.detach(), reduction='none')
    loss = criterion(q_value.squeeze(0), expected_q_value.detach())
    # prios = torch.abs(loss.cpu().numpy()) + 1e-5
    prios = torch.abs(loss) + 1e-5
    loss = (loss * weights).mean()

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()

    memory.update_priorities(torch.tensor(samples['indexes']), prios)

    optimizer.step()

    return loss


def potter(input, seed_tensor):
    randomized_input = torch.empty(size=seed_tensor.shape)
    for i, index in zip(range(seed_tensor.shape[0]), seed_tensor):
        randomized_input[i, :] = input[index]

    return randomized_input


def save_model(path):
    torch.save({
        'policy': policy_net.state_dict(),
        'target': target_net.state_dict(),
        'total_reward': reward_hisroty,
        'loss_hisroty': loss_hisroty,
    }, path)


def load_model(path):
    checkpoint = torch.load(path)
    policy_net.load_state_dict(checkpoint['policy'])
    target_net.load_state_dict(checkpoint['target'])
    reward_hisroty = checkpoint['total_reward']
    loss_hisroty = checkpoint['loss_hisroty']

    return reward_hisroty, loss_hisroty


if __name__ == '__main__':
    total_reward = []
    total_loss = []
    num_episodes = int(sys.maxsize / 2)

    potter_indexies = []
    for i in range(custom_seed_val, custom_seed_val + input_stack_size):
        random.seed(i)
        potter_indexies.append(random.sample(range(inputs), inputs))
    potter_indexies = torch.tensor(potter_indexies, device='cuda')

    # model_path = "/home/elitedog/VTT_pybullet_DQN_model_compressed_for_jeff/_agent_data/" + aim_agent_version  # Jeff's
    model_path = cur_path + '/_agent_data/' + aim_agent_version

    if os.path.exists(model_path):
        reward_hisroty, loss_hisroty = load_model(model_path)

    for i_episode in range(num_episodes):

        # Initialize the environment and state
        state = torch.tensor(env.reset(), device=device)
        state[4:8] = state[4:8]*100
        state = potter(state, potter_indexies)
        state = state.view(1, 1, state.shape[0], state.shape[1])
        

        for t in count():

            # Select and perform an action
            action = select_action(state)
            next_state, reward, done, _ = env.step(action)

            # survive_reward = math.exp(-0.0001*t)
            survive_reward = 0.0
            reward = torch.tensor([reward + survive_reward], device=device)
            total_reward.append(reward)

            # Move to the next state
            next_state = torch.tensor(next_state, device=device)
            next_state[4:8] = next_state[4:8]*100
            next_state = potter(next_state, potter_indexies)
            next_state = next_state.view(1, 1, next_state.shape[0], next_state.shape[1])

            # Add tuple to memory
            memory.push(state, action, reward, next_state)

            # storage next_state as state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            beta = beta_by_frame(t)
            loss = optimize_model(beta)

            if loss != None:
                total_loss.append(loss)

            if done or t > max_episode_len:
                reward_hisroty.append(sum(total_reward) / t)
                if (len(total_loss) != 0) and (len(loss_hisroty) != 0):
                    loss_hisroty.append(sum(total_loss) / len(total_loss))
                else:
                    loss_hisroty.append(0)
                total_reward = []
                total_loss = []
                print("substep ======================= ", t,
                      "\t|  i_episode ===================== ", i_episode)
                time.sleep(0.01)
                break

        # Update the target network, copying all weights and biases in DQN
        if i_episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
            steps_done = 0

        if i_episode % 1 == 0:  # (TARGET_UPDATE*10)
            plot_durations()
            save_model(model_path)

    print('Complete')
    env.render()
    env.close()
    plt.ioff()
    plt.show()
