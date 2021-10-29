#!/usr/bin/env python
# coding: utf-8

# In[1]:


# In[2]:


import numpy as np
import math
import multiprocessing as mp
import cma

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from collections import namedtuple, deque

from PIL import Image
import os
import os.path
import errno
import codecs
import copy
import argparse

import cv2
import random
import gym
import matplotlib.pyplot as plt
from typing import List, Tuple
from statistics import mean

from bindsnet.network import Network
from bindsnet.network.nodes import Input, LIFNodes
from bindsnet.network.topology import Connection, Conv2dConnection, MaxPool2dConnection
from bindsnet.network.monitors import Monitor
from bindsnet.network.nodes import AbstractInput
from bindsnet.learning import PostPre, WeightDependentPostPre, Hebbian, MSTDP, MSTDPET

from bindsnet.environment import GymEnvironment
from torch import Tensor
from bindsnet.analysis.plotting import plot_spikes


# In[3]:


torch.manual_seed(0)
np.random.seed(0)
print("torch.cuda.device_count()", torch.cuda.device_count())


# In[4]:


Args = namedtuple('Args', ['epochs', 'batch_size', 'lr', 'cuda', 'seed', 'log_interval', 'env', 'fitness_episodes', 'gamma', 'n_episode', 'update_rule'])


# In[5]:


args = Args(batch_size=1000, epochs=30, lr=0.001, cuda=False, seed=0, log_interval=10, gamma=0.99, env='BreakoutDeterministic-v4', fitness_episodes=2, n_episode=1000, update_rule='MSTDP')


# In[6]:


torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


# In[7]:


rules = {
    "PostPre": PostPre,
    "WeightDependentPostPre": WeightDependentPostPre,
    "Hebbian": Hebbian,
    "MSTDP": MSTDP,
    "MSTDPET": MSTDPET,
}
s_im, s_ax = None, None


# In[8]:


def compute_ranks(x):
  """
  Returns ranks in [0, len(x))
  Note: This is different from scipy.stats.rankdata, which returns ranks in [1, len(x)].
  (https://github.com/openai/evolution-strategies-starter/blob/master/es_distributed/es.py)
  """
  assert x.ndim == 1
  ranks = np.empty(len(x), dtype=int)
  ranks[x.argsort()] = np.arange(len(x))
  return ranks

def compute_centered_ranks(x):
  """
  https://github.com/openai/evolution-strategies-starter/blob/master/es_distributed/es.py
  """
  y = compute_ranks(x.ravel()).reshape(x.shape).astype(np.float32)
  y /= (x.size - 1)
  y -= .5
  return y

def compute_weight_decay(weight_decay, model_param_list):
  model_param_grid = np.array(model_param_list)
  return - weight_decay * np.mean(model_param_grid * model_param_grid, axis=1)

class CMAES:
  '''CMA-ES wrapper.'''
  def __init__(self, num_params,      # number of model parameters
               sigma_init=0.10,       # initial standard deviation
               popsize=255):          # population size

    self.num_params = num_params
    self.sigma_init = sigma_init
    self.popsize = popsize

    self.solutions = None

    import cma
    self.es = cma.CMAEvolutionStrategy( self.num_params * [0],
                                        self.sigma_init,
                                        {'popsize': self.popsize})

  def rms_stdev(self):
    sigma = self.es.result[6]
    return np.mean(np.sqrt(sigma*sigma))

  def ask(self):
    '''returns a list of parameters'''
    self.solutions = np.array(self.es.ask())
    return self.solutions

  def tell(self, reward_table_result):
    reward_table = reward_table_result
    self.es.tell(self.solutions, (-reward_table).tolist()) # convert minimizer to maximizer.

  def done(self):
    return self.es.stop()

  def current_param(self):
    return self.es.result[5] # mean solution, presumably better with noise

  def best_param(self):
    return self.es.result[0] # best evaluated solution

  def result(self): # return best params so far, along with historically best reward, curr reward, sigma
    r = self.es.result
    return (r[0], -r[1], -r[1], r[6])

class SimpleES:
  '''Simple Evolution Strategies.'''
  def __init__(self, num_params,      # number of model parameters
               sigma_init=0.10,       # initial standard deviation
               sigma_alpha=0.20,      # learning rate for standard deviation
               sigma_decay=0.999,     # anneal standard deviation
               sigma_limit=0.01,      # stop annealing if less than this
               popsize=255,           # population size
               elite_ratio=0.1,       # percentage of the elites
               done_threshold=1e-6,   # threshold when we say we are done
               average_baseline=True, # set baseline to average of batch
               forget_best=True):     # only use the best from latest generation

    self.num_params = num_params
    self.sigma_init = sigma_init
    self.sigma_alpha = sigma_alpha
    self.sigma_decay = sigma_decay
    self.sigma_limit = sigma_limit
    self.popsize = popsize
    self.average_baseline = average_baseline
    if self.average_baseline:
      assert (self.popsize & 2), "Population size must be even"
      self.batch_size = int(self.popsize / 2)
    else:
      assert (self.popsize & 1), "Population size must be odd"
      self.batch_size = int((self.popsize - 1) / 2)
    self.elite_ratio = elite_ratio
    self.elite_popsize = int(self.popsize * self.elite_ratio)
    self.forget_best = forget_best
    self.batch_reward = np.zeros(self.batch_size * 2)
    self.mu = np.zeros(self.num_params)
    self.sigma = np.ones(self.num_params) * self.sigma_init
    self.curr_best_mu = np.zeros(self.num_params)
    self.best_mu = np.zeros(self.num_params)
    self.best_reward = 0
    self.first_interation = True
    self.done_threshold = done_threshold

  def rms_stdev(self):
    sigma = self.sigma
    return np.mean(np.sqrt(sigma*sigma))

  def ask(self):
    '''returns a list of parameters'''
    # antithetic sampling
    self.epsilon = np.random.randn(self.batch_size, self.num_params) * self.sigma.reshape(1, self.num_params)
    self.epsilon_full = np.concatenate([self.epsilon, - self.epsilon])
    if self.average_baseline:
      epsilon = self.epsilon_full
    else:
      # first population is mu, then positive epsilon, then negative epsilon
      epsilon = np.concatenate([np.zeros((1, self.num_params)), self.epsilon_full])
    solutions = self.mu.reshape(1, self.num_params) + epsilon
    return solutions

  def tell(self, reward_table_result):
    # input must be a numpy float array
    assert(len(reward_table_result) == self.popsize), "Inconsistent reward_table size reported."

    reward_table = reward_table_result

    reward_offset = 1
    if self.average_baseline:
      b = np.mean(reward_table)
      reward_offset = 0
    else:
      b = reward_table[0] # baseline

    reward = reward_table[reward_offset:]
    idx = np.argsort(reward)[::-1][0:self.elite_popsize]

    best_reward = reward[idx[0]]
    if (best_reward > b or self.average_baseline):
      best_mu = self.mu + self.epsilon_full[idx[0]]
      best_reward = reward[idx[0]]
    else:
      best_mu = self.mu
      best_reward = b

    self.curr_best_reward = best_reward
    self.curr_best_mu = best_mu

    if self.first_interation:
      self.first_interation = False
      self.best_reward = self.curr_best_reward
      self.best_mu = best_mu
    else:
      if self.forget_best or (self.curr_best_reward > self.best_reward):
        self.best_mu = best_mu
        self.best_reward = self.curr_best_reward

    # adaptive sigma
    # normalization
    stdev_reward = reward.std()
    epsilon = self.epsilon
    sigma = self.sigma
    S = ((epsilon * epsilon - (sigma * sigma).reshape(1, self.num_params)) / sigma.reshape(1, self.num_params))
    reward_avg = (reward[:self.batch_size] + reward[self.batch_size:]) / 2.0
    rS = reward_avg - b
    delta_sigma = (np.dot(rS, S)) / (2 * self.batch_size * stdev_reward)

    # move mean to the average of the best idx means
    self.mu += self.epsilon_full[idx].mean(axis=0)

    # adjust sigma according to the adaptive sigma calculation
    change_sigma = self.sigma_alpha * delta_sigma
    change_sigma = np.minimum(change_sigma, self.sigma)
    change_sigma = np.maximum(change_sigma, - 0.5 * self.sigma)
    self.sigma += change_sigma
    self.sigma[self.sigma > self.sigma_limit] *= self.sigma_decay

  def done(self):
    return (self.rms_stdev() < self.done_threshold)

  def current_param(self):
    return self.curr_best_mu

  def best_param(self):
    return self.best_mu

  def result(self): # return best params so far, along with historically best reward, curr reward, sigma
    return (self.best_mu, self.best_reward, self.curr_best_reward, self.sigma)

class SimpleGA:
  '''Simple Genetic Algorithm.'''
  def __init__(self, num_params,      # number of model parameters
               sigma_init=0.1,        # initial standard deviation
               sigma_decay=0.999,     # anneal standard deviation
               sigma_limit=0.01,      # stop annealing if less than this
               popsize=255,           # population size
               elite_ratio=0.1,       # percentage of the elites
               forget_best=False,     # forget the historical best elites
               done_threshold=1e-6):  # threshold when we say we are done

    self.num_params = num_params
    self.sigma_init = sigma_init
    self.sigma_decay = sigma_decay
    self.sigma_limit = sigma_limit
    self.popsize = popsize

    self.elite_ratio = elite_ratio
    self.elite_popsize = int(self.popsize * self.elite_ratio)

    self.sigma = self.sigma_init
    self.elite_params = np.zeros((self.elite_popsize, self.num_params))
    self.elite_rewards = np.zeros(self.elite_popsize)
    self.best_param = np.zeros(self.num_params)
    self.best_reward = 0
    self.first_iteration = True
    self.forget_best = forget_best
    self.done_threshold = done_threshold

  def rms_stdev(self):
    return self.sigma # same sigma for all parameters.

  def ask(self):
    '''returns a list of parameters'''
    # antithetic sampling
    self.epsilon = np.random.randn(self.popsize, self.num_params) * self.sigma
    solutions = []

    def mate(a, b):
      c = np.copy(a)
      idx = np.where(np.random.rand((c.size)) > 0.5)
      c[idx] = b[idx]
      return c

    elite_range = range(self.elite_popsize)
    for i in range(self.popsize):
      idx_a = np.random.choice(elite_range)
      idx_b = np.random.choice(elite_range)
      child_params = mate(self.elite_params[idx_a], self.elite_params[idx_b])
      solutions.append(child_params + self.epsilon[i])

    solutions = np.array(solutions)
    self.solutions = solutions

    return solutions

  def tell(self, reward_table_result):
    # input must be a numpy float array
    assert(len(reward_table_result) == self.popsize), "Inconsistent reward_table size reported."

    if (not self.forget_best or self.first_iteration):
      reward = reward_table_result
      solution = self.solutions
    else:
      reward = np.concatenate([reward_table_result, self.elite_rewards])
      solution = np.concatenate([self.solutions, self.elite_params])

    idx = np.argsort(reward)[::-1][0:self.elite_popsize]

    self.elite_rewards = reward[idx]
    self.elite_params = solution[idx]

    self.curr_best_reward = self.elite_rewards[0]

    if self.first_iteration or (self.curr_best_reward > self.best_reward):
      self.first_iteration = False
      self.best_reward = self.elite_rewards[0]
      self.best_param = np.copy(self.elite_params[0])

    if (self.sigma > self.sigma_limit):
      self.sigma *= self.sigma_decay

  def done(self):
    return (self.rms_stdev() < self.done_threshold)

  def current_param(self):
    return self.elite_params[0]

  def best_param(self):
    return self.best_param

  def result(self): # return best params so far, along with historically best reward, curr reward, sigma
    return (self.best_param, self.best_reward, self.curr_best_reward, self.sigma)

class OpenES:
  ''' Basic Version of OpenAI Evolution Strategies.'''
  def __init__(self, num_params,             # number of model parameters
               sigma_init=0.1,               # initial standard deviation
               sigma_decay=0.999,            # anneal standard deviation
               sigma_limit=0.01,             # stop annealing if less than this
               learning_rate=0.001,          # learning rate for standard deviation
               learning_rate_decay = 0.9999, # annealing the learning rate
               learning_rate_limit = 0.001,  # stop annealing learning rate
               popsize=255,                  # population size
               antithetic=False,             # whether to use antithetic sampling
               forget_best=True):           # forget historical best

    self.num_params = num_params
    self.sigma_decay = sigma_decay
    self.sigma = sigma_init
    self.sigma_limit = sigma_limit
    self.learning_rate = learning_rate
    self.learning_rate_decay = learning_rate_decay
    self.learning_rate_limit = learning_rate_limit
    self.popsize = popsize
    self.antithetic = antithetic
    if self.antithetic:
      assert (self.popsize & 2), "Population size must be even"
      self.half_popsize = int(self.popsize / 2)

    self.reward = np.zeros(self.popsize)
    self.mu = np.zeros(self.num_params)
    self.best_mu = np.zeros(self.num_params)
    self.best_reward = 0
    self.first_interation = True
    self.forget_best = forget_best

  def rms_stdev(self):
    sigma = self.sigma
    return np.mean(np.sqrt(sigma*sigma))

  def ask(self):
    '''returns a list of parameters'''
    # antithetic sampling
    if self.antithetic:
      self.epsilon_half = np.random.randn(self.half_popsize, self.num_params)
      self.epsilon = np.concatenate([self.epsilon_half, - self.epsilon_half])
    else:
      self.epsilon = np.random.randn(self.popsize, self.num_params)

    self.solutions = self.mu.reshape(1, self.num_params) + self.epsilon * self.sigma

    return self.solutions

  def tell(self, reward):
    # input must be a numpy float array
    assert(len(reward) == self.popsize), "Inconsistent reward_table size reported."

    idx = np.argsort(reward)[::-1]

    best_reward = reward[idx[0]]
    best_mu = self.solutions[idx[0]]

    self.curr_best_reward = best_reward
    self.curr_best_mu = best_mu

    if self.first_interation:
      self.first_interation = False
      self.best_reward = self.curr_best_reward
      self.best_mu = best_mu
    else:
      if self.forget_best or (self.curr_best_reward > self.best_reward):
        self.best_mu = best_mu
        self.best_reward = self.curr_best_reward

    # main bit:
    # standardize the rewards to have a gaussian distribution
    normalized_reward = (reward - np.mean(reward)) / np.std(reward)
    self.mu += self.learning_rate/(self.popsize*self.sigma)*np.dot(self.epsilon.T, normalized_reward)

    # adjust sigma according to the adaptive sigma calculation
    if (self.sigma > self.sigma_limit):
      self.sigma *= self.sigma_decay

    if (self.learning_rate > self.learning_rate_limit):
      self.learning_rate *= self.learning_rate_decay

  def done(self):
    return False

  def current_param(self):
    return self.curr_best_mu

  def best_param(self):
    return self.best_mu

  def result(self): # return best params so far, along with historically best reward, curr reward, sigma
    return (self.best_mu, self.best_reward, self.curr_best_reward, self.sigma)


# In[9]:


class Net(object):
    def __init__(self) -> None:
        """SQN Network
        """
        input_dim, shape = 80 * 80, [1, 1, 80, 80]
        output_dim = 4
        hidden_dim = 10

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.network = Network(dt=1.0)
        self.learning_rule = rules.get(args.update_rule)
        self.time = int(self.network.dt)

        # To solve tensor formatting issues
        if args.update_rule == 'MSTDP' or args.update_rule == "HMSTDP":
            self.input = Input(n=input_dim, traces=True)
        else:
            self.input = Input(n=input_dim, shape=shape, traces=True)

        # Heterogeneous LIFNodes
        self.hidden = LIFNodes(n=hidden_dim, traces=True)
        self.output = LIFNodes(n=output_dim, traces=True)

        # First connection
        self.connection_input_hidden = Connection(
            source=self.input,
            target=self.hidden,
            update_rule=self.learning_rule,
            wmin=0,
            wmax=1,
            nu=args.gamma
        )

        # Hidden recurrent connection
#         self.connection_hidden_hidden = Connection(
#             source=self.hidden,
#             target=self.hidden,
#             update_rule=self.learning_rule,
#             wmin=0,
#             wmax=1,
#             nu=args.gamma
#         )

        # Hidden layer to Output
        self.connection_hidden_output = Connection(
            source=self.hidden,
            target=self.output,
            update_rule=self.learning_rule,
            wmin=0,
            wmax=1,
            nu=args.gamma
        )

        self.network.add_layer(
            layer=self.input, name="Input"
        )
        self.network.add_layer(
            layer=self.hidden, name="Hidden"
        )
        self.network.add_layer(
            layer=self.output, name="Output"
        )

        self.network.add_connection(
            connection=self.connection_input_hidden,
            source="Input",
            target="Hidden"
        )
#         self.network.add_connection(
#             connection=self.connection_hidden_hidden,
#             source="Hidden",
#             target="Hidden"
#         )
        self.network.add_connection(
            connection=self.connection_hidden_output,
            source="Hidden",
            target="Output"
        )

        self.inputs = [
            name
            for name, layer in self.network.layers.items()
            if isinstance(layer, AbstractInput)
        ]

        # To record outputs
        self.network.add_monitor(
            Monitor(self.output, ["s"], time=self.time),
            name="output_monitor"
        )
        self.network.add_monitor(
            Monitor(self.hidden, ["s"], time=self.time),
            name="hidden_monitor"
        )
        self.network.add_monitor(
            Monitor(self.input, ["s"], time=self.time),
            name="input_monitor"
        )

        self.spike_record = {
            "Output": torch.zeros((self.time, output_dim)).to(
                self.device
            )
        }

    def run(self, inputs: dict[str, torch.Tensor], reward: [float, torch.Tensor], **kwargs) -> None:
        self.network.train(mode=True)
        return self.network.run(inputs=inputs, time=self.time, reward=reward, **kwargs)


# In[10]:


class Agent(object):

    def __init__(self, model) -> None:
        """Agent class that chooses an action and trains
        """
        self.net = model

    def get_action(self) -> int:
        """Returns an action index
        """
        scores = self.get_Q()
        probabilities = torch.softmax(scores, dim=0)
        return torch.multinomial(probabilities, num_samples=1).item()

        _, argmax = torch.max(torch.flatten(scores), dim=0)
        return argmax.item()

    def get_Q(self) -> Tensor:
        """Returns `Q-value` based on output layer's spikes
        """
        return torch.sum(self.sqn.spike_record["Output"], dim=0)


# In[11]:


def clip_reward(reward):
    """Clip reward so that it's in [-1, 1]
    """
    if reward < -1:
        reward = -1
    elif reward > 1:
        reward = 1
    return reward


# In[12]:


def preprocess(states: np.ndarray):
    """Preprocesses gym state
    """
    # Crop
    states = states[34:194, 0:160, :]

    # Convert to grayscale
    states = cv2.cvtColor(states, cv2.COLOR_RGB2GRAY)

    # Subsample to 80x80
    states = cv2.resize(states, (80, 80))
    states = cv2.threshold(states, 0, 1, cv2.THRESH_BINARY)[1]

    states = states.reshape(1, states.shape[0], states.shape[1])
    return states


# In[13]:


def play_episode(env: GymEnvironment, agent: Agent) -> int:
    """Play an epsiode and train
    Args:
        env (gym.Env): gym environment (CartPole-v0)
        agent (Agent): agent will train and get action
    Returns:
        int: reward
    """
    env.reset()
    agent.net.network.reset_state_variables()

    done = False
    total_reward = 0

    while not done:
        env.render()
        # Select an action
        a = agent.get_action()
        # Update the state according to action a
        s, r, done, info = env.step(a)

        r = clip_reward(r)

        # Tensor shape configuration
        if FLAGS.update_rule == 'MSTDP' or FLAGS.update_rule == "HMSTDP":
            s = s.flatten()

        s_shape = [1] * len(s.shape[1:])

        # Run the agent for time t on state s with reward r
        inputs = {k: s.repeat(agent.sqn.time, *s_shape) for k in agent.sqn.inputs}
        agent.sqn.run(inputs=inputs, reward=r)

        # Update output spikes
        if agent.sqn.output is not None:
            agent.sqn.spike_record["Output"] = (
                agent.sqn.network.monitors["output_monitor"].get("s").float()
            )

        total_reward += r

        global s_im
        global s_ax
        s_im, s_ax = plot_spikes({
            "input": agent.sqn.network.monitors["input_monitor"].get("s"),
            "hidden": agent.sqn.network.monitors["hidden_monitor"].get("s"),
            "output": agent.sqn.network.monitors["output_monitor"].get("s")
        }, ims=s_im, axes=s_ax)

    return total_reward


# In[14]:


def get_env_dim(env: gym.Env) -> Tuple[int, int]:
    """Returns input_dim & output_dim
    Args:
        env (gym.Env): gym Environment
    Returns:
        int: input_dim
        int: output_dim
    """
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n

    return input_dim, output_dim


# In[15]:


NPOPULATION = 101
weight_decay_coef = 0.1


# In[16]:


'''
models = []
for i in range(NPOPULATION):
  model = Net()
  if args.cuda:
    model.cuda()
  model.eval()
  models.append(model)
'''

model = Net()
if args.cuda:
    model.cuda()

orig_model = copy.deepcopy(model)


# In[17]:


# get init params
orig_params = []
model_shapes = []
for param in orig_model.network.parameters():
    p = param.data.cpu().numpy()
    model_shapes.append(p.shape)
    orig_params.append(p.flatten())
orig_params_flat = np.concatenate(orig_params)
NPARAMS = len(orig_params_flat)
print(NPARAMS)


# In[18]:


def update_model(flat_param, model, model_shapes):
    idx = 0
    i = 0
    for param in model.network.parameters():
        delta = np.product(model_shapes[i])
        block = flat_param[idx:idx+delta]
        block = np.reshape(block, model_shapes[i])
        i += 1
        idx += delta
        block_data = torch.from_numpy(block).float()
        if args.cuda:
          block_data = block_data.cuda()
        param.data = block_data


# In[19]:


def evaluate(model, env, print_mode=True):
    agent = Agent(model)
    total_reward = 0
    for i in range(args.fitness_episodes):
        reward = play_episode(env, agent)
        total_reward += reward

    if print_mode:
        print('\nAverage reward: {:.4f}'.format(total_reward/args.fitness_episodes))

    return total_reward


# In[ ]:


"""
es = SimpleES(NPARAMS,
              popsize=NPOPULATION,
              sigma_init=0.01,
              sigma_decay=0.999,
              sigma_alpha=0.2,
              sigma_limit=0.001,
              elite_ratio=0.1,
              average_baseline=False,
              forget_best=True
             )
"""
es = CMAES(NPARAMS, sigma_init=0.01, popsize=NPOPULATION)


# In[ ]:


env = gym.make(args.env)
try:
    for epoch in range(1, args.epochs):

        # train loop
        model.eval()
        solutions = es.ask()
        reward = np.zeros(es.popsize)

        for i in range(es.popsize):
            update_model(solutions[i], model, model_shapes)
            reward[i] = evaluate(model, env, print_mode=False)

        best_raw_reward = reward.max()
        es.tell(reward)
        result = es.result()

        curr_solution = es.current_param()
        update_model(curr_solution, model, model_shapes)
        solution_reward = evaluate(model, env, print_mode=True)
finally:
    env.close()
