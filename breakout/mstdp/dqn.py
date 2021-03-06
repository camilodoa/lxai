"""
CNN DQN algorithm that solves BreakoutDeterministic-v4

DQN in PyTorch

@author: @kkweon, @camilodoa
"""
import argparse

import cv2
import torch
import torch.nn
import numpy as np
import random
import gym
from collections import namedtuple
import matplotlib.pyplot as plt
from typing import List, Tuple
from analysis import save_list
from collections import deque
from statistics import mean

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--gamma",
                    type=float,
                    default=0.99,
                    help="Discount rate for Q_target")
parser.add_argument("--env",
                    type=str,
                    default="BreakoutDeterministic-v4",
                    help="Gym environment name")
parser.add_argument("--n-episode",
                    type=int,
                    default=1000,
                    help="Number of epsidoes to run")
parser.add_argument("--batch-size",
                    type=int,
                    default=32,
                    help="Mini-batch size")
parser.add_argument("--hidden-dim",
                    type=int,
                    default=12,
                    help="Hidden dimension")
parser.add_argument("--capacity",
                    type=int,
                    default=50000,
                    help="Replay memory capacity")
parser.add_argument("--max-episode",
                    type=int,
                    default=1000,
                    help="e-Greedy target episode (eps will be the lowest at this episode)")
parser.add_argument("--min-eps",
                    type=float,
                    default=0.01,
                    help="Min epsilon")
FLAGS = parser.parse_args()


class DQN(torch.nn.Module):
    def __init__(self, input_shape: [int], output_dim: int, hidden_dim: int, batch_size: int) -> None:
        """DQN Network
        """
        super(DQN, self).__init__()
        w, h, c = input_shape
        kernel_size = 3
        padding = 1
        stride = 1
        out_channels = 3

        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            torch.nn.ReLU(),
            torch.nn.Flatten()
        )

        self.layer2 = torch.nn.Sequential(
            torch.nn.Linear(19200, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.25)
        )

        self.final = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, output_dim),
            torch.nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns a Q_value
        """
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.final(x)

        return x


Transition = namedtuple("Transition",
                        field_names=["state", "action", "reward", "next_state", "done"])


class ReplayMemory(object):

    def __init__(self, capacity: int) -> None:
        """Replay memory class
        """
        self.capacity = capacity
        self.cursor = 0
        self.memory = []

    def push(self,
             state: np.ndarray,
             action: int,
             reward: int,
             next_state: np.ndarray,
             done: bool) -> None:
        """Creates `Transition` and insert
        """
        if len(self) < self.capacity:
            self.memory.append(None)

        self.memory[self.cursor] = Transition(state,
                                              action, reward, next_state, done)
        self.cursor = (self.cursor + 1) % self.capacity

    def pop(self, batch_size: int) -> List[Transition]:
        """Returns a randomly sampled minibatch
        """
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        """Returns the length """
        return len(self.memory)


class Agent(object):

    def __init__(self, input_shape: [int], output_dim: int, hidden_dim: int, batch_size: int) -> None:
        """Agent class
        """
        self.dqn = DQN(input_shape, output_dim, hidden_dim, batch_size)
        self.input_dim = input_shape
        self.output_dim = output_dim

        self.loss_fn = torch.nn.MSELoss()
        self.optim = torch.optim.Adam(self.dqn.parameters())

    def _to_variable(self, x: np.ndarray) -> torch.Tensor:
        """torch.Variable syntax helper
        """
        return torch.autograd.Variable(torch.Tensor(x))

    def get_action(self, states: np.ndarray, eps: float) -> int:
        """Returns an action
        """
        if np.random.rand() < eps:
            return np.random.choice(self.output_dim)
        else:
            self.dqn.train(mode=False)
            scores = self.get_Q(np.array([states]))
            _, argmax = torch.max(scores.data, 1)
            return int(argmax.numpy())

    def get_Q(self, states: np.ndarray) -> torch.FloatTensor:
        """Returns `Q-value`
        """
        states = self._to_variable(states)
        self.dqn.train(mode=False)

        return self.dqn(states)

    def train(self, Q_pred: torch.FloatTensor, Q_true: torch.FloatTensor) -> float:
        """Computes `loss` and backpropagation
        """
        self.dqn.train(mode=True)
        self.optim.zero_grad()
        loss = self.loss_fn(Q_pred, Q_true)
        loss.backward()
        self.optim.step()
        return loss


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


def train_helper(agent: Agent, minibatch: List[Transition], gamma: float) -> float:
    """ Train on minibatch data
    """
    states = np.array([x.state for x in minibatch])
    actions = np.array([x.action for x in minibatch])
    rewards = np.array([x.reward for x in minibatch])
    next_states = np.array([x.next_state for x in minibatch])
    Q_predict = agent.get_Q(states)
    Q_target = Q_predict.clone().data.numpy()
    Q_target[np.arange(len(Q_target)), actions] = rewards + gamma * np.max(agent.get_Q(next_states).data.numpy(),
                                                                           axis=1)
    Q_target = agent._to_variable(Q_target)

    return agent.train(Q_predict, Q_target)


def play_episode(env: gym.Env,
                 agent: Agent,
                 replay_memory: ReplayMemory,
                 eps: float,
                 batch_size: int) -> int:
    """Play an episode
    """
    s = env.reset()
    s = preprocess(s)
    done = False
    total_reward = 0

    while not done:
        a = agent.get_action(s, eps)
        s2, r, done, info = env.step(a)
        env.render()

        # Preprocessing step
        s2 = preprocess(s2)
        r = clip_reward(r)

        total_reward += r

        if done:
            r = -1
        replay_memory.push(s, a, r, s2, done)

        if len(replay_memory) > batch_size:
            minibatch = replay_memory.pop(batch_size)
            train_helper(agent, minibatch, FLAGS.gamma)

        s = s2

    return total_reward


def get_env_dim(env: gym.Env) -> Tuple[int, int]:
    """Returns input_dim & output_dim
    """
    input_dim = env.observation_space.shape
    output_dim = env.action_space.n

    return input_dim, output_dim


def epsilon_annealing(epsiode: int, max_episode: int, min_eps: float) -> float:
    """Returns ???? for ????-annealing
    1.0---|\
          | \
          |  \
    min_e +---+------->
              |
              max_episode
    """

    slope = (min_eps - 1.0) / max_episode
    return max(slope * epsiode + 1.0, min_eps)

def clip_reward(reward):
    """Clip reward so that it's in [-1, 1]
    """
    if reward < -1:
        reward = -1
    elif reward > 1:
        reward = 1
    return reward


def main(save: bool = True, plot: bool = False) -> None:
    """Main
    """
    try:
        env = gym.make(FLAGS.env)
        env = gym.wrappers.Monitor(env, directory="monitors", force=True)

        average_rewards = []
        q = deque(maxlen=100)

        input_dim, output_dim = get_env_dim(env)

        agent = Agent((80, 80, 1) , output_dim, FLAGS.hidden_dim, FLAGS.batch_size)
        replay_memory = ReplayMemory(FLAGS.capacity)

        for i in range(FLAGS.n_episode):
            eps = epsilon_annealing(i, FLAGS.max_episode, FLAGS.min_eps)
            r = play_episode(env, agent, replay_memory, eps, FLAGS.batch_size)
            print("[Episode: {:5}] Reward: {:5} ????-greedy: {:5.2f}".format(i + 1, r, eps))

            q.append(r)
            if i % 100 == 0:
                average_rewards.append(mean(q))

        name = "DQN-cnn-{}-{}-{}-reward_clamping".format(FLAGS.env, FLAGS.n_episode, FLAGS.gamma)

        if plot:
            fig, ax = plt.subplots()
            ax.plot(average_rewards)

            ax.set(xlabel='Episode', ylabel='Reward',
                   title='DQN (CNN) performance on {}'.format(FLAGS.env))
            plt.show()

        if save:
            save_list(average_rewards, "{}.fli".format(name))

    finally:
        env.close()


if __name__ == '__main__':
    main(save=True, plot=True)
