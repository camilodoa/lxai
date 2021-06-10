"""
SQN algorithm to solve CartPole-v0

SQN in BindsNet
"""
import argparse
import torch
import random
import numpy as np
from bindsnet.network import Network
from bindsnet.network.nodes import Input, LIFNodes
from bindsnet.network.topology import Connection
from bindsnet.network.monitors import Monitor
from bindsnet.network.nodes import AbstractInput
from collections import namedtuple
from collections import deque
import gym
from bindsnet.environment import GymEnvironment
from typing import List, Tuple


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
                    default=64,
                    help="Mini-batch size")
parser.add_argument("--hidden-dim",
                    type=int,
                    default=100,
                    help="Hidden dimension")
parser.add_argument("--capacity",
                    type=int,
                    default=50000,
                    help="Replay memory capacity")
parser.add_argument("--max-episode",
                    type=int,
                    default=50,
                    help="e-Greedy target episode (eps will be the lowest at this episode)")
parser.add_argument("--min-eps",
                    type=float,
                    default=0.01,
                    help="Min epsilon")
FLAGS = parser.parse_args()


class SQN(object):
    def __init__(self, input_dim: int, shape: [int], output_dim: int, hidden_dim: int) -> None:
        """DQN Network
        Args:
            input_dim (int): `state` dimension.
                `state` is 2-D tensor of shape (n, input_dim)
            output_dim (int): Number of actions.
                Q_value is 2-D tensor of shape (n, output_dim)
            hidden_dim (int): Hidden dimension in fc layer
        """
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.network = Network(dt=1.0)
        self.time = int(self.network.dt)

        self.input = Input(n=input_dim, shape=shape, traces=True)
        self.network.add_layer(
            layer=self.input, name="Input"
        )

        self.hidden = LIFNodes(n=hidden_dim, traces=True)
        self.network.add_layer(
            layer=self.hidden, name="Hidden"
        )

        self.output = LIFNodes(n=output_dim, refrac=0, traces=True)
        self.network.add_layer(
            layer=self.output, name="Output"
        )

        # First connection
        self.connection_input_hidden = Connection(
            source=self.input,
            target=self.hidden,
            wmin=0,
            wmax=1e-1
        )
        self.network.add_connection(
            connection=self.connection_input_hidden,
            source="Input",
            target="Hidden"
        )

        # Recurrent connection in hidden layer
        self.connection_hidden_hidden= Connection(
            source=self.hidden,
            target=self.hidden,
            # w=0.025 * (torch.eye(self.hidden.n) - 1) # Self-connecting small weights
        )
        self.network.add_connection(
            connection=self.connection_hidden_hidden,
            source="Hidden",
            target="Hidden"
        )

        # Hidden layer connection to output
        self.connection_hidden_output = Connection(
            source=self.hidden,
            target=self.output,
            wmin=0, wmax=1
        )
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

        self.spike_record = {
            "Output": torch.zeros((self.time, output_dim)).to(
                self.device
            )
        }

    def run(self, inputs: dict[str, torch.Tensor], reward: [float, torch.Tensor]) -> None:
        return self.network.run(inputs=inputs, time=self.time, reward=reward)


Transition = namedtuple("Transition",
                        field_names=["state", "action", "reward", "next_state", "done"])

class ReplayMemory(object):

    def __init__(self, capacity: int) -> None:
        """Replay memory class
        Args:
            capacity (int): Max size of this memory
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
        Args:
            state (np.ndarray): 1-D tensor of shape (input_dim,)
            action (int): action index (0 <= action < output_dim)
            reward (int): reward value
            next_state (np.ndarray): 1-D tensor of shape (input_dim,)
            done (bool): whether this state was last step
        """
        if len(self) < self.capacity:
            self.memory.append(None)

        self.memory[self.cursor] = Transition(state,
                                              action, reward, next_state, done)
        self.cursor = (self.cursor + 1) % self.capacity

    def pop(self, batch_size: int) -> List[Transition]:
        """Returns a minibatch of `Transition` randomly
        Args:
            batch_size (int): Size of mini-bach
        Returns:
            List[Transition]: Minibatch of `Transition`
        """
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        """Returns the length """
        return len(self.memory)

class Agent(object):
    def __init__(self, input_dim: int, shape: [int], output_dim: int, hidden_dim: int) -> None:
        """Agent class that chooses an action and trains
        Args:
            input_dim (int): input dimension
            shape ([int]): shape of input
            output_dim (int): output dimension
            hidden_dim (int): hidden dimension
        """
        self.sqn = SQN(input_dim, shape, output_dim, hidden_dim)
        self.input_dim = input_dim
        self.output_dim = output_dim

    def get_action(self, eps: float) -> int:
        """Returns an action
        Args:
            eps (float): 𝜺-greedy for exploration
        Returns:
            int: action index
        """
        if np.random.rand() < eps:
            return np.random.choice(self.output_dim)
        else:
            scores = self.get_Q()
            probabilities = torch.softmax(scores, dim=0)
            return torch.multinomial(probabilities, num_samples=1).item()

    def get_Q(self) -> torch.FloatTensor:
        """Returns `Q-value`
        Returns:
            torch.FloatTensor: 2-D Tensor of shape (n, output_dim)
        """
        return torch.sum(self.sqn.spike_record["Output"], dim=0)

def play_episode(env: gym.Env,
                 agent: Agent,
                 # replay_memory: ReplayMemory,
                 eps: float,
                 # batch_size: int
                 ) -> int:
    """Play an epsiode and train
    Args:
        env (gym.Env): gym environment (CartPole-v0)
        agent (Agent): agent will train and get action
        replay_memory (ReplayMemory): trajectory is saved here
        eps (float): 𝜺-greedy for exploration
        batch_size (int): batch size
    Returns:
        int: reward earned in this episode
    """
    env.reset()
    done = False
    total_reward = 0

    while not done:
        env.render()
        # Select an action
        a = agent.get_action(eps)
        # Update the state according to action a
        s2, r, done, info = env.step(a)

        # Run the agent for time t on state s with reward r
        s_shape = [1] * len(s2.shape[1:])
        inputs = {k: s2.repeat(agent.sqn.time, *s_shape) for k in agent.sqn.inputs}
        agent.sqn.run(inputs=inputs, reward=r)

        # Update output spikes
        if agent.sqn.output is not None:
            agent.sqn.spike_record["Output"] = (
                agent.sqn.network.monitors["output_monitor"].get("s").float()
            )

        total_reward += r

    # Reset network variables
    agent.sqn.network.reset_state_variables()

    return total_reward

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

def epsilon_annealing(epsiode: int, max_episode: int, min_eps: float) -> float:
    """Returns 𝜺-greedy
    1.0---|\
          | \
          |  \
    min_e +---+------->
              |
              max_episode
    Args:
        epsiode (int): Current episode (0<= episode)
        max_episode (int): After max episode, 𝜺 will be `min_eps`
        min_eps (float): 𝜺 will never go below this value
    Returns:
        float: 𝜺 value
    """

    slope = (min_eps - 1.0) / max_episode
    return max(slope * epsiode + 1.0, min_eps)


def main():
    """Main
    """
    try:
        env = GymEnvironment(FLAGS.env)
        rewards = deque(maxlen=100)
        input_dim, output_dim = get_env_dim(env.env)
        agent = Agent(80 * 80, [1, 1, 80, 80], output_dim, FLAGS.hidden_dim)

        for i in range(FLAGS.n_episode):
            eps = epsilon_annealing(i, FLAGS.max_episode, FLAGS.min_eps)
            r = play_episode(env, agent, eps)
            print("[Episode: {:5}] Reward: {:5} 𝜺-greedy: {:5.2f}".format(i + 1, r, eps))

            rewards.append(r)

            if len(rewards) == rewards.maxlen:

                if np.mean(rewards) >= 200:
                    print("Game cleared in {} games with {}".format(i + 1, np.mean(rewards)))
                    break
    finally:
        env.close()

if __name__ == '__main__':
    main()