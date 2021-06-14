"""
SQN algorithm to solve CartPole-v0

Uniform SQN implementation

@author: camilodoa
"""
import argparse
import torch
import numpy as np
from bindsnet.network import Network
from bindsnet.network.nodes import Input, LIFNodes
from bindsnet.network.topology import Connection
from bindsnet.network.monitors import Monitor
from bindsnet.network.nodes import AbstractInput
from bindsnet.learning import PostPre, WeightDependentPostPre, Hebbian, MSTDP, MSTDPET, Rmax
import matplotlib.pyplot as plt
import gym
from bindsnet.environment import GymEnvironment
from typing import Tuple

from torch import Tensor

from analysis import save_list

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--env",
                    type=str,
                    default="BreakoutDeterministic-v4",
                    help="Gym environment name")
parser.add_argument("--n-episode",
                    type=int,
                    default=1000,
                    help="Number of epsidoes to run")
parser.add_argument("--hidden-dim",
                    type=int,
                    default=200,
                    help="Hidden dimension")
parser.add_argument("--max-episode",
                    type=int,
                    default=200,
                    help="e-Greedy target episode (eps will be the lowest at this episode)")
parser.add_argument("--min-eps",
                    type=float,
                    default=0.01,
                    help="Min epsilon")
parser.add_argument("--update-rule",
                    type=str,
                    default="MSTDP",
                    help="Learning rule used to update weights")
FLAGS = parser.parse_args()

# Update rules
rules = {
    "PostPre": PostPre,
    "WeightDependentPostPre": WeightDependentPostPre,
    "Hebbian": Hebbian,
    "MSTDP": MSTDP,
    "MSTDPET": MSTDPET,
    "Rmax": Rmax  # Didn't work with current configurations
}


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
        self.learning_rule = rules.get(FLAGS.update_rule) if rules.get(FLAGS.update_rule) is not None else PostPre
        self.time = int(self.network.dt)

        # To solve tensor formatting issues
        if FLAGS.update_rule == 'MSTDP':
            self.input = Input(n=input_dim, traces=True)
        else:
            self.input = Input(n=input_dim, shape=shape, traces=True)

        self.network.add_layer(
            layer=self.input, name="Input"
        )

        self.hidden = LIFNodes(n=hidden_dim, refrac=0, traces=True)
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
            update_rule=self.learning_rule,
            wmin=-1,
            wmax=1
        )
        self.network.add_connection(
            connection=self.connection_input_hidden,
            source="Input",
            target="Hidden"
        )

        # Recurrent inhibitory connection in hidden layer
        self.connection_hidden_hidden = Connection(
            source=self.hidden,
            target=self.hidden,
            update_rule=self.learning_rule,
            wmin=-1, wmax=0
        )
        self.network.add_connection(
            connection=self.connection_hidden_hidden,
            source="Hidden",
            target="Hidden",
        )

        # Hidden layer to output
        self.connection_hidden_output = Connection(
            source=self.hidden,
            target=self.output,
            update_rule=self.learning_rule,
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
        self.network.train(mode=True)
        return self.network.run(inputs=inputs, time=self.time, reward=reward)


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
            # Only have the network select left or right
            if FLAGS.env == "BreakoutDeterministic-v4":
                return np.random.choice(self.output_dim) + 1

            return np.random.choice(self.output_dim)
        else:
            scores = self.get_Q()
            probabilities = torch.softmax(scores, dim=0)

            # Only have the network select left or right
            if FLAGS.env == "BreakoutDeterministic-v4":
                return torch.multinomial(probabilities, num_samples=1).item() + 1

            return torch.multinomial(probabilities, num_samples=1).item()

    def get_Q(self) -> Tensor:
        """Returns `Q-value` based on internal state
        Returns:
            torch.Tensor: 2-D Tensor of shape (n, output_dim)
        """
        return torch.sum(self.sqn.spike_record["Output"], dim=0)


def play_episode(env: gym.Env,
                 agent: Agent,
                 eps: float,
                 ) -> int:
    """Play an epsiode and train
    Args:
        env (gym.Env): gym environment (CartPole-v0)
        agent (Agent): agent will train and get action
        eps (float): 𝜺-greedy for exploration
    Returns:
        int: reward earned in this episode
    """
    env.reset()
    agent.sqn.network.reset_state_variables()

    done = False
    total_reward = 0

    while not done:
        # env.render()
        # Select an action
        a = agent.get_action(eps)
        # Update the state according to action a
        s, r, done, info = env.step(a)

        # Tensor shape configuration
        if FLAGS.update_rule == 'MSTDP':
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

    name = env.unwrapped.spec.id
    if name == "BreakoutDeterministic-v4":
        # Only choose left or right actions
        output_dim = 3

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


def main(save: bool = True, plot: bool = False) -> None:
    """Main
    """
    try:
        env = GymEnvironment(FLAGS.env)
        rewards = []
        input_dim, output_dim = get_env_dim(env.env)
        agent = Agent(80 * 80, [1, 1, 80, 80], output_dim, FLAGS.hidden_dim)

        for i in range(FLAGS.n_episode):
            eps = epsilon_annealing(i, FLAGS.max_episode, FLAGS.min_eps)
            r = play_episode(env, agent, eps)
            print("[Episode: {:5}] Reward: {:5} 𝜺-greedy: {:5.2f}".format(i + 1, r, eps))

            rewards.append(r)

        name = "SQN-{}-{}-{}-3-actions".format(FLAGS.update_rule.replace(" ", ""), FLAGS.env, FLAGS.n_episode)

        if plot:
            fig, ax = plt.subplots()
            ax.plot(rewards)

            ax.set(xlabel='Episode', ylabel='Reward',
                   title='SQN performance with {} on {}'.format(FLAGS.update_rule, FLAGS.env))
            plt.show()

        if save:
            save_list(rewards, "{}.fli".format(name))

    finally:
        env.close()


if __name__ == '__main__':
    main()
