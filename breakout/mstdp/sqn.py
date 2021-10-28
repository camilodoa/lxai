"""
SQN algorithm to solve CartPole-v0

Heterogeneous SQN implementation

author: @camilodoa
"""
import argparse
import torch
from bindsnet.network import Network
from bindsnet.network.nodes import Input, LIFNodes
from bindsnet.network.topology import Connection
from bindsnet.network.monitors import Monitor
from bindsnet.network.nodes import AbstractInput
from bindsnet.learning import PostPre, WeightDependentPostPre, Hebbian, MSTDP, MSTDPET

from timeconstants.Tau import Tau
from learning import HMSTDP
import matplotlib.pyplot as plt
import gym
from bindsnet.environment import GymEnvironment
from typing import Tuple
from collections import deque
from statistics import mean
from torch import Tensor
from analysis import save_list
from bindsnet.analysis.plotting import plot_spikes

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
                    default=1000,
                    help="Hidden dimension")
parser.add_argument("--update-rule",
                    type=str,
                    default="HMSTDP",
                    help="Learning rule used to update weights")
parser.add_argument("--gamma",
                    type=float,
                    default=0.01,
                    help="Neuron learning rate")
FLAGS = parser.parse_args()

# Update rules
rules = {
    "PostPre": PostPre,
    "WeightDependentPostPre": WeightDependentPostPre,
    "Hebbian": Hebbian,
    "MSTDP": MSTDP,
    "MSTDPET": MSTDPET,
    "HMSTDP": HMSTDP
}

s_im, s_ax = None, None

class SQN(object):
    def __init__(self, input_dim: int, shape: [int], output_dim: int, hidden_dim: int) -> None:
        """SQN Network
        """
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.network = Network(dt=1.0)
        self.learning_rule = rules.get(FLAGS.update_rule)
        self.time = int(self.network.dt)

        tau = Tau()
        # To solve tensor formatting issues
        if FLAGS.update_rule == 'MSTDP' or FLAGS.update_rule == "HMSTDP":
            self.input = Input(n=input_dim, traces=True)
        else:
            self.input = Input(n=input_dim, shape=shape, traces=True)

        # Heterogeneous LIFNodes
        self.hidden = LIFNodes(n=hidden_dim, traces=True, tc_decay=tau.sample(hidden_dim))
        self.output = LIFNodes(n=output_dim, traces=True, tc_decay=tau.sample(output_dim))

        # First connection
        self.connection_input_hidden = Connection(
            source=self.input,
            target=self.hidden,
            update_rule=self.learning_rule,
            wmin=0,
            wmax=1,
            nu=FLAGS.gamma
        )

        # Hidden recurrent connection
        self.connection_hidden_hidden = Connection(
            source=self.hidden,
            target=self.hidden,
            update_rule=self.learning_rule,
            wmin=0,
            wmax=1,
            nu=FLAGS.gamma
        )

        # Hidden layer to Output
        self.connection_hidden_output = Connection(
            source=self.hidden,
            target=self.output,
            update_rule=self.learning_rule,
            wmin=0,
            wmax=1,
            nu=FLAGS.gamma
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
        self.network.add_connection(
            connection=self.connection_hidden_hidden,
            source="Hidden",
            target="Hidden"
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


class Agent(object):
    def __init__(self, input_dim: int, shape: [int], output_dim: int, hidden_dim: int) -> None:
        """Agent class that chooses an action and trains
        """
        self.sqn = SQN(input_dim, shape, output_dim, hidden_dim)
        self.input_dim = input_dim
        self.output_dim = output_dim

    def get_action(self) -> int:
        """Returns an action index
        """
        scores = self.get_Q()
        probabilities = torch.softmax(scores, dim=0)
        return torch.multinomial(probabilities, num_samples=1).item()

        _, argmax = torch.max(torch.flatten(scores), dim=0)
        # print(torch.flatten(scores), argmax.item())

        # if np.random.rand() < 0.2:
        #     return np.random.choice(self.output_dim)
        # else:
        return argmax.item()

    def get_Q(self) -> Tensor:
        """Returns `Q-value` based on output layer's spikes
        """
        return torch.sum(self.sqn.spike_record["Output"], dim=0)


def play_episode(env: GymEnvironment,
                 agent: Agent) -> int:
    """Play an epsiode and train
    Args:
        env (gym.Env): gym environment (CartPole-v0)
        agent (Agent): agent will train and get action
    Returns:
        int: reward
    """
    env.reset()
    agent.sqn.network.reset_state_variables()

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
        env = GymEnvironment(FLAGS.env)

        average_rewards = []
        q = deque(maxlen=100)

        input_dim, output_dim = get_env_dim(env.env)
        agent = Agent(80 * 80, [1, 1, 80, 80], output_dim, FLAGS.hidden_dim)

        for i in range(FLAGS.n_episode):
            r = play_episode(env, agent)
            print("[Episode: {:5}] Reward: {:5}".format(i + 1, r))

            q.append(r)
            if i % 100 == 0:
                average_rewards.append(mean(q))

        name = "SQN-{}-{}-{}-{}-reward_clamping".format(FLAGS.update_rule.replace(" ", ""), FLAGS.env, FLAGS.n_episode, FLAGS.gamma)

        if plot:
            fig, ax = plt.subplots()
            ax.plot(average_rewards)

            ax.set(xlabel='Episode', ylabel='Reward',
                   title='SQN performance with {} on {}'.format(FLAGS.update_rule, FLAGS.env))
            plt.show()

        if save:
            save_list(average_rewards, "{}.fli".format(name))

    finally:
        env.close()


if __name__ == '__main__':
    main(save=True, plot=True)
