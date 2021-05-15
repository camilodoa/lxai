"""
SQN algorithm to solve CartPole-v0

SQN in BindsNet
"""
import argparse
from bindsnet.network import Network
from bindsnet.network.nodes import Input, LIFNodes
from bindsnet.network.topology import Connection
import gym

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# parser.add_argument("--gamma",
#                     type=float,
#                     default=0.99,
#                     help="Discount rate for Q_target")
parser.add_argument("--env",
                    type=str,
                    default="CartPole-v0",
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
                    default=12,
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


class SQN():
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int) -> None:
        """DQN Network
        Args:
            input_dim (int): `state` dimension.
                `state` is 2-D tensor of shape (n, input_dim)
            output_dim (int): Number of actions.
                Q_value is 2-D tensor of shape (n, output_dim)
            hidden_dim (int): Hidden dimension in fc layer
        """
        self.network = Network()

        self.layer1 = Input(n=input_dim)
        self.network.add_layer(
            layer=self.layer1, name="layer1"
        )

        self.layer2 = LIFNodes(n=hidden_dim)
        self.network.add_layer(
            layer=self.layer2, name="layer2"
        )

        self.final = LIFNodes(n=output_dim)
        self.network.add_layer(
            layer=self.final, name="final"
        )

        # First connection
        self.connection_layer1_layer2 = Connection(
            source=self.layer1,
            target=self.layer2,
            w=0.05 + 0.1 * torch.randn(self.layer1.n, self.layer2.n) # Normal (0.05, 0.01) weights
        )
        self.network.add_connection(
            connection=self.connection_layer1_layer2,
            source="layer1",
            target="layer2"
        )

        # Recurrent connection in hidden layer
        self.connection_layer2_layer2 = Connection(
            source=self.layer2,
            target=self.layer2,
            w=0.025 * (torch.eye(self.layer2.n) - 1) # Self-connecting small weights
        )
        self.network.add_connection(
            connection=self.connection_layer2_layer2
            source="layer2",
            target="layer2"
        )

        # Hidden layer connection to output
        self.connection_layer2_final = Connection(
            source=self.layer2,
            target=self.final,
            w=0.05 + 0.1 * torch.randn(self.layer2.n, self.final.n) # Normal (0.05, 0.01) weights
        )
        self.network.add_connection(
            connection=self.connection_layer2_final,
            source="layer2",
            target="final"
        )
