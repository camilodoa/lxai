"""
 Helpful analysis and saving functions

 @author: camilodoa
"""

from datetime import date
import matplotlib.pyplot as plt
import random
import gym


def save_list(arr: [float], path: str) -> None:
    """"Saves a list of floats to file in local experiments folder
    Args:
        arr ([float]): List to be saved
        path (str): Location of saved list
    Returns:
        None
    """
    with open("./experiments/{}".format(path), 'w') as f:
        for el in arr:
            f.write("%s\n" % el)


def load_list(path: str) -> [float]:
    """" Loads from file a list of foats in local experiments folder
    Args:
        path (str): Where to load list from
    Returns:
        [float]: List of floats
    """
    with open("./experiments/{}".format(path), 'r') as f:
        arr = f.readlines()
        arr = [float(x.strip()) for x in arr]
    return arr


def compare(paths: [str], save=False) -> None:
    """"Loads a list of lists from local experiments folder of floats and plots them all together
    Saves to comparison-date
    Args:
        paths ([str]): Paths to the lists we are comparing
    """
    lists = []
    fig, ax = plt.subplots()

    for path in paths:
        list = load_list(path)
        color = "#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)])

        ax.plot(list, color=color, label=path[:-4])
        lists.append(list)

    ax.set(xlabel='100 Episode Average', ylabel='Reward',
           title='Reward over Episodes')
    ax.legend()
    ax.set_ylim([0, 30])

    if save:
        fig.savefig("comparison-{}".format(date.today()))
    plt.show()


def explore_actions(name: str = "BreakoutDeterministic-v4") -> [str]:
    """"Prints a description for each action given the name of the Gym Env
        Args:
            name (str): Name of Gym Env to explore
        Returns:
            [string]: List of short action descriptions
        """
    env = gym.make(name)

    meanings = env.unwrapped.get_action_meanings()
    print(meanings)

    return meanings


if __name__ == '__main__':
    # Baseline
    pre_06_22_2021 = "06-22-2021-update/"
    # Initial unrefined run with a variety of learning rules
    run_10000_preliminary_results = [
        pre_06_22_2021 + "DQN-Linear-BreakoutDeterministic-v4-10000.fli",
        pre_06_22_2021 + "SQN-PostPre-BreakoutDeterministic-v4-10000.fli",
        # pre_06_22_2021 + "SQN-WeightDependentPostPre-BreakoutDeterministic-v4-10000.fli",
        # pre_06_22_2021 + "SQN-Hebbian-BreakoutDeterministic-v4-10000.fli",
        pre_06_22_2021 + "SQN-MSTDPET-BreakoutDeterministic-v4-10000.fli"
    ]

    # Second run where voltage state was not reset before each episode
    run_1000_no_state_reset = [
        pre_06_22_2021 + "DQN-Linear-BreakoutDeterministic-v4-1000.fli",
        pre_06_22_2021 + "SQN-MSTDP-BreakoutDeterministic-v4-1000.fli",
        pre_06_22_2021 + "SQN-MSTDPET-BreakoutDeterministic-v4-1000.fli"
    ]

    # Third run where I re-implemented the LIF and connection parameters from Florian 2007 for MSTDP
    # With different learning rates (gamma)
    run_1000_florian = [
        pre_06_22_2021 + "DQN-Linear-BreakoutDeterministic-v4-1000.fli",
        pre_06_22_2021 + "SQN-MSTDP-BreakoutDeterministic-v4-1000.fli",
    ]

    run_sliding_window = [
        pre_06_22_2021 + "DQN-linear-slidingwindow-BreakoutDeterministic-v4-8000-0.5.fli",
        pre_06_22_2021 + "DQN-linear-slidingwindow-BreakoutDeterministic-v4-8000-0.99.fli",
        pre_06_22_2021 + "SQN-slidingwindow-MSTDP-BreakoutDeterministic-v4-8000-0.01.fli",
        pre_06_22_2021 + "SQN-slidingwindow-MSTDP-BreakoutDeterministic-v4-8000-0.5.fli",
        pre_06_22_2021 + "SQN-slidingwindow-MSTDP-BreakoutDeterministic-v4-8000-1.25.fli",
        pre_06_22_2021 + "SQN-slidingwindow-MSTDP-BreakoutDeterministic-v4-8000-10.0.fli",
        pre_06_22_2021 + "SQN-slidingwindow-MSTDP-BreakoutDeterministic-v4-8000-100.0.fli"
    ]

    # CNN vs linear
    run_100_MSTDP = [
        pre_06_22_2021 + "DQN-linear-slidingwindow-BreakoutDeterministic-v4-8000-0.99.fli",
        "DQN-cnn-BreakoutDeterministic-v4-2000-0.99-reward_clamping_1.fli",
        "DQN-cnn-BreakoutDeterministic-v4-2000-0.99-reward_clamping_2.fli",
        "DQN-cnn-BreakoutDeterministic-v4-2000-0.99-reward_clamping.fli",
        "DQN-cnn-BreakoutDeterministic-v4-10000-0.99-reward_clamping.fli",
    ]

    compare(run_100_MSTDP)
