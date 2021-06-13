from datetime import date
import matplotlib.pyplot as plt
import random


def save_list(arr: [float], path: str) -> None:
    """"Saves a list of floats to file
    Args:
        arr ([float]): List to be saved
        path (str): Location of saved list
    Returns:
        None
    """
    with open(path, 'w') as f:
        for el in arr:
            f.write("%s\n" % el)


def load_list(path: str) -> [float]:
    """" Loads from file a list of foats
    Args:
        path (str): Where to load list from
    Returns:
        [float]: List of floats
    """
    with open(path, 'r') as f:
        arr = f.readlines()
        arr = [float(x.strip()) for x in arr]
    return arr


def compare(paths: [str]) -> None:
    """"Loads a list of lists of floats and plots them all together
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

    ax.set(xlabel='Episode', ylabel='Reward',
           title='Reward over Episodes')
    ax.legend()
    ax.set_ylim([0, 150])

    fig.savefig("comparison-{}".format(date.today()))
    plt.show()


if __name__ == '__main__':
    arr = [
        "SQN-PostPre-BreakoutDeterministic-v4-10000.fli",
        "DQN-Linear-BreakoutDeterministic-v4-10000.fli"
    ]
    compare(arr)
