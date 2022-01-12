"""
Class used to sample the distribution of membrane time constants in the brain
author: @camilodoa
"""

import pandas as pd
import numpy as np
import torch

class Tau:
    def __init__(self):
        self.data = pd.read_csv('./timeconstants/article_ephys_metadata_curated.csv', sep="	")
        self.data = self.data["tau"].dropna()
        self.tau_distribution = self.data.to_numpy()

    def get_distribution(self):
        return self.tau_distribution

    def sample(self, number):
        return torch.tensor([np.random.choice(self.tau_distribution) for x in range(number)])
