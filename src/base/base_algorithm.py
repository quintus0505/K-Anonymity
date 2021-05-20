import sys

sys.path.append('../')
from abc import ABC, abstractmethod
from utils.user_config import load_data


class BaseAlgorithm(ABC):
    """Trainer base class."""

    def __init__(self, algorithm_name: str, k: int):
        super().__init__()
        self.algorithm_name = algorithm_name
        self.k = k
        self.data_set = dict()
        self.data_num = 0

    def load_dataset(self):
        self.data_set = load_data()
        self.data_num = len(self.data_set['occupation'])

    @abstractmethod
    def initial_setting(self):
        """
        Implement initial_setting method that sets reduction pair and pop redundant keys in data set
        depends on different algorithm.
        """
        pass

    @abstractmethod
    def save_data(self):
        """
        Implement save_data method that depends on different algorithm.
        """
        pass

    @abstractmethod
    def process(self):
        """
        Implement process method that process dataset to achieve K-Anonymity through different algorithm.
        """
        pass

    @abstractmethod
    def compute_loss_metric(self):
        """
        Implement compute_loss_metric method.
        """
        pass
