import numpy as np


class ApproximationModel:
    def __init__(self):
        pass

    @staticmethod
    def reshape_input(input_matrix):
        return input_matrix.transpose().flatten()


class PolicyNetwork(ApproximationModel):
    def __init__(self):
        super().__init__()
        pass


class ValueNetwork(ApproximationModel):
    def __init__(self):
        super().__init__()
        pass
