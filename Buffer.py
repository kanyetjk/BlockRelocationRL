import numpy as np
import pandas as pd


class Buffer:
    __slots__ = ["max_size", "storage"]

    def __init__(self, size=20000):
        self.max_size = size
        self.storage = pd.DataFrame(columns=["StateRepresentation", "StateValue"])

    def resize(self):
        if self.storage.shape[0] > self.max_size:
            cutoff = self.storage.shape[0] - self.max_size
            self.storage = self.storage.iloc[cutoff:]

    def add(self, s, v):
        self.storage = self.storage.append({"StateRepresentation": s, "StateValue": np.array(v)}, ignore_index=True)
        self.resize()

    def get_sample(self, size):
        if size > self.storage.shape[0]:
            size = self.storage.shape[0]

        selected_rows = self.storage.sample(size)
        x = selected_rows["StateRepresentation"].values
        y = selected_rows["StateValue"].values

        x = np.stack(x, axis=0)
        y = np.reshape(y, (-1, 1))

        return x, y
