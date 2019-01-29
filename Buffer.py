import numpy as np
import pandas as pd
# TODO ADD Function needs to take in a dataframe
# TODO GET SAMPLE ADD THE REMOVE OPTION


class Buffer:
    """A class to store data temporarily to use for Deep Learning.

    A Buffer holds a pandas DataFrame with the columns [StateRepresentation, StateValue, Move].
    It is limited by a size and removes data FIFO. A function can be called to return a random sample of the data.

    Attributes:
        size: An integer specifying the max amount of rows to keep.
    """

    def __init__(self, size=20000):
        """Inits class Buffer with given size."""
        self.max_size = size
        self.storage = pd.DataFrame(columns=["StateRepresentation", "StateValue", "Move"])

    def resize(self):
        """Removes rows FIFO when the pandas data frame holds more rows than the max_size allows."""
        if self.storage.shape[0] > self.max_size:
            cutoff = self.storage.shape[0] - self.max_size
            self.storage = self.storage.iloc[cutoff:]

    def add(self, s: np.array, v: int):
        self.storage = self.storage.append({"StateRepresentation": s, "StateValue": np.array(v)}, ignore_index=True)
        self.resize()

    def get_sample(self, size, remove=False):
        """Randomly fetches rows from the storage.

        Args:
            size: The amount of samples to randomly draw from the storage.
            remove: A boolean specifying if the entries should be removed from the storage.

        Returns:
            A pandas DataFrame with columns [StateRepresentation, StateValue, Move].
        """

        if size > self.storage.shape[0]:
            size = self.storage.shape[0]

        selected_rows = self.storage.sample(size)

        if remove:
            self.storage = self.storage.drop(selected_rows.index)

        return selected_rows
        """ keep because this code is needed elsewhere
        x = selected_rows["StateRepresentation"].values
        y = selected_rows["StateValue"].values

        x = np.stack(x, axis=0)
        y = np.reshape(y, (-1, 1))

        return x, y
        """

    def append(self, df):
        """Appends new entries to the storage. Removes old entries if necessary.

        Args:
            df: A pandas DataFrame with the same columns as the storage DataFrame.
        """
        self.storage = self.storage.append(df, ignore_index=True)
        self.resize()
