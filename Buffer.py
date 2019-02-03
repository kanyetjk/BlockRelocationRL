import pandas as pd


class Buffer:
    """A class to store data temporarily to use for Deep Learning.

    A Buffer holds a pandas DataFrame with the columns [StateRepresentation, StateValue, Move, MovesEncoded].
    It is limited by a size and removes data FIFO. A function can be called to return a random sample of the data.

    The idea of using a buffer is, that each simulation creates data which is heavily correlated, especially
    when also creating permutations. So after new data is created we only train on a subset and save the rest in the
    buffer, where we can access it at a later training state. Some data may never be used for training.

    Attributes:
        size: An integer specifying the max amount of rows to keep.
    """

    def __init__(self, size=20000):
        """Inits class Buffer with given size."""
        self.max_size = size
        self.storage = pd.DataFrame(columns=["StateRepresentation", "StateValue", "Move", "MovesEncoded"])

    def resize(self):
        """Removes rows FIFO when the pandas data frame holds more rows than the max_size allows."""
        if self.storage.shape[0] > self.max_size:
            cutoff = self.storage.shape[0] - self.max_size
            self.storage = self.storage.iloc[cutoff:]

    def increase_max_size(self, percent=0.1):
        """Increases the max size of the buffer by a given percentage."""
        self.max_size = self.max_size * (1 + percent)

    def get_sample(self, size, remove=True):
        """Randomly fetches rows from the storage.

        Args:
            size: The amount of samples to randomly draw from the storage.
            remove: A boolean specifying if the entries should be removed from the storage.

        Returns:
            A pandas DataFrame with columns [StateRepresentation, StateValue, Move, MovesEncoded].
        """

        if size > self.storage.shape[0]:
            size = self.storage.shape[0]

        selected_rows = self.storage.sample(size)

        if remove:
            self.storage = self.storage.drop(selected_rows.index)

        return selected_rows

    def append(self, df):
        """Appends new entries to the storage. Removes old entries if necessary.

        Args:
            df: A pandas DataFrame with the same columns as the storage DataFrame.
        """
        self.storage = self.storage.append(df, ignore_index=True, sort=False)
        self.resize()
