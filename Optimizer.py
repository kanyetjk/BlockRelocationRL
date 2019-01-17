from Buffer import Buffer
from BlockRelocation import BlockRelocation
from TreeSearch import TreeSearch
from ApproximationModel import ApproximationModel


class Optimizer:
    def __init__(self, width=4, height=4):
        self.width = width
        self.height = height

        self.buffer = Buffer(size=200000)
        self.env = BlockRelocation(self.height, self.width)
        self.model = ApproximationModel()
        self.tree_searcher = TreeSearch()
