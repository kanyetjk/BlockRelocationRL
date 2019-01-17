import numpy as np


class TreeSearch:
    def __init__(self, model, block_relocation):
        self.model = model
        self.env = block_relocation

    def find_path(self, matrix, search_depth=3):
        # TODO Handle solved

        self.env.matrix = matrix
        path = []

        while not self.env.is_solved():
            possible_next_states = self.env.all_next_states_n_moves(depth=search_depth)
            possible_next_states["StateValue"] = self.model.predict(possible_next_states.StateRepresentation.values)

            best_row = possible_next_states.StateValue.argmax()
            path += best_row.Move[0]
            self.env.move(*best_row.Move[0])

        return path

    def move_to_hot_one_encoding(self, move):
        # (0,1), (0,2), (1,0), (1,2), (2,0), (2,1)
        width = self.env.width
        size = (width - 1) * width
        vector = np.zeros(size)

        position = move[0] * (width - 1)
        position += move[1]

        if move[0] < move[1]:
            position -= 1

        vector[position] = 1

        return vector
