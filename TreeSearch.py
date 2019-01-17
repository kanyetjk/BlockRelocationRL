import numpy as np
import pandas as pd


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

    def reverse_hot_one_encoding(self, vector):
        width = self.env.width - 1
        position_in_vector = np.argmax(vector)
        first_pos = position_in_vector // width
        second_pos = position_in_vector % width

        if second_pos >= first_pos:
            second_pos += 1

        return first_pos, second_pos

    def move_along_path(self, matrix, path):
        self.env.matrix = matrix
        steps_left = len(path)
        df = pd.DataFrame(columns=["StateRepresentation", "Move", "Value"])

        for move in path:
            rep = self.env.matrix.copy()
            move_encoded = self.move_to_hot_one_encoding(move)

            df = df.append({"StateRepresentation": rep, "Move": move_encoded, "value": -steps_left}, ignore_index=True)
            self.env.move(*move)
            steps_left -= 1

        return df


from BlockRelocation import BlockRelocation

test = TreeSearch(4, BlockRelocation(4,4))

for x in range(4):
    for y in range(4):
        if x != y:
            a = test.move_to_hot_one_encoding((x,y))
            b = test.reverse_hot_one_encoding(a)
            if (x,y) != b:
                print(x,y)
