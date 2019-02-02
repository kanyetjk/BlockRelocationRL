import numpy as np
import pandas as pd


class TreeSearch:
    def __init__(self, model, block_relocation):
        self.model = model
        self.env = block_relocation

    def find_path(self, matrix, search_depth=3):
        # TODO Handle solved

        self.env.matrix = matrix
        while self.env.can_remove_matrix(matrix):
            matrix = self.env.remove_container_from_matrix(matrix)
        path = []

        counter = 0
        while not self.env.is_solved(matrix=matrix):
            counter += 1
            if counter > 25:
                return

            # predicting values of the states
            possible_next_states = self.env.all_next_states_n_moves(depth=search_depth, matrix=matrix)
            values = list(self.model.predict_df(possible_next_states))
            values = [x[0] for x in values]

            possible_next_states["StateValue"] = values

            best_row_index = possible_next_states.StateValue.idxmax()
            best_row = possible_next_states.loc[best_row_index, :]
            path.append(best_row.Move[0])
            print(matrix)
            print(best_row.Move[0])
            matrix = self.env.move_on_matrix(matrix, *best_row.Move[0])

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

            df = df.append({"StateRepresentation": rep, "Move": move, "Value": -steps_left}, ignore_index=True)
            self.env.move(*move)
            steps_left -= 1

        return df

    def generate_basic_starting_data(self, num_examples):
        list_of_dfs = []
        for _ in range(num_examples):
            self.env.matrix = self.env.create_instance(4, 4)
            paths = self.env.solve_greedy()
            df = self.move_along_path(self.env.matrix, paths)
            list_of_dfs.append(df)

        final_df = pd.concat(list_of_dfs, axis=0, ignore_index=True)
        return final_df


if __name__ == "__main__":
    from BlockRelocation import BlockRelocation

    test = TreeSearch(4, BlockRelocation(6, 6))
    a = test.generate_basic_starting_data(250)
    print(a.shape)
