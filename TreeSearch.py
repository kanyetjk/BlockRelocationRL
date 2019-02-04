import numpy as np
import pandas as pd
from functools import lru_cache


class TreeSearch:
    def __init__(self, model, block_relocation, policy_network):
        self.model = model
        self.policy_network = policy_network
        self.env = block_relocation

    def find_path_2(self, matrix, search_depth=4, moves_per_turn=2, epsilon=0.05, threshold=0.05):
        # search depth maybe visited states
        # set for seen states
        # predict possible moves, add random epsilon value, pick values that get over the threshold
        # make all moves, repeat with search_depth times
        # evaluate all states -> move some steps to the best one, restart
        # a better policy network should allow for a deeper evaluation
        # have on big DataFrame and kick out all other moves after we chose an option
        # the value function may not be that accurate or stable, maybe grouping by the first move and averaging
        #  -> over all the results may be good, or not (maybe good to remove the weakest states


        self.env.matrix = matrix
        while self.env.can_remove_matrix(matrix):
            matrix = self.env.remove_container_from_matrix(matrix)

        # initializing the set and DataFrame
        seen_states = set()
        data = pd.DataFrame(columns=["StateRepresentation", "Move", "CurrentValue"])
        data = data.append({"StateRepresentation": matrix.copy(), "Move": [], "CurrentValue": 0}, ignore_index=True)

        #while not self.env.is_solved(matrix=matrix):
        for x in range(9):
            new_data = []
            for _, row in data.iterrows():
                state = row.StateRepresentation
                moves = row.Move

                # getting all the next states
                next_states = self.env.all_next_states_and_moves(matrix=state)
                next_states["CurrentValue"] = np.zeros(next_states.shape[0])
                next_states["Move"] = moves + next_states["Move"]
                new_data.append(next_states)

            # removing all the duplicates
            data = pd.concat(new_data)
            data["hashed"] = data["StateRepresentation"].apply(lambda s: s.tostring())
            data = data[~data['hashed'].isin(seen_states)]
            data = data.drop_duplicates(subset="hashed")

            seen_states = set(data.hashed.values)
            data = data.drop(columns=["hashed"])

            if x % 3 == 2:
                data["CurrentMove"] = data.Move.apply(lambda arr: arr[x])
                data = data[data.CurrentMove == (2, 3)]
                data = data.drop(columns="CurrentMove")

        print(data)
            #return

        pass

    def find_path(self, matrix, search_depth=4, moves_per_turn=2):
        # TODO Handle solved

        self.env.matrix = matrix
        while self.env.can_remove_matrix(matrix):
            matrix = self.env.remove_container_from_matrix(matrix)
        path = []

        counter = 0
        while not self.env.is_solved(matrix=matrix):
            if counter > 20:
                return

            # predicting values of the states
            possible_next_states = self.env.all_next_states_n_moves(depth=search_depth, matrix=matrix)
            values = list(self.model.predict_df(possible_next_states))
            values = [x[0] for x in values]

            possible_next_states["StateValue"] = values

            best_row_index = possible_next_states.StateValue.idxmax()
            best_row = possible_next_states.loc[best_row_index, :]
            for x in range(min(moves_per_turn, len(best_row.Move))):
                counter += 1
                if not self.env.is_solved(matrix=matrix):
                    path.append(best_row.Move[x])
                    print(matrix)
                    #print(best_row.Move[x])
                    matrix = self.env.move_on_matrix(matrix, *best_row.Move[x])

        return path

    @lru_cache(maxsize=900)
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

    @lru_cache(maxsize=900)
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

        df_list = []
        for move in path:
            rep = self.env.matrix.copy()
            df_list.append({"StateRepresentation": rep, "Move": move, "Value": -steps_left})
            self.env.move(*move)
            steps_left -= 1

        df = pd.DataFrame(df_list)
        return df

    def generate_basic_starting_data(self, num_examples):
        # TODO Currently hardcoded
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

    test = TreeSearch(4, BlockRelocation(4, 4), 2)
    test.find_path_2(test.env.matrix)
