import numpy as np
import pandas as pd
from functools import lru_cache
from Utils import load_obj
import time
from collections import deque


class TreeSearch:
    def __init__(self, model, block_relocation, policy_network):
        self.model = model
        self.policy_network = policy_network
        self.env = block_relocation
        self.index_to_moves_dict = {}
        self.best_path = []
        height = block_relocation.height
        width = block_relocation.width
        std = "Deviations/" + str(height-2) + "x" + str(width) + "_std"
        self.std_vals = load_obj(std)
        self.total = 0
        self.possible_moves = width * (width-1)

        # DFS Helpes
        self.seen_states = {}
        self.start = time.time()
        self.k = 20

    def iterative_dfs(self, matrix, stop_param=1, cutoff_param=0.1, k=10, time_limit=10, max_steps=30):
        self.best_path = list(range(100))
        self.seen_states = {}
        self.start_time = time.time()
        self.next_nodes = deque([(matrix, [])])

        def dfs(matrix, current_path):
            if len(current_path) > max_steps:
                #print("Path Too Long")
                return

            predicted_moves = self.model.predict_single(matrix)[0]
            dict_access = min(round(predicted_moves), -1)
            if dict_access not in self.std_vals:
                dict_access = min(self.std_vals.keys())

            combined_val = -1 * predicted_moves - self.std_vals[dict_access] * stop_param
            if len(current_path) + combined_val >= len(self.best_path) - 1:
                return

            moves_pred = self.policy_network.predict_single(matrix)
            placeholder = list(range(len(moves_pred)))
            sorted_moves = [x for _, x in sorted(zip(moves_pred, placeholder), reverse=True)]
            sorted_output = sorted(moves_pred, reverse=True)

            sorted_moves = sorted_moves[:k]
            sorted_output = sorted_output[:k]
            while sorted_output:
                current_move = self.index_to_moves(sorted_moves.pop())
                if sorted_output.pop() < cutoff_param:
                    continue

                if not self.env.is_legal_move(current_move[0], current_move[1], matrix):
                    continue

                new_path = current_path.copy()
                new_path.append(current_move)
                new_state = self.env.move_on_matrix(matrix.copy(), *current_move)

                state_hash = new_state.tostring()
                if state_hash in self.seen_states:
                    if len(new_path) >= self.seen_states[state_hash]:
                        continue
                self.seen_states[state_hash] = len(new_path)

                if self.env.is_solved(new_state):
                    if len(current_path) < len(self.best_path):
                        self.best_path = new_path

                self.next_nodes.append((new_state, new_path))

        while self.next_nodes and time.time() - self.start_time < time_limit:
            m, path = self.next_nodes.pop()
            dfs(matrix=m, current_path=path)

        return self.best_path

    def find_path_dfs(self, matrix, stop_param=1, k=6, time_limit=10, max_steps=40,
                      cutoff_param=0.001, cutoff_increase=1, dynamic_k=False):
        self.best_path = list(range(100))
        self.seen_states = {}
        self.start = time.time()
        self.k = k
        while self.env.can_remove_matrix(matrix):
            matrix = self.env.remove_container_from_matrix(matrix)

        def dfs(matrix, current_path, cutoff=cutoff_param):
            if dynamic_k:
                new_k = matrix.max()
                self.k = min(new_k, self.possible_moves)

            self.end = time.time()

            if self.end - self.start > time_limit:
                return
            if len(current_path) > max_steps:
                return

            predicted_moves = self.model.predict_single(matrix)[0]
            dict_access = min(round(predicted_moves), -1)
            if dict_access not in self.std_vals:
                dict_access = min(self.std_vals.keys())

            combined_val = -1 * predicted_moves - self.std_vals[dict_access] * stop_param
            if len(current_path) + combined_val >= len(self.best_path) - 1:
                return

            moves_pred = self.policy_network.predict_single(matrix)
            placeholder = list(range(len(moves_pred)))
            sorted_moves = [x for _, x in sorted(zip(moves_pred, placeholder), reverse=True)]
            sorted_output = sorted(moves_pred, reverse=True)
            for i in range(self.k):
                if sorted_output.pop(0) < cutoff_param or time.time() - self.start > time_limit:
                    break
                current_move = self.index_to_moves(sorted_moves.pop(0))
                if not self.env.is_legal_move(current_move[0], current_move[1], matrix):
                    continue

                new_path = current_path.copy()
                new_path.append(current_move)
                new_state = self.env.move_on_matrix(matrix.copy(), *current_move)

                state_hash = new_state.tostring()
                if state_hash in self.seen_states:
                    if len(new_path) >= self.seen_states[state_hash]:
                        continue
                self.seen_states[state_hash] = len(new_path)

                if self.env.is_solved(new_state):
                    if len(current_path) < len(self.best_path):
                        self.best_path = new_path
                new_cutoff = min(cutoff*cutoff_increase, 0.4)
                dfs(new_state, new_path, new_cutoff)

        dfs(matrix, [])
        if len(self.best_path) == 100:
            return []

        return self.best_path

    def find_path_2(self, matrix, search_depth=4, epsilon=0.03, threshold=0.02, drop_percent=0.6, factor=0.1):
        # BFS
        self.env.matrix = matrix
        while self.env.can_remove_matrix(matrix):
            matrix = self.env.remove_container_from_matrix(matrix)

        # print(matrix)
        # initializing the set and DataFrame
        seen_states = set()
        data = pd.DataFrame(columns=["StateRepresentation", "Move", "CurrentValue"])
        data = data.append({"StateRepresentation": matrix.copy(), "Move": [], "CurrentValue": 0}, ignore_index=True)

        counter = -1 * search_depth  # search depth
        while not self.env.is_solved(matrix=matrix):

            # stopping if no new states possible
            if data.shape[0] == 0:
                print("No paths found")
                return []

            if counter > 40:
                print("Too many steps")
                return range(40)

            counter += 1
            policy = self.policy_network.predict_df(data)
            policy = list(policy)
            data["policy"] = policy
            new_data = []
            for _, row in data.iterrows():
                # preparing all relevant data
                state = row.StateRepresentation
                moves = row.Move
                policy = row.policy
                threshold *= (factor + 1)
                policy = self.policy_vector_to_moves(policy, threshold, epsilon)

                # getting all the next states
                next_states = self.env.all_next_states_and_moves(matrix=state, moves=policy)

                # exit if solved
                if "Solved" in next_states.columns:
                    next_states["Move"] = next_states["Move"].apply(lambda x: moves + x)
                    return next_states.Move[0]
                # TODO is this used?
                next_states["CurrentValue"] = np.zeros(next_states.shape[0])

                if next_states.shape[0] == 0:
                    continue

                # I do not understand this bug
                if next_states.shape[0] == 1:
                    next_states["Move"] = next_states["Move"].apply(lambda x: moves + x)
                else:
                    next_states["Move"] = moves + next_states["Move"]

                new_data.append(next_states)

            # removing all the duplicates
            if len(new_data) is 0:
                print("No Path found, Duplicates")
                return []
            data = pd.concat(new_data, sort=False)
            data["hashed"] = data["StateRepresentation"].apply(lambda s: s.tostring())
            data = data[~data['hashed'].isin(seen_states)]
            data = data.drop_duplicates(subset="hashed")

            seen_states = set(data.hashed.values)
            data = data.drop(columns=["hashed"])
            data = data.reset_index(drop=True)

            if counter >= 0:
                values = list(self.model.predict_df(data))
                values = [x[0] for x in values]
                data["StateValue"] = values

                num_rows = int(data.shape[0] * (1-drop_percent))
                num_rows = max(1000, num_rows)
                data = data.nlargest(num_rows, "StateValue")
        #print(matrix)

        return []

    def policy_vector_to_moves(self, vector, threshold, random_exploration_factor):
        # print(vector)
        noise = np.random.rand(*vector.shape) * random_exploration_factor
        vector += noise
        branches = np.where(vector >= threshold, 1, 0)
        branches = np.nonzero(branches)[0]

        moves = []
        for index in branches:
            moves.append(self.index_to_moves(index))

        if len(moves) is 0:
            moves.append(self.index_to_moves(np.argmax(vector)))

        return moves

    def index_to_moves(self, position_in_vector):
        if position_in_vector in self.index_to_moves_dict:
            return self.index_to_moves_dict[position_in_vector]

        width = self.env.width - 1
        first_pos = position_in_vector // width
        second_pos = position_in_vector % width

        if second_pos >= first_pos:
            second_pos += 1

        self.index_to_moves_dict[position_in_vector] = (first_pos, second_pos)
        return first_pos, second_pos

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
        while self.env.can_remove_matrix(matrix):
            matrix = self.env.remove_container_from_matrix(matrix)
        steps_left = len(path)

        df_list = []
        for move in path:
            rep = matrix.copy()
            df_list.append({"StateRepresentation": rep, "Move": move, "Value": -steps_left})
            matrix = self.env.move_on_matrix(matrix, *move)
            steps_left -= 1

        df = pd.DataFrame(df_list)
        return df


if __name__ == "__main__":
    from BlockRelocation import BlockRelocation
    from ApproximationModel import PolicyNetwork, ValueNetwork
    from Utils import load_configs

    configs = load_configs("Configs.json")
    val = ValueNetwork(configs)
    pol = PolicyNetwork(configs)
    test = TreeSearch(val, BlockRelocation(5, 5), pol)
    matrix = test.env.create_instance_random(10)
    print(matrix)
    print(test.find_path_2(matrix, search_depth=3))
