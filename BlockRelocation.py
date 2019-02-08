import numpy as np
import pandas as pd
from itertools import permutations
import timeit


# TODO AIR STUFF
# DFS no space problem
# Beam not so good, because only using the value network


class BlockRelocation:
    __slots__ = ["height", "width", "num_container", "matrix", "air"]

    def __init__(self, height, width, num_container=0, air=2):
        self.height = height + air
        self.air = air
        self.width = width
        self.num_container = num_container
        self.matrix = self.create_instance(self.height, self.width)

    def create_instance(self, height, width):
        self.width = width

        containers = np.arange(1, (self.height - self.air) * self.width + 1)
        np.random.shuffle(containers)
        containers = containers.reshape((self.height - self.air), self.width)
        air_rows = np.zeros((self.air, self.width))
        return np.concatenate((air_rows, containers), axis=0)

    def create_instance_random(self, num_containers):
        containers = list(range(1, num_containers + 1))
        containers += [0] * ((self.height * self.width) - len(containers))

        np.random.shuffle(containers)
        representation = np.array(containers).reshape((self.height, self.width))

        def fix_col(column):
            top = 0
            for i in range(len(column)):
                if column[i] == 0:
                    column[i], column[top] = column[top], column[i]
                    top += 1
            return column

        for c in range(self.width):
            representation[:, c] = fix_col(representation[:, c])

        return representation

    def remove_container(self):
        self.matrix = self.matrix - 1
        self.matrix[self.matrix < 0] = 0

    def can_remove(self):
        for c in range(self.width):
            for val in self.matrix[:, c]:
                if val == 0:
                    continue
                if val == 1:
                    return True
                if val > 1:
                    break
        return False

    # noinspection PyUnboundLocalVariable
    def move(self, first_pos, second_pos):
        # Checking for invalid moves
        if self.matrix[0, second_pos] > 0:
            return

        if not np.any(self.matrix[:, first_pos] > 0):
            return

        # finding the block too move

        for ii in range(self.height):
            if self.matrix[ii, first_pos] > 0:
                val = self.matrix[ii, first_pos]
                self.matrix[ii, first_pos] = 0
                break

        # inserting the block
        ii = 0
        while ii <= self.height - 1 and self.matrix[ii, second_pos] == 0:
            ii += 1

        self.matrix[ii - 1, second_pos] = val
        while self.can_remove():
            self.remove_container()

    def move_on_matrix(self, matrix, first_pos, second_pos):
        # TODO NOT GOOD
        # finding the block too move
        for ii in range(self.height):
            if matrix[ii, first_pos] > 0:
                val = matrix[ii, first_pos]
                matrix[ii, first_pos] = 0
                break

        # inserting the block
        ii = 0
        while ii <= self.height - 1 and matrix[ii, second_pos] == 0:
            ii += 1

        matrix[ii - 1, second_pos] = val
        while self.can_remove_matrix(matrix):
            matrix = self.remove_container_from_matrix(matrix)

        return matrix

    def remove_container_from_matrix(self, matrix):
        matrix = matrix - 1
        matrix[matrix < 0] = 0
        return matrix

    def can_remove_matrix(self, matrix):
        if np.max(matrix) == 0:
            return False
        c = int(np.where(matrix == 1)[1])

        for val in matrix[:, c]:
            if val == 1:
                return True
            if val > 1:
                return False
        return False

    def stochastic_greedy_policy(self):
        column_with_one = int(np.where(self.matrix == 1)[1])
        possible_targets = np.setdiff1d(np.where(self.matrix[0, :] == 0), column_with_one)
        return np.random.choice(possible_targets)

    def is_solved(self, matrix=None):
        if matrix is None:
            matrix = self.matrix
        return not np.any(matrix > 0)

    def solve_greedy(self):
        matrix_copy = self.matrix.copy()
        counter = 0
        moves = []
        while not self.is_solved():
            if self.can_remove():
                self.remove_container()
                continue

            with_one = int(np.where(self.matrix == 1)[1])
            goal = self.stochastic_greedy_policy()
            moves.append((with_one, goal))

            self.move(with_one, goal)
            counter += 1
        self.matrix = matrix_copy.copy()
        return moves

    def all_permutations_state(self, matrix=None, flatten=None):
        # TODO COULD RETURN AN ITERATOR FOR BETTER SPACE COMPLEXITY
        if matrix is None:
            matrix = self.matrix

        transposed_matrix = matrix.transpose()
        perm = np.array([np.array(p).flatten() for p in permutations(transposed_matrix)])
        return perm

    def all_permutations_move(self, pos1, pos2):
        x = list(range(self.width))
        all_permutations = []
        b = permutations(x)
        for perm in b:
            new_pos1 = perm.index(pos1)
            new_pos2 = perm.index(pos2)
            all_permutations.append((new_pos1, new_pos2))

        return all_permutations

    def is_legal_move(self, first_pos, second_pos, matrix=None):
        if matrix is None:
            matrix = self.matrix

        if matrix[0, second_pos] > 0:
            return False

        if not np.any(matrix[:, first_pos] > 0):
            return False

        return True

    def all_legal_moves(self, matrix):
        if matrix is None:
            matrix = self.matrix

        legal_moves = []
        for x in range(self.width):
            for y in range(self.width):
                if x != y:
                    if self.is_legal_move(x, y, matrix):
                        legal_moves.append((x, y))

        return legal_moves

    def all_next_states_and_moves(self, matrix=None, moves=None):
        if matrix is None:
            matrix = self.matrix

        if moves is None:
            moves = self.all_legal_moves(matrix.copy())
        else:
            temp = []
            for move in moves:
                if self.is_legal_move(move[0], move[1], matrix):
                    temp.append(move)
            moves = temp

        df_list = []
        for move in moves:
            temp = self.move_on_matrix(matrix.copy(), *move)
            if self.is_solved(temp):
                df = pd.DataFrame(columns=["StateRepresentation", "Move", "Solved"])
                df = df.append({"StateRepresentation": matrix.copy(), "Move": [move], "Solved":True}, ignore_index=True)
                return df

            df_list.append({"StateRepresentation": temp.copy(), "Move": [move]})
        df = pd.DataFrame(df_list)
        return df

    def all_next_states_n_moves(self, depth, matrix=None):
        if matrix is None:
            matrix = self.matrix
        seen_states = set()
        possible_states = self.all_next_states_and_moves(matrix)
        for x in range(1, depth):
            next_list = []
            for i, row in possible_states.iterrows():
                m = row.StateRepresentation
                prev_moves = row.Move
                temp = self.all_next_states_and_moves(m.copy())
                temp["Move"] = (prev_moves + temp.Move).copy()
                if temp.shape[0] == 1:
                    #print("solved_fully")
                    return temp
                next_list.append(temp)

            #  removing duplicate states
            possible_states = pd.concat(next_list, ignore_index=True)
            possible_states["hashed"] = possible_states["StateRepresentation"].apply(lambda x: x.tostring())
            possible_states = possible_states[~possible_states['hashed'].isin(seen_states)]
            possible_states = possible_states.drop_duplicates(subset="hashed")

            seen_states = set(possible_states.hashed.values)
            possible_states = possible_states.drop(columns=["hashed"])

        return possible_states


if __name__ == "__main__":
    a = BlockRelocation(4,4)
    c = a.all_permutations_state(a.matrix)
    b = a.all_permutations_move(0,1)
    for xx, move in zip(c, b):
        print(xx.reshape(6,4))
        print(move)



# TODO HOW DO I ACTUALLY SAVE THE DATA FOR THE NEURAL NET, CURRENTLY ROW BY ROW AS OPPOSED TO COL BY COL
