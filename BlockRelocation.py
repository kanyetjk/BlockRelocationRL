import numpy as np
import pandas as pd
from itertools import permutations
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

        containers = np.arange(1, (self.height-self.air) * self.width + 1)
        np.random.shuffle(containers)
        containers = containers.reshape((self.height-self.air), self.width)
        air_rows = np.zeros((self.air, self.width))
        return np.concatenate((air_rows, containers), axis=0)

    def create_instance_random(self):
        containers = list(range(1, self.num_container + 1))
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
        """
        ii = np.argmax(self.matrix[:, first_pos] > 0)
        val = self.matrix[ii, first_pos]
        self.matrix[ii, first_pos] = 0
        """

        # inserting the block
        ii = 0
        while ii <= self.height - 1 and self.matrix[ii, second_pos] == 0:
            ii += 1

        self.matrix[ii - 1, second_pos] = val
        while self.can_remove():
            self.remove_container()

    def stochastic_greedy_policy(self):
        column_with_one = int(np.where(self.matrix == 1)[1])
        possible_targets = np.setdiff1d(np.where(self.matrix[0, :] == 0), column_with_one)
        return np.random.choice(possible_targets)

    def is_solved(self):
        return not np.any(self.matrix > 0)

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

    def all_permutations(self, flatten=None):
        # TODO COULD RETURN AN ITERATOR FOR BETTER SPACE COMPLEXITY
        transposed_matrix = self.matrix.transpose()
        perm = np.array([np.array(p).transpose().flatten() for p in permutations(transposed_matrix)])
        print(perm.shape)
        return perm

    def is_legal_move(self, first_pos, second_pos):
        if self.matrix[0, second_pos] > 0:
            return False

        if not np.any(self.matrix[:, first_pos] > 0):
            return False

        return True

    def all_legal_moves(self):
        legal_moves = []
        for x in range(self.width):
            for y in range(self.width):
                if x != y:
                    if self.is_legal_move(x, y):
                        legal_moves.append((x, y))

        return legal_moves

    def all_next_states_and_moves(self, env=None):
        if env is not None:
            self.matrix = env

        saved_matrix = self.matrix.copy()
        df = pd.DataFrame(columns=["StateRepresentation", "Move"])
        for move in self.all_legal_moves():
            self.matrix = saved_matrix.copy()
            self.move(*move)
            df = df.append({"StateRepresentation": self.matrix.copy(), "Move": [move]}, ignore_index=True)
        self.matrix = saved_matrix.copy()

        return df

    def all_next_states_n_moves(self, depth):
        saved_matrix = self.matrix.copy()
        possible_states = self.all_next_states_and_moves()
        next_list = []
        for x in range(1, depth):
            for i, row in possible_states.iterrows():
                m = row.StateRepresentation
                prev_moves = row.Move
                temp = self.all_next_states_and_moves(env=m.copy())
                temp["Move"] = (prev_moves + temp.Move).copy()
                next_list.append(temp)

            #  removing duplicate states
            possible_states = pd.concat(next_list, ignore_index=True)
            possible_states["dup"] = possible_states.StateRepresentation.astype("str")
            possible_states = possible_states.drop_duplicates(subset="dup")
            possible_states = possible_states.drop(columns=["dup"])

        self.matrix = saved_matrix.copy()

        return possible_states


if __name__ == "__main__":
    #test = BlockRelocation(5, 5)
    #print(test.matrix)
    #a = test.solve_greedy()
    #print(a)
    pass


# TODO HOW DO I ACTUALLY SAVE THE DATA FOR THE NEURAL NET, CURRENTLY ROW BY ROW AS OPPOSED TO COL BY COL


