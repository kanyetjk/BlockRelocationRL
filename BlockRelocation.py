import numpy as np
import pandas as pd
from itertools import permutations


class BlockRelocation:
    __slots__ = ["height", "width", "num_container", "matrix"]

    def __init__(self, height, width, num_container=0):
        self.height = height
        self.width = width
        self.num_container = num_container
        self.matrix = self.create_instance_h_2()

    def create_instance_h_2(self):
        containers = np.arange(1, self.height * self.width + 1)
        np.random.shuffle(containers)
        containers = containers.reshape(self.height, self.width)
        air = np.zeros((2, self.width))
        return np.concatenate((air, containers), axis=0)

    def create_instance(self):
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

    def stochastic_greedy_policy(self):
        column_with_one = int(np.where(self.matrix == 1)[1])
        possible_targets = np.setdiff1d(np.where(self.matrix[0, :] == 0), column_with_one)
        return np.random.choice(possible_targets)

    def is_solved(self):
        return not np.any(self.matrix > 0)

    def solve_greedy(self):
        counter = 0
        while not self.is_solved():
            if self.can_remove():
                self.remove_container()
                continue

            with_one = int(np.where(self.matrix == 1)[1])
            goal = self.stochastic_greedy_policy()

            self.move(with_one, goal)
            counter += 1
        return counter

    def stats(self, num):
        s = []
        for xx in range(num):
            self.matrix = self.create_instance()
            s.append(self.solve_greedy())
        print(np.mean(s))

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

    def all_next_states_and_moves(self):
        saved_matrix = self.matrix.copy()
        df = pd.DataFrame(columns=["StateRepresentation", "Move"])
        for move in self.all_legal_moves():
            self.matrix = saved_matrix.copy()
            self.move(*move)
            df = df.append({"StateRepresentation": self.matrix.copy(), "Move": move}, ignore_index=True)

        return df


test = BlockRelocation(4, 4)
test.all_next_states_and_moves()
# TODO HOW DO I ACTUALLY SAVE THE DATA FOR THE NEURAL NET, CURRENTLY ROW BY ROW AS OPPOSED TO COL BY COL


