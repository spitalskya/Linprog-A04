from __future__ import annotations
import numpy as np
from scipy.optimize import linprog, OptimizeResult
from typing import List


class Model:
    y: np.array
    x_vect: List[np.array]
    var_count: int
    x_dim: int
    space_dim: int
    solved: OptimizeResult
    beta: np.array

    def __init__(self, dependent_vect: np.array, independent_vect: np.array) -> None:
        # dependent vector of variables
        self.y = dependent_vect
        # list of independent vectors of variables
        self.x_vect = independent_vect
        # number of elements in a vector
        self.var_count = len(self.y)
        # dimension of x-es
        self.x_dim = len(independent_vect)
        # total space dimension
        self.space_dim = self.x_dim + 1
        # all attributes of solved problem
        self.solved = None
        # beta values
        self.beta = None


class L1Model(Model):

    def __init__(self, dependent_vect: np.array, independent_vect: np.array):
        super().__init__(dependent_vect, independent_vect)

    def solve(self) -> np.array:
        # form LP
        c = np.array([0] * self.space_dim + [1] * self.var_count)
        A = np.vstack([np.array([1] * self.var_count), self.x_vect]).transpose()
        I = np.identity(self.var_count)
        A_ub = np.block([[-A, -I], [A, -I]])
        b_ub = np.concatenate([-self.y, self.y])
        # solve
        self.solved = linprog(c, A_ub, b_ub, bounds=(None, None))
        self.beta = self.solved.x[: self.space_dim]
        return self.beta


class LInfModel(Model):

    def __init__(self, dependent_vect: np.array, independent_vect: np.array):
        super().__init__(dependent_vect, independent_vect)

    def solve(self) -> np.array:
        # form LP
        c = np.array([0] * self.space_dim + [1])
        A = np.vstack([np.array([1] * self.var_count), self.x_vect]).transpose()
        ones = np.array([[1] * self.var_count]).transpose()
        A_ub = np.block([[-A, -ones], [A, -ones]])
        b_ub = np.concatenate([-self.y, self.y])
        # solve
        self.solved = linprog(c, A_ub, b_ub, bounds=(None, None))
        self.beta = self.solved.x[: self.space_dim]
        return self.beta
