import numpy as np
from numpy.typing import NDArray
from typing import List


class Solution:
    def forward(self, x: NDArray[np.float64], weights: List[NDArray[np.float64]], biases: List[NDArray[np.float64]]) -> NDArray[np.float64]:
        # x: 1D input array
        # weights: list of 2D weight matrices
        # biases: list of 1D bias vectors
        # Apply ReLU after each hidden layer, no activation on output layer
        # return np.round(your_answer, 5)
        h= x
        num_hidden_layers = len(weights) -1

        c = 0
        while c < num_hidden_layers:
            z = np.dot(h,weights[c]) + biases[c]
            h = np.maximum(0,z)
            c +=1
        h = np.dot(h,weights[-1]) + biases[-1]

        return np.round(h,5)
