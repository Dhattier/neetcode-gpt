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
        def layer(x: NDArray[np.float64], weights: NDArray[np.float64], bias: NDArray[np.float64]) -> np.float64:
            return x @ weights + bias

        def ReLU (h: np.float64) -> np.float64:
            return np.maximum(0, h)

        h = x
        for i in range(len(weights)-1):
            h = ReLU(layer(h, weights[i], biases[i]))
        h = layer(h, weights[-1], biases[-1])

        return np.round(h, 5)
