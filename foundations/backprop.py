import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def backward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, y_true: float) -> Tuple[NDArray[np.float64], float]:
        # x: 1D input array
        # w: 1D weight array
        # b: scalar bias
        # y_true: true target value
        #
        # Forward: z = dot(x, w) + b, y_hat = sigmoid(z)
        # Loss: L = 0.5 * (y_hat - y_true)^2
        # Return: (dL_dw rounded to 5 decimals, dL_db rounded to 5 decimals)
        z = np.dot(w, x) + b
        y_pred = 1 / (1 + np.exp(-z))
        dL_dw = np.zeros(len(w))
        dL_db = np.dot(y_pred - y_true, np.dot(y_pred, (1 - y_pred)))
        for i in range(len(w)):
            dL_dw[i] = np.dot(dL_db, x[i])
        return (np.round(dL_dw, 5), round(dL_db, 5))
