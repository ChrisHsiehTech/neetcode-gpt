import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        shift = z - max(z)
        shift = np.exp(shift)
        ans = shift/sum(shift)
        return np.round(ans, 4)