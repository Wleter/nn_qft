from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
from typing import Protocol

# todo! maybe change to [batch, nParticles, nDim]
class Problem(Protocol):
    def volume(self) -> npt.NDArray:
        """
        Gets the dimension sizes that are of shape [1, nDim]
        """
        ...
    
    
    def get_amplitude(self, x_n: npt.NDArray) -> float:
        """
        Gets the amplitude of particle configuration of shape [nParticles, nDim]
        """
        ...


@dataclass
class FockSpaceMetropolis:
    problem: Problem
    p_plus: float = 0.25
    p_minus: float = 0.25
    configuration_change: float = 0.3

    def propose_new(self, x_n: npt.NDArray) -> npt.NDArray:
        choice = np.random.default_rng().uniform(0, 1)

        if self.p_plus > choice:
            return self.add_new(x_n)
        if self.p_minus + self.p_plus > choice:
            self.remove_one(x_n)
        else:
            pass
        ...

    def add_new(self, x_n: npt.NDArray) -> npt.NDArray:
        volume = self.problem.volume()

        x = np.array([np.random.default_rng().uniform(0, v) for v in volume])
        x.reshape(volume.shape)
        
        return np.stack((x_n, x), axis = 0)

    def remove_one(self, x_n: npt.NDArray) -> npt.NDArray:
        if x_n.shape[0] == 0:
            return x_n
        
        index = np.random.choice(x_n.shape[0])

        return np.delete(x_n, index, 0)
