from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
from typing import Optional, Protocol

# todo! maybe change to [batch, nParticles, nDim]
class QFTProblem(Protocol):
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


@dataclass(frozen = True)
class FockSpaceMetropolis:
    problem: QFTProblem
    p_plus: float = 0.25
    p_minus: float = 0.25
    configuration_change: float = 0.5

    rng: np.random.Generator = np.random.default_rng()

    def new_configuration(self, n: int) -> npt.NDArray:
        volume = self.problem.volume()

        return np.array(self.rng.uniform(size=(n, volume.shape[1]))) * volume

    def step(self, x_n: npt.NDArray) -> Optional[npt.NDArray]:
        volume = self.problem.volume()
        choice = self.rng.uniform(0, 1)
        acceptance_rng = self.rng.uniform(0, 1)

        if self.p_plus > choice:
            x_new = self.add_new(x_n)
            acceptance = np.minimum(1, np.prod(volume) * # todo! assuming boson symmetry
                np.abs(self.problem.get_amplitude(x_new) / self.problem.get_amplitude(x_n)) ** 2)

        elif self.p_minus + self.p_plus > choice:
            x_new = self.remove_one(x_n)
            if x_new is None:
                return None

            acceptance = np.minimum(1, 1 / np.prod(volume) * # todo! assuming boson symmetry
                np.abs(self.problem.get_amplitude(x_new) / self.problem.get_amplitude(x_n)) ** 2)
        else:
            x_new = self.change_positions(x_n)
            acceptance = np.minimum(1, 
                np.abs(self.problem.get_amplitude(x_new) / self.problem.get_amplitude(x_n)) ** 2)

        if acceptance > acceptance_rng:
            return x_new
        else:
            return None


    def add_new(self, x_n: npt.NDArray) -> npt.NDArray:
        volume = self.problem.volume()

        x = np.array(self.rng.uniform(size=volume.shape)) * volume
        x.reshape(volume.shape)
        
        return np.vstack([x_n, x])

    def remove_one(self, x_n: npt.NDArray) -> Optional[npt.NDArray]:
        if x_n.shape[0] == 0:
            return None
        
        index = np.random.choice(x_n.shape[0])

        return np.delete(x_n, index, 0)
    
    def change_positions(self, x_n: npt.NDArray) -> npt.NDArray:
        volume = self.problem.volume()

        rel_change = self.configuration_change
        dx_n = self.rng.uniform(-rel_change, rel_change, size = x_n.shape) * volume

        x_n_changed = x_n + dx_n
        for col in range(volume.shape[1]):
            vol = volume[0, col]
            x_n_changed[:, col] = x_n_changed[:, col].clip(0, vol)

        return x_n_changed
