from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
from typing import Optional

import tensorflow as tf
from modules.qft_problem import QFTProblem

@dataclass(frozen = True)
class FockSpaceMetropolis:
    problem: QFTProblem
    p_plus: float = 0.25
    p_minus: float = 0.25
    configuration_change: float = 0.5

    rng: np.random.Generator = np.random.default_rng()

    def new_configuration(self, n: int) -> tf.Tensor:
        volume = self.problem.volume()
        shape = (n, int(volume.shape[-1]))
        sample = self.rng.uniform(0., 1., size = shape)

        return sample * volume

    def step(self, x_n: npt.NDArray) -> npt.NDArray:
        volume = self.problem.volume()
        choice = self.rng.uniform(0, 1)
        acceptance_rng = self.rng.uniform(0, 1)

        if self.p_plus > choice:
            x_new = self.add_new(x_n)
            x_new_t = tf.expand_dims(tf.convert_to_tensor(x_new), axis = 0)
            x_n_t = tf.expand_dims(tf.convert_to_tensor(x_n), axis = 0)

            ratio = self.problem.get_amplitude(x_new_t) / self.problem.get_amplitude(x_n_t)
            ratio = ratio.numpy()

            acceptance = np.minimum(1, np.prod(volume) * np.abs(ratio) ** 2) # todo! assuming boson symmetry

        elif self.p_minus + self.p_plus > choice:
            x_new = self.remove_one(x_n)
            if x_new is None:
                return x_n
            
            x_new_t = tf.expand_dims(tf.convert_to_tensor(x_new), axis = 0)
            x_n_t = tf.expand_dims(tf.convert_to_tensor(x_n), axis = 0)

            ratio = self.problem.get_amplitude(x_new_t) / self.problem.get_amplitude(x_n_t)
            ratio = ratio.numpy()

            acceptance = np.minimum(1, np.abs(ratio) ** 2 / np.prod(volume)) # todo! assuming boson symmetry
        else:
            x_new = self.change_positions(x_n)

            x_new_t = tf.expand_dims(tf.convert_to_tensor(x_new), axis = 0)
            x_n_t = tf.expand_dims(tf.convert_to_tensor(x_n), axis = 0)

            ratio = self.problem.get_amplitude(x_new_t) / self.problem.get_amplitude(x_n_t)
            ratio = ratio.numpy()

            acceptance = np.minimum(1, np.abs(ratio) ** 2)

        if acceptance > acceptance_rng:
            return x_new
        else:
            return x_n


    def add_new(self, x_n: npt.NDArray) -> npt.NDArray:
        volume = self.problem.volume()

        x = np.array(self.rng.uniform(size=volume.shape), dtype=np.float32) * volume
        x.reshape(volume.shape)
        
        return np.vstack([x_n, x])

    def remove_one(self, x_n: npt.NDArray) -> Optional[npt.NDArray]:
        if x_n.shape[0] == 0:
            return None
        
        index = self.rng.choice(x_n.shape[0])

        return np.delete(x_n, index, 0)

    def change_positions(self, x_n: npt.NDArray) -> npt.NDArray:
        volume = self.problem.volume()

        rel_change = self.configuration_change
        dx_n = self.rng.uniform(-rel_change, rel_change, size = x_n.shape) * volume

        x_n_changed = x_n + dx_n
        for col in range(volume.shape[1]):
            vol = volume[0, col]
            x_n_changed[:, col] = x_n_changed[:, col] % vol

        return x_n_changed
