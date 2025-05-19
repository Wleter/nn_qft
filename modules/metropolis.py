from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
from typing import Optional

import tensorflow as tf
from modules.qft_problem import ConfigurationBatch, QFTProblem

@dataclass(frozen = True)
class FockSpaceMetropolis:
    problem: QFTProblem
    p_plus: float = 0.25
    p_minus: float = 0.25
    configuration_change: float = 0.5

    n_max = 20

    rng: np.random.Generator = np.random.default_rng()

    def new_configuration(self, batch: int, n: int) -> ConfigurationBatch:
        volume = self.problem.volume()
        shape = (batch, self.n_max, int(volume.shape[0]))
        sample = self.rng.uniform(0., 1., size = shape)

        return ConfigurationBatch(sample * volume, n * np.ones(shape=(batch), dtype=np.int32))
 
    def step(self, configurations: ConfigurationBatch) -> ConfigurationBatch:
        volume = self.problem.volume()
        choice = self.rng.uniform(0, 1, size = (configurations.x_n.shape[0],))
        acceptance_rng = self.rng.uniform(0, 1, size = (configurations.x_n.shape[0],))

        add_mask = self.p_plus > choice
        remove_mask = ((self.p_plus + self.p_minus) > choice) * (np.logical_not(add_mask))
        vary_mask = np.logical_not(add_mask * remove_mask)

        x_new = configurations.x_n
        n = configurations.n_s

        x_new[add_mask, :, :] = self.add_new(ConfigurationBatch(x_new[add_mask, :, :], n[add_mask]))
        x_new[add_mask, :, :] = self.remove_one(ConfigurationBatch(x_new[remove_mask, :, :], n[remove_mask]))
        x_new[vary_mask, :, :] = self.change_positions(ConfigurationBatch(x_new[vary_mask, :, :], n[vary_mask]))

        x_new_t = tf.expand_dims(tf.convert_to_tensor(x_new), axis = 0)
        x_n_t = tf.expand_dims(tf.convert_to_tensor(configurations.x_n), axis = 0)

        ratio = self.problem.get_amplitude(x_new_t) / self.problem.get_amplitude(x_n_t) # type: ignore
        ratio = ratio.numpy()

        reject_add = np.minimum(1, np.prod(volume) * np.abs(ratio) ** 2) <= acceptance_rng
        reject_remove = np.minimum(1, np.abs(ratio) ** 2 / np.prod(volume)) <= acceptance_rng
        reject_vary = np.minimum(1, np.abs(ratio) ** 2) <= acceptance_rng

        x_new[reject_add, :, :] = configurations.x_n[reject_add, :, :]
        n[reject_add] = configurations.n_s[reject_add]
        x_new[reject_remove, :, :] = configurations.x_n[reject_remove, :, :]
        n[reject_remove] = configurations.n_s[reject_remove]
        x_new[reject_vary, :, :] = configurations.x_n[reject_vary :, :]
        n[reject_vary] = configurations.n_s[reject_vary]

        return ConfigurationBatch(x_new, n)

    def add_new(self, configurations: ConfigurationBatch) -> ConfigurationBatch:
        volume = self.problem.volume()

        x = configurations.x_n
        for i, n in enumerate(configurations.n_s):
            if n != x.shape[1]:
                x[i, n, :] = self.rng.uniform(size=volume.shape) * volume

        n = np.clip(configurations.n_s + 1, 0, x.shape[1])

        return ConfigurationBatch(x, n)

    def remove_one(self, configurations: ConfigurationBatch) -> ConfigurationBatch:
        n = np.clip(configurations.n_s - 1, 0, configurations.x_n.shape[1])

        return ConfigurationBatch(configurations.x_n, n)

    def change_positions(self, configurations: ConfigurationBatch) -> ConfigurationBatch:
        volume = self.problem.volume()
        x_n = configurations.x_n

        rel_change = self.configuration_change
        dx_n = self.rng.uniform(-rel_change, rel_change, size = x_n.shape) * volume

        x_n_changed = x_n + dx_n
        for col in range(volume.shape[-1]):
            vol = volume[col]

            x_n_changed[:, :, col] = (x_n_changed[:, :, col] % vol)

        return ConfigurationBatch(x_n_changed, configurations.n_s)
