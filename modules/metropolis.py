from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
from typing import Optional

import tensorflow as tf
from modules.qft_problem import ConfigurationBatch, QFTProblem

# todo! still a little different results from without batching
@dataclass(frozen = True)
class FockSpaceMetropolis:
    problem: QFTProblem
    n_max: int

    p_plus: float = 0.25
    p_minus: float = 0.25
    configuration_change: float = 0.5

    rng: np.random.Generator = np.random.default_rng()

    def new_configuration(self, batch: int, n: int) -> tuple[ConfigurationBatch, tf.Tensor]:
        volume = self.problem.volume()
        shape = (batch, self.n_max, int(volume.shape[0]))
        sample = self.rng.uniform(0., 1., size = shape)

        configurations = ConfigurationBatch(sample * volume, n * np.ones(shape=(batch), dtype=np.int32))

        x_n_t = tf.convert_to_tensor(configurations.x_n)
        ampiltude = self.problem.get_amplitude(x_n_t, configurations.n_s)

        return configurations, ampiltude
 
    def step(self, configurations: ConfigurationBatch, last_amplitude: tf.Tensor) -> tuple[ConfigurationBatch, tf.Tensor]:
        volume = self.problem.volume()
        choice = self.rng.uniform(0, 1, size = (configurations.x_n.shape[0],))
        acceptance_rng = self.rng.uniform(0, 1, size = (configurations.x_n.shape[0],))

        add_mask = self.p_plus > choice
        remove_mask = ((self.p_plus + self.p_minus) > choice) * (np.logical_not(add_mask))
        vary_mask = np.logical_not(np.logical_or(add_mask, remove_mask))

        x_new = np.copy(configurations.x_n)
        n_new = np.copy(configurations.n_s)

        added = self.add_new(ConfigurationBatch(x_new[add_mask, :, :], n_new[add_mask]))
        removed = self.remove_one(ConfigurationBatch(x_new[remove_mask, :, :], n_new[remove_mask]))
        varied = self.change_positions(ConfigurationBatch(x_new[vary_mask, :, :], n_new[vary_mask]))

        x_new[add_mask, :, :] = added.x_n
        x_new[remove_mask, :, :] = removed.x_n
        x_new[vary_mask, :, :] = varied.x_n

        n_new[add_mask] = added.n_s
        n_new[remove_mask] = removed.n_s
        n_new[vary_mask] = varied.n_s

        x_new_t = tf.convert_to_tensor(x_new)

        new_ampiltude = self.problem.get_amplitude(x_new_t, n_new)

        ratio = new_ampiltude / last_amplitude # type: ignore
        ratio = ratio.numpy()

        reject_add = np.minimum(1, np.prod(volume) * np.abs(ratio) ** 2) < acceptance_rng
        reject_remove = np.minimum(1, np.abs(ratio) ** 2 / np.prod(volume)) < acceptance_rng
        reject_vary = np.minimum(1, np.abs(ratio) ** 2) < acceptance_rng

        reject_add = reject_add * add_mask
        reject_remove = reject_remove * remove_mask
        reject_vary = reject_vary * vary_mask

        x_new[reject_add, :, :] = configurations.x_n[reject_add, :, :]
        x_new[reject_remove, :, :] = configurations.x_n[reject_remove, :, :]
        x_new[reject_vary, :, :] = configurations.x_n[reject_vary, :, :]

        n_new[reject_add] = configurations.n_s[reject_add]
        n_new[reject_remove] = configurations.n_s[reject_remove]
        n_new[reject_vary] = configurations.n_s[reject_vary]

        return ConfigurationBatch(x_new, n_new), new_ampiltude

    def add_new(self, configurations: ConfigurationBatch) -> ConfigurationBatch:
        volume = self.problem.volume()

        x = np.copy(configurations.x_n)
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

def make_dataset(problem: QFTProblem, batch: int, n_max: int, n_init = 1, rng = np.random.default_rng()) -> tf.data.Dataset:
    metropolis = FockSpaceMetropolis(problem, n_max, rng=rng)
    n_dim = problem.volume().shape[-1]

    def metropolis_step():
        x_recent, amplitude_recent = metropolis.new_configuration(batch, n_init)
        while True:
            x_recent, amplitude_recent = metropolis.step(x_recent, amplitude_recent)
            yield x_recent.x_n, x_recent.n_s

    dataset = tf.data.Dataset.from_generator(
        metropolis_step,
        output_signature = (
            tf.TensorSpec(shape=(batch, n_max, n_dim), dtype=tf.float32, name = "configuration"), # type: ignore
            tf.TensorSpec(shape=(batch), dtype=tf.int32, name = "particle_no") # type: ignore
        )
    )

    return dataset