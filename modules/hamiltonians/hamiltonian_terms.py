from dataclasses import dataclass
from typing import Callable
import numpy as np
import numpy.typing as npt
import tensorflow as tf

from modules.qft_problem import QFTProblem

@dataclass
class KineticTerm:
    mass: float

    def local_energy(self, x_n: tf.Tensor, n_s: npt.NDArray, model: QFTProblem) -> tf.Tensor:
        with tf.GradientTape() as tape2:
            with tf.GradientTape() as tape1:
                tape1.watch(x_n)
                tape2.watch(x_n)
                psi_n = model.get_amplitude(x_n, n_s)
                psi_n_log = tf.math.log(psi_n)

            gradient = tape1.gradient(psi_n_log, x_n)

        hessian = tape2.jacobian(gradient, x_n)
        laplacian = tf.einsum("bpibpi->bpi", hessian)

# mask is not needed since amplitude should not depend on non existent positions

        # mask = tf.sequence_mask(n_s, maxlen = x_n.shape[-2])
        # mask = tf.cast(mask, x_n.dtype)
        # mask = tf.expand_dims(mask, axis=-1)
        # mask = tf.tile(mask, [1, 1, x_n.shape[-1]])

        return -1 / (2 * self.mass) * tf.reduce_sum(laplacian + tf.pow(gradient, 2), axis = [1, 2])

@dataclass
class ExternalPotential:
    potential: Callable[[tf.Tensor, npt.NDArray], tf.Tensor]

    @staticmethod
    def chemical_potential(value: float) -> 'ExternalPotential':
        return ExternalPotential(lambda x, n: tf.cast(n, tf.float32) * value) # type: ignore

    def local_energy(self, x_n: tf.Tensor, n_s: npt.NDArray, model: QFTProblem) -> tf.Tensor:
        return self.potential(x_n, n_s)