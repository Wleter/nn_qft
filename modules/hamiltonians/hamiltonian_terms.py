from dataclasses import dataclass
from typing import Callable
import numpy as np
import numpy.typing as npt
import tensorflow as tf

from modules.qft_problem import QFTProblem

@dataclass
class KineticTerm:
    mass: float

    def local_energy(self, x_n: tf.Tensor, model: QFTProblem) -> tf.Tensor:
        with tf.GradientTape() as tape2:
            with tf.GradientTape() as tape1:
                tape1.watch(x_n)
                tape2.watch(x_n)
                psi_n = model.get_amplitude(x_n)
                psi_n_log = tf.math.log(psi_n)

            gradient = tape1.gradient(psi_n_log, x_n)

        hessian = tape2.jacobian(gradient, x_n)
        laplacian = tf.einsum("bpibpi->bpi", hessian)
            
        return -1 / (2 * self.mass) * tf.reduce_sum(laplacian + tf.pow(gradient, 2))

@dataclass
class ExternalPotential:
    potential: Callable[[tf.Tensor], tf.Tensor]

    @staticmethod
    def chemical_potential(value: float) -> 'ExternalPotential':
        return ExternalPotential(lambda x: value * tf.ones((x.shape[0], 1)))

    def local_energy(self, x_n: tf.Tensor, model: QFTProblem) -> tf.Tensor:
        return self.potential(x_n)