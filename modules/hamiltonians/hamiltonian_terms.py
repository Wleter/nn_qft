from dataclasses import dataclass
from typing import Callable
import numpy as np
import numpy.typing as npt
import tensorflow as tf

from modules.qft_problem import QFTProblem, QFTHamiltonian

@dataclass
class HamiltonianSum:
    hamiltonian_terms: list[QFTHamiltonian]

    def local_energy(self, x_n: tf.Tensor, n_s: npt.NDArray, model: QFTProblem, training = False) -> tf.Tensor:
        return sum(map(lambda h: h.local_energy(x_n, n_s, model, training=training), self.hamiltonian_terms))

    def jastrow_cusp(self, x_ij: tf.Tensor, mask_ij: tf.Tensor, mask_n: tf.Tensor, volume: npt.NDArray) -> tf.Tensor | None:
        cusp = None

        for h in self.hamiltonian_terms:
            cusp_h = h.jastrow_cusp(x_ij, mask_ij, mask_n, volume)

            if cusp_h is not None:
                assert cusp is None, "Only one cusp inducing hamiltonian term is allowed"

                cusp = cusp_h

        return cusp

@dataclass
class KineticTerm:
    mass: float
    
    epsilon: float = 1e-10
    """
    Epsilon that is added to logarithm to prevent overflows
    """

    def local_energy(self, x_n: tf.Tensor, n_s: npt.NDArray, model: QFTProblem, training = False) -> tf.Tensor:
    # with tf.GradientTape() as tape2:
        with tf.GradientTape() as tape1:
            tape1.watch(x_n)
            # tape2.watch(x_n)
            psi_n = model.get_amplitude(x_n, n_s, training=training)
            psi_n_log = tf.math.log(psi_n + self.epsilon)

        gradient = tape1.gradient(psi_n_log, x_n)
        return 1 / (2 * self.mass) * tf.reduce_sum(tf.pow(gradient, 2), axis = [1, 2])

        # hessian = tape2.jacobian(gradient, x_n)
        # laplacian = tf.einsum("bpibpi->bpi", hessian)

# mask is not needed since amplitude should not depend on non existent positions

        # mask = tf.sequence_mask(n_s, maxlen = x_n.shape[-2])
        # mask = tf.cast(mask, x_n.dtype)
        # mask = tf.expand_dims(mask, axis=-1)
        # mask = tf.tile(mask, [1, 1, x_n.shape[-1]])


        # return -1 / (2 * self.mass) * tf.reduce_sum(tf.pow(gradient, 2) + laplacian, axis = [1, 2])

    def jastrow_cusp(self, x_ij: tf.Tensor, mask_ij: tf.Tensor, mask_n: tf.Tensor, volume: npt.NDArray) -> tf.Tensor | None:
        return None
@dataclass
class ExternalPotential:
    potential: Callable[[tf.Tensor, npt.NDArray], tf.Tensor]

    @staticmethod
    def chemical_potential(value: float) -> 'ExternalPotential':
        return ExternalPotential(lambda x, n: -value * tf.cast(n, tf.float32)) # type: ignore

    def local_energy(self, x_n: tf.Tensor, n_s: npt.NDArray, model: QFTProblem, training = False) -> tf.Tensor:
        return self.potential(x_n, n_s)
    
    def jastrow_cusp(self, x_ij: tf.Tensor, mask_ij: tf.Tensor, mask_n: tf.Tensor, volume: npt.NDArray) -> tf.Tensor | None:
        return None
    
@dataclass
class ContactPotential:
    mass: float
    g: float

    def local_energy(self, x_n: tf.Tensor, n_s: npt.NDArray, model: QFTProblem, training = False) -> tf.Tensor:
        return tf.zeros_like(n_s)

    def jastrow_cusp(self, x_ij: tf.Tensor, mask_ij: tf.Tensor, mask_n: tf.Tensor, volume: npt.NDArray) -> tf.Tensor | None:
        vol = np.prod(volume)
        
        x_ij_norm = tf.sqrt(tf.reduce_sum(x_ij * x_ij, axis = -1))

        masked_prod = tf.where(mask_ij[:, :, 0] > 0, x_ij_norm + 1. / (self.mass * self.g * vol), tf.ones_like(x_ij_norm))

        # n_max = mask_n.shape[1]
        # selberg = 1.
        # if self.g > 1e4:
        #     selberg_factors = [2 * tf.math.lgamma(float(1 + j)) + tf.math.lgamma(float(2 + j)) - tf.math.lgamma(float(1 + n_max + j)) + vol \
        #         for j in range(n_max)
        #     ]
        #     selberg_factors = tf.cast(tf.stack(selberg_factors, axis = -1), tf.float32)
        #     selberg = -tf.reduce_sum(selberg_factors * tf.squeeze(mask_n, axis = -1), axis = -1) / 2.

        #     selberg = tf.exp(selberg)

        return tf.reduce_prod(masked_prod, axis = -1)# * selberg
