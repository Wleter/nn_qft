from dataclasses import dataclass
from typing import Callable
import numpy as np
import numpy.typing as npt
import tensorflow as tf

from modules.qft_problem import QFTProblem, QFTHamiltonian
from modules.ansatz_nn import x_difference

@dataclass
class HamiltonianSum:
    hamiltonian_terms: list[QFTHamiltonian]

    def accept(self, model: QFTProblem):
        for h in self.hamiltonian_terms:
            h.accept(model)

    def local_energy(self, x_n: tf.Tensor, n_s: tf.Tensor) -> tf.Tensor:
        return sum(map(lambda h: h.local_energy(x_n, n_s), self.hamiltonian_terms))

    def jastrow_cusp(self, x_ij: tf.Tensor, mask_ij: tf.Tensor, n: tf.Tensor, n_max: float, volume: npt.NDArray) -> tf.Tensor | None:
        cusp = None

        for h in self.hamiltonian_terms:
            cusp_h = h.jastrow_cusp(x_ij, mask_ij, n, n_max, volume)

            if cusp_h is not None:
                assert cusp is None, "Only one cusp inducing hamiltonian term is allowed"

                cusp = cusp_h

        return cusp

@dataclass
class KineticTerm:
    mass: float
    model: QFTProblem | None = None
    
    epsilon: float = 1e-10
    """
    Epsilon that is added to logarithm to prevent overflows
    """
    def accept(self, model: QFTProblem):
        self.model = model

    def local_energy(self, x_n: tf.Tensor, n_s: tf.Tensor) -> tf.Tensor:
    # with tf.GradientTape() as tape2:
        with tf.GradientTape() as tape1:
            tape1.watch(x_n)
            # tape2.watch(x_n)
            psi_n = self.model.get_amplitude(x_n, n_s, training = True)
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

    def jastrow_cusp(self, x_ij: tf.Tensor, mask_ij: tf.Tensor, n: tf.Tensor, n_max: float, volume: npt.NDArray) -> tf.Tensor | None:
        return None
@dataclass
class ExternalPotential:
    potential: Callable[[tf.Tensor, npt.NDArray], tf.Tensor]

    @staticmethod
    def chemical_potential(value: float) -> 'ExternalPotential':
        return ExternalPotential(lambda x, n: -value * tf.cast(n, tf.float32)) # type: ignore
    
    def accept(self, model: QFTProblem):
        pass

    def local_energy(self, x_n: tf.Tensor, n_s: tf.Tensor) -> tf.Tensor:
        return self.potential(x_n, n_s)
    
    def jastrow_cusp(self, x_ij: tf.Tensor, mask_ij: tf.Tensor, n: tf.Tensor, n_max: float, volume: npt.NDArray) -> tf.Tensor | None:
        return None
    
@dataclass
class ContactPotential:
    mass: float
    g: float

    def accept(self, model: QFTProblem):
        pass

    def local_energy(self, x_n: tf.Tensor, n_s: tf.Tensor) -> tf.Tensor:
        return tf.zeros_like(n_s)

    def jastrow_cusp(self, x_ij: tf.Tensor, mask_ij: tf.Tensor, n: tf.Tensor, n_max: float, volume: npt.NDArray) -> tf.Tensor | None:
        vol = np.prod(volume)
        
        x_ij_norm = tf.sqrt(tf.reduce_sum(x_ij * x_ij, axis = -1))

        masked_prod = tf.where(mask_ij[:, :, 0] > 0, x_ij_norm + 1. / (self.mass * self.g * vol), tf.ones_like(x_ij_norm))

        selberg = 1.
        if self.g > 1e4:
            mask_n_bool = tf.sequence_mask(n, maxlen = n_max)
            mask_n = tf.cast(mask_n_bool, tf.float32)
            mask_n = tf.expand_dims(mask_n, -1)

            selberg_factors = [2 * tf.math.lgamma(float(1 + j)) + tf.math.lgamma(float(2 + j)) - tf.math.lgamma(tf.cast(1 + n + j, tf.float32)) + vol \
                for j in range(n_max)
            ]
            selberg_factors = tf.cast(tf.stack(selberg_factors, axis = -1), tf.float32)
            selberg = -tf.reduce_sum(selberg_factors * tf.cast(tf.squeeze(mask_n, axis = -1), tf.float32), axis = -1) / 2.

            selberg = tf.exp(selberg)

        return tf.cast(tf.reduce_prod(masked_prod, axis = -1), tf.float32) * selberg

@dataclass
class InvSineSqrPotential:
    mass: float
    g: float

    vol: float

    epsilon: float = 1e-10

    def accept(self, model: QFTProblem):
        self.vol = np.prod(model.volume())

    def local_energy(self, x_n: tf.Tensor, n_s: tf.Tensor) -> tf.Tensor:
        mask_n_bool = tf.sequence_mask(n_s, maxlen = x_n.shape[1])
        mask_n = tf.cast(mask_n_bool, x_n.dtype)
        mask_n = tf.expand_dims(mask_n, -1)

        x_ij, mask_ij = x_difference(x_n, mask_n)

        w_ij = self.g * (np.pi / self.vol) ** 2 * tf.reduce_sum(1. / (tf.pow(tf.sin(np.pi * x_ij), 2) + self.epsilon) * mask_ij, axis = [1, 2])

        return w_ij

    def jastrow_cusp(self, x_ij: tf.Tensor, mask_ij: tf.Tensor, n: tf.Tensor, n_max: float, volume: npt.NDArray) -> tf.Tensor | None:
        x_ij_norm = tf.sqrt(tf.reduce_sum(x_ij * x_ij, axis = -1))
        lamb = 0.5 * (1. + np.sqrt(1. + 4. * self.mass * self.g))

        masked_prod = tf.where(
            mask_ij[:, :, 0] > 0, 
            tf.pow(x_ij_norm * (1 - x_ij_norm), lamb), 
            tf.ones_like(x_ij_norm)
        )

        return tf.reduce_prod(masked_prod, axis = -1)

@dataclass
class ParticleInteraction:
    potential: Callable[[tf.Tensor], tf.Tensor]

    def accept(self, model: QFTProblem):
        pass

    def local_energy(self, x_n: tf.Tensor, n_s: tf.Tensor) -> tf.Tensor:
        mask_n_bool = tf.sequence_mask(n_s, maxlen = x_n.shape[1])
        mask_n = tf.cast(mask_n_bool, x_n.dtype)
        mask_n = tf.expand_dims(mask_n, -1)

        x_ij, mask_ij = x_difference(x_n, mask_n)

        w_ij = tf.reduce_sum(self.potential(x_ij) * mask_ij, axis = [1, 2])

        return w_ij

    def jastrow_cusp(self, x_ij: tf.Tensor, mask_ij: tf.Tensor, n: tf.Tensor, n_max: float, volume: npt.NDArray) -> tf.Tensor | None:
        return None
