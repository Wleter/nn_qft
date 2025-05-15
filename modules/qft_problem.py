import numpy.typing as npt
from typing import Protocol

import tensorflow as tf

class QFTProblem(Protocol):
    def volume(self) -> npt.NDArray:
        """
        Gets the dimension sizes that are of shape [1, nDim]
        """
        ...
    
    def get_amplitude(self, x_n: tf.Tensor) -> tf.Tensor:
        """
        Gets the amplitude of particle configuration of shape [1, nParticles, nDim]
        and returns single element tensor.
        """
        ...


class QFTHamiltonian(Protocol):
    def local_energy(self, x_n: tf.Tensor, model: QFTProblem) -> tf.Tensor:
        """
        Gets the expectation energy of the current QFT model in a batch.
         - x_n shape [nParticles, nDim]
        """
        ...
