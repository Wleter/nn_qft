import numpy.typing as npt
from typing import Protocol
import tensorflow as tf

class QFTProblem(Protocol):
    def volume(self) -> tf.Tensor:
        """
        Gets the dimension sizes that are of shape [1, 1, nDim]
        """
        ...
    
    def get_amplitude(self, x_n: tf.Tensor) -> tf.Tensor:
        """
        Gets the amplitude of particle configuration of shape [batch, nParticles, nDim]
         - Output [batch, 1]
        """
        ...

class QFTHamiltonian(Protocol):
    def local_energy(self, x_n: tf.Tensor, model: QFTProblem) -> tf.Tensor:
        """
        Gets the expectation energy of the current QFT model in a batch.
         - x_n shape [batch, nParticles, nDim]
        """
        ...
    
    def local_energy_gradient(self, x_n: tf.Tensor, model: QFTProblem) -> tf.Tensor:
        """
        Gets the gradient of the energy in respect to the current model parameters in a batch.
         - x_n shape [batch, nParticles, nDim]
        """
        ...
