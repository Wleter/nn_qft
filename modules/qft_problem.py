from dataclasses import dataclass
import numpy.typing as npt
from typing import Protocol
import tensorflow as tf

@dataclass
class ConfigurationBatch:
    x_n: npt.NDArray
    """
    Current configurations in a batch of shape [nBatch, nMax, nDim]
    """
    n_s: npt.NDArray
    """
    Current configurations number of particles in a batch of shape [nBatch]
    """

class QFTProblem(Protocol):
    def volume(self) -> npt.NDArray:
        """
        Gets the dimension sizes that are of shape [nDim]
        """
        ...
    
    def get_amplitude(self, x_n: tf.Tensor, ns: npt.NDArray, training: bool = False) -> tf.Tensor:
        """
        Gets the amplitude of particle configuration of shape [nBatch, nParticles, nDim]
        and returns single element tensor.
        """
        ...

class QFTHamiltonian(Protocol):
    def local_energy(self, x_n: tf.Tensor, ns: npt.NDArray, model: QFTProblem, training = False) -> tf.Tensor:
        """
        Gets the local energies of the configuration given current QFT model.
         - returns [nBatch]
        """
        ...
    
    def jastrow_cusp(self, x_ij: tf.Tensor, mask_ij: tf.Tensor, mask_n: tf.Tensor, volume: npt.NDArray) -> tf.Tensor | None:
        """
        Returns the jastrow factor cusp tensor [nBatch] needed for infinite contact potential values
        or None if no such potential is needed.
        """
        return None
