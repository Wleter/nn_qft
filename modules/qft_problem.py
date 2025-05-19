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
    
    def get_amplitude(self, configurations: ConfigurationBatch) -> tf.Tensor:
        """
        Gets the amplitude of particle configuration of shape [nBatch, nParticles, nDim]
        and returns single element tensor.
        """
        ...

class QFTHamiltonian(Protocol):
    def local_energy(self, configurations: ConfigurationBatch, model: QFTProblem) -> tf.Tensor:
        """
        Gets the local energies of the configuration given current QFT model.
         - x_n shape [nBatch, nParticles, nDim] returns [nBatch]
        """
        ...
