
from dataclasses import dataclass
import tensorflow as tf
from modules.qft_problem import QFTProblem
import numpy as np


@dataclass
class NRKineticTerm:
    mass: float

    def local_energy(self, x_n: tf.Tensor, model: QFTProblem) -> tf.Tensor:
        # model.get_amplitude(x_n)
        ...
    
    def local_energy_gradient(self, x_n: tf.Tensor, model: QFTProblem) -> tf.Tensor:
        ...
