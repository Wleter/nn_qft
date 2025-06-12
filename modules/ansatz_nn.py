from dataclasses import dataclass
import numpy as np
import numpy.typing as npt

import tensorflow as tf
import keras
from modules.qft_problem import QFTHamiltonian

class DeepSets(keras.Model):
    def __init__(self, hidden_neurons, hidden_layers_phi, hidden_layers_rho, input_dim = 2, **kwargs):
        super(DeepSets, self).__init__(**kwargs)

        self.hidden_neurons = hidden_neurons
        self.hidden_layers_phi = hidden_layers_phi
        self.hidden_layers_rho = hidden_layers_rho
        self.input_dim = input_dim

        self.phi = keras.Sequential(
                layers = [
                    keras.Input(shape=(self.input_dim,)),
                    *(keras.layers.Dense(self.hidden_neurons, activation = "relu") for _ in range(self.hidden_layers_phi)),
                ],
                name = "phi",
        )

        self.rho = keras.Sequential(
                layers = [
                    *(keras.layers.Dense(self.hidden_neurons, activation = "relu") for _ in range(self.hidden_layers_phi)),
                    keras.layers.Dense(1, activation = "softplus")
                ],
                name = "rho",
        )

    def call(self, x, mask, training=False):
        batch_size = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]

        y = tf.reshape(x, [-1, self.input_dim])
        y = self.phi(y, training=training)

        y = tf.reshape(y, (batch_size, seq_len, self.hidden_neurons))
        y = tf.reduce_sum(y * mask, axis=1)

        y = self.rho(y, training=training)
        return y
    
def x_difference(x_n: tf.Tensor, mask_n: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    idx = tf.range(tf.shape(x_n)[1])
    ii, jj = tf.meshgrid(idx, idx, indexing='ij')
    mask = ii < jj

    x_i = tf.expand_dims(x_n, 2)
    x_j = tf.expand_dims(x_n, 1)
    D = x_i - x_j
    x_diff = tf.boolean_mask(D, mask, axis=1)

    mask_i = tf.expand_dims(mask_n, 2)
    mask_j = tf.expand_dims(mask_n, 1)
    D = mask_i * mask_j
    mask_ij = tf.boolean_mask(D, mask, axis = 1)

    return x_diff, mask_ij

class ParticleNoReg(keras.Model):
    def __init__(self, c_mean: float, c_diff: float, slope: float):
        super(ParticleNoReg, self).__init__()

        self.q1 =  self.add_weight(initializer = keras.initializers.Constant(c_mean - c_diff / 2), trainable = True, name = "q1")
        self.q2 =  self.add_weight(initializer = keras.initializers.Constant(c_mean + c_diff / 2), trainable = True, name = "q1")
        self.slope =  self.add_weight(initializer = keras.initializers.Constant(slope), trainable = True, name = "q1")
        
    def call(self, n, training = False):
        return tf.sigmoid(self.slope * (n - self.q1)) * tf.sigmoid(-self.slope * (n - self.q2))

class QFTNeuralNet(keras.Model):
    def __init__(
        self, 
        volume: npt.NDArray, 
        ds1: DeepSets, 
        ds2: DeepSets, 
        hamiltonian: QFTHamiltonian,
        is_periodic: bool,
        steps_per_epoch = 1, 
        regularization = ParticleNoReg(2., 0.3, 0.3),
    ):
        super(QFTNeuralNet, self).__init__()

        self.volume_arr = np.astype(volume, np.float32)

        assert volume.shape[-1] == ds1.input_dim, "Volume of the problem and input_dim of the deep sets should be the same"
        assert volume.shape[-1] == ds1.input_dim, "Volume of the problem and input_dim of the deep sets should be the same"

        self.loss_tracker = keras.metrics.Mean(name = "loss")
        self.energy_tracker = keras.metrics.Mean(name = "energy")
        self.particle_tracker = keras.metrics.Mean(name = "mean_particle_number")

        self.ds1 = ds1
        self.ds2 = ds2
        self.regularization = regularization
        
        self.hamiltonian = hamiltonian
        self.is_periodic = is_periodic

        self.gradient_accumulators = None
        self.steps_per_epoch = steps_per_epoch

    def volume(self) -> npt.NDArray:
        return self.volume_arr

    def get_amplitude(self, x_n: tf.Tensor, ns: npt.NDArray, training: bool = False) -> tf.Tensor:
        return self.call(x_n, ns, training = training)

    def call(self, x, n, training=False):
        # max_len = tf.reduce_max(n) + 1
        # x = x[:, :max_len, :]
        x_norm = x / self.volume_arr

        mask_n_bool = tf.sequence_mask(n, maxlen = x_norm.shape[1])
        mask_n = tf.cast(mask_n_bool, x_norm.dtype)
        mask_n = tf.expand_dims(mask_n, -1)

        y1 = self.ds1(x_norm, mask_n, training=training)

        x_ij, mask_ij = x_difference(x_norm, mask_n)
        y2 = self.ds2(x_ij, mask_ij, training=training)

        n_float = tf.cast(n, tf.float32)

        reg = self.regularization(n_float, training = training)

        cusp = self.hamiltonian.jastrow_cusp(x_ij, mask_ij, mask_n, self.volume_arr)
        if cusp is not None:
            reg *= tf.cast(cusp, tf.float32)

        if not self.is_periodic:
            cutoff = tf.where(mask_n_bool, tf.cast(tf.reduce_prod(x_norm * (1. - x_norm), axis = -1), tf.float32), tf.ones_like(mask_n_bool, dtype = tf.float32))

            cutoff = tf.reduce_prod(cutoff, axis = 1)
            cutoff *= tf.pow(30. / tf.reduce_sum(self.volume()), n_float / 2.)
            reg *= cutoff

        return reg / tf.pow(tf.reduce_sum(self.volume()), n_float / 2.) * tf.squeeze(y1 * y2, axis = 1)

    def compile(self, optimizer, **kwargs):
        super().compile(optimizer=optimizer, metrics=None, **kwargs)

        self.gradient_accumulators = None

    def train_step(self, data):
        x, n = data

        with tf.GradientTape() as tape:
            loss = tf.reduce_mean(self.hamiltonian.local_energy(x, n, self, training=True))
        
        gradients = tape.gradient(loss, self.trainable_variables)

        if self.gradient_accumulators is None:
            self.gradient_accumulators = [
                tf.Variable(tf.zeros_like(var), trainable=False)
                for var in gradients
            ]

        for acc, grad in zip(self.gradient_accumulators, gradients): # type: ignore
            acc.assign_add(grad)

        self.loss_tracker.update_state(loss)
        self.energy_tracker.update_state(loss)
        self.particle_tracker.update_state(tf.reduce_mean(tf.cast(n, tf.float32)))
        
        metrics_dict = {m.name: m.result() for m in self.metrics}
        metrics_dict[self.loss_tracker.name] = self.loss_tracker.result()
        metrics_dict[self.particle_tracker.name] = self.particle_tracker.result()
        metrics_dict[self.energy_tracker.name] = self.energy_tracker.result()

        return metrics_dict

    def on_epoch_end(self):
        averaged_gradients = [
            acc / tf.cast(self.steps_per_epoch, tf.float32)
            for acc in self.gradient_accumulators
        ]
        self.optimizer.apply_gradients(zip(averaged_gradients, self.trainable_variables))

        for acc in self.gradient_accumulators:
            acc.assign(tf.zeros_like(acc))

    def fit(self, *args, **kwargs):
        callbacks = kwargs.get('callbacks', [])
        if callbacks is None:
            callbacks = []

        callbacks.append(ApplyGradientCallback())

        kwargs['callbacks'] = callbacks

        return super().fit(*args, **kwargs)
    
class ApplyGradientCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        self.model.on_epoch_end()