from modules.hamiltonians.hamiltonian_terms import *

class LiebLinigerHamiltonian:
    def __init__(self, mass: float, mu: float, g: float):
        self.kinetic_term = KineticTerm(mass)
        self.chemical_potential = ExternalPotential.chemical_potential(mu)
        self.contact_term = ContactPotential(mass, g)

    def local_energy(self, x_n: tf.Tensor, n_s: npt.NDArray, model: QFTProblem, training = False) -> tf.Tensor:
        kinetic = self.kinetic_term.local_energy(x_n, n_s, model, training=training)
        chemical = self.chemical_potential.local_energy(x_n, n_s, model, training=training)

        return kinetic + chemical

    def jastrow_cusp(self, x_ij: tf.Tensor, mask_ij: tf.Tensor, mask_n: tf.Tensor, volume: npt.NDArray) -> tf.Tensor | None:
        return self.contact_term.jastrow_cusp(x_ij, mask_ij, mask_n, volume)
