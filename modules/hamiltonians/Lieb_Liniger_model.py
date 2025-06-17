from modules.hamiltonians.hamiltonian_terms import *

class LiebLinigerHamiltonian:
    def __init__(self, mass: float, mu: float, g: float):
        self.kinetic_term = KineticTerm(mass)
        self.chemical_potential = ExternalPotential.chemical_potential(mu)
        self.contact_term = ContactPotential(mass, g)

    def accept(self, model: QFTProblem):
        self.kinetic_term.accept(model)

    def local_energy(self, x_n: tf.Tensor, n_s: tf.Tensor) -> tf.Tensor:
        kinetic = self.kinetic_term.local_energy(x_n, n_s)
        chemical = self.chemical_potential.local_energy(x_n, n_s)

        return kinetic + chemical

    def jastrow_cusp(self, x_ij: tf.Tensor, mask_ij: tf.Tensor, n: tf.Tensor, n_max: float, volume: npt.NDArray) -> tf.Tensor | None:
        return self.contact_term.jastrow_cusp(x_ij, mask_ij, n, n_max, volume)
