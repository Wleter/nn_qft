from modules.metropolis import *
import unittest

@dataclass
class SimpleProblem(QFTProblem):
    size: float = 10

    def volume(self) -> npt.NDArray:
        return np.array([10, 100]).reshape((1, 2))

    def get_amplitude(self, x_n: npt.NDArray) -> float:
        return 1 / np.power(np.prod(self.volume()), x_n.shape[0] / 2)

class TestMetropolis(unittest.TestCase):
    def test_metropolis(self):
        test_problem = SimpleProblem()

        metropolis = FockSpaceMetropolis(test_problem, rng=np.random.default_rng())

        x_n = metropolis.new_configuration(5)
        print("initial\n", x_n)
        
        x_add = metropolis.add_new(x_n)
        print("add new\n", x_add)

        x_remove = metropolis.remove_one(x_n)
        print("remove one\n", x_remove)
                    
        x_changed = metropolis.change_positions(x_n)
        print("change position\n", x_changed)
        
        x_changed = metropolis.change_positions(x_n)
        print("change position\n", x_changed)
        
        x_n = metropolis.new_configuration(0)
        print("initial\n", x_n)
        
        x_remove = metropolis.remove_one(x_n)
        print("remove one\n", x_remove)
        
        x_add = metropolis.add_new(x_n)
        print("add new\n", x_add)

        x_recent = metropolis.new_configuration(5)
        configurations = [x_recent]
        for _ in range(100):
            result = metropolis.step(x_recent)
            if result is None:
                continue

            x_recent = result
            configurations.append(x_recent)

        mean_n = np.mean(list(map(lambda x: x.shape[0], configurations)))
        print(configurations)
        print(mean_n)


if __name__ == "__main__":
    unittest.main()
