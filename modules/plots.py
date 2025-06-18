from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

def plot(nrows = 1, ncols = 1, figsize = None):
    fig, ax = plt.subplots(nrows, ncols, figsize = figsize)

    if nrows == 1 and ncols == 1:
        ax.grid()
        ax.tick_params(which='both', direction="in")
    else:
        for a in ax:
            a.grid()
            a.tick_params(which='both', direction="in")
    fig.tight_layout()

    return fig, ax

def plot_train_history(history, exact_energy = None, exact_particle_number = None):
    fig, axes = plot(1, 2, figsize = (15, 5))
    
    axes[0].plot(history.history['energy'], label = "training")
    if exact_energy is not None:
        axes[0].axhline(exact_energy, label = "exact", color = "red")

    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Estimated energy')
    axes[0].legend()
    
    axes[1].plot(history.history['mean_particle_number'], label = "training")
    if exact_particle_number is not None:
        axes[1].axhline(exact_particle_number, label = "exact", color = "red")

    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Mean particle number')
    axes[1].legend()


    return fig, axes

def plot_metropolis_samples(dataset, dim = 1, bins = 50):
    configurations = dataset.cache()
    fig, axes = plot(1, dim + 1, figsize = (7 * (dim + 1), 5))

    n_values = list(map(lambda x_n: x_n[1].numpy(), configurations))
    n_counter = Counter(n_values)
    axes[0].bar(n_counter.keys(), n_counter.values())
    axes[0].set_xlabel("Number of particles")
    axes[0].set_ylabel("Count")

    for i in range(dim):
        positions = np.concatenate(list(map(lambda x_n: x_n[0][:x_n[1], i], configurations))).ravel()
        axes[i+1].hist(positions, bins = bins)
        axes[i+1].set_xlabel("Positions")
        axes[i+1].set_ylabel("Count")

    return fig, axes
