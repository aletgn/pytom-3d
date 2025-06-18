import pickle

import torch
torch.set_default_dtype(torch.float64)
device = "cpu"

import pyro
pyro.set_rng_seed(0)
import pyro.contrib.gp as gp
import pyro.optim as optim
from pyro.infer import Trace_ELBO, SVI

from tqdm import tqdm
from tqdm import trange

from matplotlib import pyplot as plt


def make_gpr(X, y, var=torch.tensor(1.0), length=torch.tensor(2.0), noise=torch.tensor(1.0e-6), jitter=1.e-12):
    kernel = gp.kernels.RBF(input_dim=2, variance=var, lengthscale=length)
    gpr = gp.models.GPRegression(X, y, kernel, noise=noise, jitter=jitter)
    return gpr, kernel


def train(gpr, lr=0.1, steps=1000, log=100, position=0):
    pyro.clear_param_store()
    
    optimiser = optim.Adam({"lr": lr})
    loss_fn = Trace_ELBO()
    losses = []
    svi = SVI(gpr.model, gpr.guide, optimiser, loss=loss_fn)

    pbar = trange(steps, desc=f"Optimising [{position}]", position=position, leave=True)
    for step in pbar:
        loss = svi.step()
        losses.append(loss)

        if step % log == 0:
            tqdm.write(f"[{position}] Step {step} - Loss: {loss:.5f}")
            tqdm.write(f"[{position}] * Variance: {gpr.kernel.variance.item():.5f}")
            tqdm.write(f"[{position}] * Lengthscale: {gpr.kernel.lengthscale.item():.5f}")
            tqdm.write(f"[{position}] * Noise: {gpr.noise.item():.5f}\n")
    
    print(f"[{position}] ---------- Summary ----------")
    print(f"[{position}] * Variance: {gpr.kernel.variance}")
    print(f"[{position}] * Lengthscale: {gpr.kernel.lengthscale}")
    print(f"[{position}] * Noise: {gpr.noise}")

    return gpr, losses


def save_gpr_parameters(path, gpr, losses=None):
    
    out_dict = {"variance": gpr.kernel.variance.item(),
                "length": gpr.kernel.lengthscale.item(),
                "noise": gpr.noise.item(),
                "jitter": gpr.jitter, "losses": losses}
    
    with open(path, "wb") as f:
        pickle.dump(out_dict, f)


def load_gpr_parameters(path):
    with open(path, "rb") as f:
        loaded_dict = pickle.load(f)
    return loaded_dict


def inspect_gpr(X, y, gpr):
    grid_x1, grid_x2 = torch.meshgrid(torch.linspace(X[:, 0].min(), X[:, 0].max(), 50),
                                      torch.linspace(X[:, 1].min(), X[:, 1].max(), 50), indexing='ij')
    X_grid = torch.stack([grid_x1.reshape(-1), grid_x2.reshape(-1)], dim=-1)
    mean, cov = gpr(X_grid, full_cov=True)
    std = cov.diag().sqrt()

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:, 0].numpy(), X[:, 1].numpy(), y.numpy(), c='b', label='Training Data', alpha=0.3)

    ax.plot_trisurf(X_grid[:, 0].detach().numpy(), 
                    X_grid[:, 1].detach().numpy(),
                    mean.detach().numpy(), cmap='viridis', alpha=0.7)

    # ax.plot_trisurf(X_grid[:, 0].detach().numpy(), 
    #                 X_grid[:, 1].detach().numpy(),
    #                 mean.detach().numpy() - std.detach().numpy(), cmap='viridis', alpha=0.7)

    # ax.plot_trisurf(X_grid[:, 0].detach().numpy(), 
    #                 X_grid[:, 1].detach().numpy(),
    #                 mean.detach().numpy() + std.detach().numpy(), cmap='viridis', alpha=0.7)
    plt.show()