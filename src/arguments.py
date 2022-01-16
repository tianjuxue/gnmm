import numpy as onp
import jax
import jax.numpy as np
import argparse
import sys
import numpy as onp
import matplotlib.pyplot as plt
from jax.config import config
import torch

torch.manual_seed(0)

# config.update("jax_enable_x64", True)


# Set numpy printing format
onp.set_printoptions(threshold=sys.maxsize, linewidth=1000, suppress=True)
onp.set_printoptions(precision=10)
onp.random.seed(0)


# np.set_printoptions(threshold=sys.maxsize, linewidth=1000, suppress=True)
# np.set_printoptions(precision=5)

# Manage arguments
parser = argparse.ArgumentParser()
parser.add_argument('--porosity', type=float, default=0.6)
parser.add_argument('--L0', type=float, default=0.5)
parser.add_argument('--young_modulus', type=float, default=100.)
parser.add_argument('--poisson_ratio', type=float, default=0.3)
parser.add_argument('--density', type=float, default=1e-3)
parser.add_argument('--overwrite_mesh', default=False)
parser.add_argument('--resolution', type=int, default=20)
parser.add_argument('--dns_dynamics', default=True)

parser.add_argument('--activation', choices=['tanh', 'selu', 'relu', 'sigmoid', 'softplus'], default='relu')
parser.add_argument('--width_hidden', type=int, default=128)
parser.add_argument('--n_hidden', type=int, default=4)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--input_size', type=int, default=3)
parser.add_argument('--num_samples', type=int, default=1000)

parser.add_argument('--deactivated_nodes', type=list, default=[])

args = parser.parse_args()


# Latex style plot
plt.rcParams.update({
    "text.latex.preamble": r"\usepackage{amsmath}",
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})


# params = {'legend.fontsize': 'x-large',
#           'figure.figsize': (15, 5),
#          'axes.labelsize': 'x-large',
#          'axes.titlesize':'x-large',
#          'xtick.labelsize':'x-large',
#          'ytick.labelsize':'x-large'}
# pylab.rcParams.update(params)