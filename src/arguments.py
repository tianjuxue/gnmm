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
parser.add_argument('--activation', choices=['tanh', 'selu', 'relu', 'sigmoid', 'softplus'], default='relu')
parser.add_argument('--width_hidden', type=int, default=128)
parser.add_argument('--n_hidden', type=int, default=5)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--input_size', type=int, default=3)

# parser.add_argument('--verbose', help='Verbose for debug', action='store_true', default=True)
# parser.add_argument('--gravity', type=float, default=9.8)
# parser.add_argument('--dir', type=str, default='data')
# parser.add_argument('--dim', type=int, default=3)
args = parser.parse_args()


# Latex style plot
# plt.rcParams.update({
#     "text.latex.preamble": r"\usepackage{amsmath}",
#     "text.usetex": True,
#     "font.family": "sans-serif",
#     "font.sans-serif": ["Helvetica"]})

