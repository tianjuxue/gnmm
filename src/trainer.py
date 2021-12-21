import jax
import jax.numpy as np
from jax.experimental import optimizers, stax
from jax.experimental.stax import Dense, Relu, Sigmoid, Selu, Tanh, Softplus, Identity
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as onp
import shutil
import os
import glob
from src.arguments import args
import datetime
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from src.gpr import train_scipy, marginal_likelihood, predict
import pickle
from functools import partial
from src.fem_commons import *


# def show_energy(x1, x2, x3, out):  
#     fig = plt.figure(figsize=(8, 8))
#     ax = plt.axes(projection="3d")
#     img = ax.scatter3D(x1, x2, x3, c=out, alpha=0.7, marker='.', cmap=plt.hot())
#     fig.colorbar(img)


# def inspect_data():
#     data = load_data()
#     show_energy(data[:, 0], data[:, 1], data[:, 2],  data[:, 3])


def load_data():
    file_path = get_file_path('numpy', 'data')
    xy_file = f"{file_path}/data_xy.npy"

    if os.path.isfile(xy_file):
        data_xy = onp.load(xy_file)
    else:
        data_files = glob.glob(f'{file_path}/16*.npy')
        assert len(data_files) > 0, f"No data file found in {file_path}!"
        data_xy = onp.stack([onp.load(f) for f in data_files])
        onp.save(xy_file, data_xy)

    assert len(data_xy) == args.num_samples, f"total number of valid data {len(data_xy)}, should be {args.num_samples}"
    return data_xy[:, 2:]


class EnergyDataset(Dataset):
    def __init__(self, data):
        self.data = data
        
    def __getitem__(self, index):
        return (self.data[index, :-1], self.data[index, -1])

    def __len__(self):
        return len(self.data)


def get_mlp():
    if args.activation == 'selu':
        act_fun = Selu
    elif args.activation == 'tanh':
        act_fun = Tanh
    elif args.activation == 'relu':
        act_fun = Relu
    elif args.activation == 'sigmoid':
        act_fun = Sigmoid
    elif args.activation == 'softplus':
        act_fun = Softplus
    else:
        raise ValueError(f"Invalid activation function {args.activation}.")

    layers_hidden = []
    for _ in range(args.n_hidden):
        layers_hidden.extend([Dense(args.width_hidden), act_fun])

    layers_hidden.append(Dense(1))
    mlp = stax.serial(*layers_hidden)
    return mlp


def shuffle_data(data):
    train_validation_cut = 0.8
    validation_test_cut = 0.9
    n_samps = len(data)
    n_train_validation = int(train_validation_cut * n_samps)
    n_validation_test = int(validation_test_cut * n_samps)
    inds = list(jax.random.permutation(jax.random.PRNGKey(0), n_samps))
    inds_train = inds[:n_train_validation]
    inds_validation = inds[n_train_validation:n_validation_test]
    inds_test = inds[n_validation_test:]
    train_data = data[inds_train]
    validation_data = data[inds_validation]
    test_data = data[inds_test]
    train_loader = DataLoader(EnergyDataset(train_data), batch_size=args.batch_size, shuffle=False)
    validation_loader = DataLoader(EnergyDataset(validation_data), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(EnergyDataset(test_data), batch_size=args.batch_size, shuffle=False)
    return train_data, validation_data, test_data, train_loader, validation_loader, test_loader



def min_max_scale(arr1, train_y):
    return (arr1 - np.min(train_y)) / (np.max(train_y) - np.min(train_y))


def evaluate_errors(partial_data, train_data, batch_forward):
    x = partial_data[:, :-1]
    true_vals = partial_data[:, -1]
    train_y = train_data[:, -1]
    preds = batch_forward(x).reshape(-1)
    scaled_true_vals = min_max_scale(true_vals, train_y)
    scaled_preds = min_max_scale(preds, train_y)
    compare = np.stack((scaled_true_vals, scaled_preds)).T
    absolute_error = np.absolute(compare[:, 0] - compare[:, 1])
    percent_error = np.absolute(absolute_error / compare[:, 0])
    scaled_MSE = np.sum((compare[:, 0] - compare[:, 1])**2) / len(compare)

    # print(compare)
    print(f"max percent error is {100*np.max(percent_error):03f}%")
    print(f"median percent error is {100*np.median(percent_error):03f}%")
    print(f"scaled MSE = {scaled_MSE}")

    return scaled_MSE, scaled_true_vals, scaled_preds

 
def mlp_surrogate(train, train_data, train_loader, validation_data=None): 
    opt_init, opt_update, get_params = optimizers.adam(step_size=args.lr)
    init_random_params, nn_batch_forward = get_mlp()
    output_shape, params = init_random_params(jax.random.PRNGKey(0), (-1, args.input_size))
    opt_state = opt_init(params)
    num_epochs = 1000
    pickle_path = get_file_path('pickle', 'mlp')
    batch_forward = lambda x_new: nn_batch_forward(params, x_new).reshape(-1)

    def loss_fn(params, x, y):
        preds = nn_batch_forward(params, x)
        y = y[:, None]
        assert preds.shape == y.shape, f"preds.shape = {preds.shape}, while y.shape = {y.shape}"
        return np.sum((preds - y)**2)

    @jax.jit
    def update(params, x, y, opt_state):
        """ Compute the gradient for a batch and update the parameters """
        value, grads = jax.value_and_grad(loss_fn)(params, x, y)
        opt_state = opt_update(0, grads, opt_state)
        return get_params(opt_state), opt_state, value

    if train:
        for epoch in range(num_epochs):
            # training_loss = 0.
            # validatin_loss = 0.
            for batch_idx, (x, y) in enumerate(train_loader):
                params, opt_state, loss = update(params, np.array(x), np.array(y), opt_state)
                # training_loss = training_loss + loss

            if epoch % 100 == 0:
                training_smse, _, _ = evaluate_errors(train_data, train_data, batch_forward)
                if validation_data is not None:
                    validatin_smse, _, _ = evaluate_errors(validation_data, train_data, batch_forward)
                    print(f"Epoch {epoch} training_smse = {training_smse}, Epoch {epoch} validatin_smse = {validatin_smse}")
                else:
                    print(f"Epoch {epoch} training_smse = {training_smse}")                    
 
        with open(pickle_path, 'wb') as handle:
            pickle.dump(params, handle, protocol=pickle.HIGHEST_PROTOCOL)        
    else:
        with open(pickle_path, 'rb') as handle:
            params = pickle.load(handle)        

    return  batch_forward


def jax_gpr_surrogate(train, train_data, train_loader):
    x = train_data[:, :-1]
    y = train_data[:, -1]
    pickle_path = get_file_path('pickle', 'gpr')

    if train:
        params = {"amplitude": 0.1,
                  "lengthscale": 1.,
                  "noise": 5*1e-5} 

        bounds = ((0.05, 5), (0.05, 5), (5*1e-5, 5*1e-4))
        params = train_scipy(params, x, y, bounds)
        print(params)

        with open(pickle_path, 'wb') as handle:
            pickle.dump(params, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(pickle_path, 'rb') as handle:
            params = pickle.load(handle)

    batch_forward = lambda x_new: predict(params, x, y, x_new)[0]

    return batch_forward
 

def safe_batch_forward(batch_forward):
    rot_bound, disp_bound = np.load(get_file_path('numpy', 'bounds'))
    xyz0 = np.array([rot_bound, rot_bound, disp_bound])
    zero_baseline = batch_forward(np.array([[0., 0., 0.]])).reshape(-1)

    print(f"zero_baseline = {zero_baseline}")

    def safe_fn(xtest):
        assert len(xtest.shape) == 2, f"Wrong shape of xtest: {xtest.shape}"
        preds = batch_forward(xtest)
        r_square = np.sum((xtest / xyz0[None, :])**2, axis=-1)
        xtest_norm = xtest / np.sqrt(r_square + 1e-10)[:, None]
        preds_norm = batch_forward(xtest_norm)
        result = np.where(r_square < 1., preds, preds_norm * r_square)
        # result = result - zero_baseline
        return result

    return safe_fn


def regression(surrogate_fn, train):
    data = load_data()
    train_data, validation_data, test_data, train_loader, validation_loader, test_loader = shuffle_data(data) 
    batch_forward = surrogate_fn(train, train_data, train_loader)
    return safe_batch_forward(batch_forward)


def main():
    args.pore_id = 'poreA'
    args.num_samples = 1000
    args.shape_tag = 'beam'
    data = load_data()
    train_data, validation_data, test_data, train_loader, validation_loader, test_loader = shuffle_data(data) 

    args.width_hidden = 64
    args.n_hidden = 2

    # batch_forward = mlp_surrogate(True, train_data, train_loader, validation_data)
    batch_forward = jax_gpr_surrogate(True, train_data, train_loader)
    evaluate_errors(validation_data, train_data, batch_forward)

    # show_contours(batch_forward)


if __name__ == '__main__':
    # inspect_data()
    main()
    # plt.show()
 