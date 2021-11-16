import jax
import jax.numpy as np
from jax.experimental import optimizers, stax
from jax.experimental.stax import Dense, Relu, Sigmoid, Selu, Tanh, Softplus, Identity
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib.tri as tri
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


def show_energy(x1, x2, x3, out):  
    fig = plt.figure(figsize=(8, 8))
    ax = plt.axes(projection="3d")
    img = ax.scatter3D(x1, x2, x3, c=out, alpha=0.7, marker='.', cmap=plt.hot())
    fig.colorbar(img)


def load_data():
    file_path = f"data/{args.shape_tag}/numpy/energy_{args.num_samples}_{args.case_id}"
    xy_file = f"{file_path}/data_xy.npy"

    if os.path.isfile(xy_file):
        data_xy = onp.load(xy_file)
    else:
        data_files = glob.glob(f'{file_path}/16*.npy')
        assert len(data_files) > 0, f"No data file found in {file_path}!"
        data_xy = onp.stack([onp.load(f) for f in data_files])
        onp.save(xy_file, data_xy)

    print(f"total number of valid data {len(data_xy)}, should be {args.num_samples}")

    return data_xy[:, 2:]


def inspect_data():
    data = load_data()
    show_energy(data[:, 0], data[:, 1], data[:, 2],  data[:, 3])


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


@jax.jit
def update(params, x, y, opt_state):
    """ Compute the gradient for a batch and update the parameters """
    value, grads = jax.value_and_grad(loss)(params, x, y)
    opt_state = opt_update(0, grads, opt_state)
    return get_params(opt_state), opt_state, value


def loss(params, x, y):
    preds = nn_batch_forward(params, x)
    y = y[:, None]
    assert preds.shape == y.shape, f"preds.shape = {preds.shape}, while y.shape = {y.shape}"
    return np.sum((preds - y)**2)


def shuffle_data(data):
    train_portion = 0.9
    n_samps = len(data)
    n_train = int(train_portion * n_samps)
    inds = list(jax.random.permutation(jax.random.PRNGKey(0), n_samps))
    inds_train = inds[:n_train]
    inds_test = inds[n_train:]
    train_data = data[inds_train]
    test_data = data[inds_test]
    train_loader = DataLoader(EnergyDataset(train_data), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(EnergyDataset(test_data), batch_size=args.batch_size, shuffle=False)
    return train_data, test_data, train_loader, test_loader


opt_init, opt_update, get_params = optimizers.adam(step_size=args.lr)
init_random_params, nn_batch_forward = get_mlp()


def evaluate_test_errors(test_data, preds):
    compare = np.hstack((np.array(test_data[:, -1:]), preds[:, None]))
    absolute_error = np.absolute(compare[:, 0] - compare[:, 1])
    percent_error = np.absolute(absolute_error / compare[:, 0])
    squared_error = (compare[:, 0] - compare[:, 1])**2

    # print(compare)
    # print(f"max absolute error is {np.max(absolute_error):03f}")
    # print(f"mean absolute error is {np.mean(absolute_error):03f}")  
    print(f"max percent error is {100*np.max(percent_error):03f}%")
    print(f"mean percent error is {100*np.mean(percent_error):03f}%")
    print(f"total squared error is {np.sum(squared_error)}")
    # print(f"mean squared error is {np.mean(squared_error)}")


def mlp_surrogate():
    data = load_data()
    train_data, test_data, train_loader, test_loader = shuffle_data(data)    
    output_shape, params = init_random_params(jax.random.PRNGKey(0), (-1, args.input_size))
    opt_state = opt_init(params)
    num_epochs = 1000
    num_train_samples = len(train_data)
    print(f"num_train_samples = {num_train_samples}")
    loss_vals = []
    for epoch in range(num_epochs):
        for batch_idx, (x, y) in enumerate(train_loader):
            params, opt_state, loss = update(params, np.array(x), np.array(y), opt_state)
            loss_vals.append(loss)

        if epoch % 20 == 0:
            print(f"Epoch {epoch} train loss per sample {loss/num_train_samples}")
 
    preds = nn_batch_forward(params, test_data[:, :-1])
    evaluate_test_errors(test_data, preds[:, 0])

    batch_forward = lambda xtest: nn_batch_forward(params, xtest).reshape(-1)
    return  safe_batch_forward(batch_forward)


def sklearn_gpr_surrogate():
    data = load_data()
    train_data, test_data, train_loader, test_loader = shuffle_data(data)
    kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
    gp.fit(train_data[:, :-1], train_data[:, -1])
    preds, sigma = gp.predict(test_data[:, :-1], return_std=True)
    evaluate_test_errors(test_data, preds)
    print(gp.kernel_.get_params())


def jax_gpr_surrogate(train):
    data = load_data()
    train_data, test_data, train_loader, test_loader = shuffle_data(data)
    x = train_data[:, :-1]
    y = train_data[:, -1]
    xtest = test_data[:, :-1]
 
    if train:
        # poreA
        # params = {"amplitude": 0.1,
        #           "lengthscale": 0.1,
        #           "noise": 1e-6}   

        params = {"amplitude": 0.1,
                  "lengthscale": 0.5,
                  "noise": 1e-6} 

        params = train_scipy(params, x, y)
        print(params)
        mu, var = predict(params, x, y, xtest)
        evaluate_test_errors(test_data, mu)
        with open(f"data/pickle/jax_gpr_{args.case_id}.pkl", 'wb') as handle:
            pickle.dump(params, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        # params = {"amplitude": 0.054845882032491876,
        #           "lengthscale": 0.17854716524496028,
        #           "noise": 1e-7}  

        with open(f"data/pickle/jax_gpr_{args.case_id}.pkl", 'rb') as handle:
            params = pickle.load(handle)

        mu, var = predict(params, x, y, xtest)
        evaluate_test_errors(test_data, mu)

    batch_forward = lambda xtest: predict(params, x, y, xtest)[0]

    return  safe_batch_forward(batch_forward)
 

def safe_batch_forward(batch_forward):

    xyz0 = np.array([1/5*np.pi, 1/5*np.pi, 0.08])
    def safe_fn(xtest):
        assert len(xtest.shape) == 2, f"Wrong shape of xtest: {xtest.shape}"
        preds = batch_forward(xtest)
        r_square = np.sum((xtest / xyz0[None, :])**2, axis=-1)
        xtest_norm = xtest / np.sqrt(r_square + 1e-10)[:, None]
        preds_norm = batch_forward(xtest_norm)
        result = np.where(r_square < 1., preds, preds_norm * r_square)

        return result

    return safe_fn


def show_contours():
    batch_forward = jax_gpr_surrogate(True)
    # batch_forward = mlp_surrogate()

    x1_range = np.linspace(-1/5*np.pi, 1/5*np.pi, 100)
    x2_range = np.linspace(-1/5*np.pi, 1/5*np.pi, 100)
    x1, x2 = np.meshgrid(x1_range, x2_range)
    xx = np.vstack([x1.ravel(), x2.ravel()]).T

    # print(batch_forward(np.array([0., 0., 0.])))

    for i, disp in enumerate(np.linspace(-0.08, 0.0, 3)):
        disp = disp * np.ones((len(xx), 1))
        inputs = np.hstack((xx, disp))
        out = batch_forward(inputs)
        plt.figure(num=i, figsize=(8, 8))
        out = out.reshape(x1.shape)
        plt.contourf(x1, x2, out, levels=50, cmap='seismic')
        plt.colorbar()
        contours = np.linspace(0, 0.1, 11)
        plt.contour(x1, x2, out, contours, colors=['black']*len(contours))
        plt.axis('equal')

    # plt.xlim([-4, 4])
    # plt.ylim([-4, 4])


def main():
    args.case_id = 'poreB'
    args.num_samples = 1000
    args.shape_tag = 'beam'
    # jax_gpr_surrogate(train=True)
    # sklearn_gpr_surrogate()
    # mlp_surrogate()
    show_contours()


if __name__ == '__main__':
    # inspect_data()
    main()
    plt.show()
 
