import numpy as onp
import jax.numpy as np
import pickle
import matplotlib.pyplot as plt
from src.arguments import args
from src.fem_commons import *
from src.dns import dns_solve_implicit_euler
from src.beam import fem_solve
from src.trainer import mlp_surrogate, jax_gpr_surrogate, load_data, shuffle_data, evaluate_errors
from src.utils import plot_energy, plot_dynamics
from src.graph_net import simulate, get_cross_energy


def overview():
    '''
    Produce figure 'overview' in the manuscript
    '''
    args.shape_tag = 'dns'
    args.pore_id = 'poreA'
    args.dns_n_rows = 4
    args.dns_n_cols = 4
    args.damping_coeff = 5*1e-3
    args.description = 'demo'
    args.coef = 2
    args.amp = 0.05

    dt = 1e-3
    ts = np.arange(0, 101*dt, dt)
    uv_bottom_x = compute_uv_bc_vals(ts, bc_excitation_impulse, args.dns_n_rows)
    uv_bottom_y = compute_uv_bc_vals(ts, bc_excitation_fixed, args.dns_n_rows)    
    uv_top_x = compute_uv_bc_vals(ts, bc_excitation_fixed, args.dns_n_rows)
    uv_top_y = compute_uv_bc_vals(ts, bc_excitation_impulse, args.dns_n_rows)
    uv_list = [None, uv_bottom_y, None, uv_top_y]
    dns_solve_implicit_euler(*get_shape_params(), uv_list, ts)


def beam():
    '''
    Produce figure 'beam' in the manuscript
    '''
    args.shape_tag = 'beam'
    args.pore_id = 'poreA'
    fem_solve(*get_shape_params(), rot_angle_1=30./180.*onp.pi, rot_angle_2=-30./180.*onp.pi, disp=-0.05, save_data=False, save_sols=True)



def train_val_test():
    '''
    Produce figure 'train_val_test' in the manuscript
    '''
    args.shape_tag = 'beam'

    regressions = [mlp_surrogate, mlp_surrogate,  mlp_surrogate, jax_gpr_surrogate]
    reg_names = ['MLP1', 'MLP2', 'MLP3', 'GPR']
    pore_ids = ['poreA', 'poreB', 'poreC', 'poreD', 'poreE']
    width_hiddens = [32, 64, 128, None]
    n_hiddens = [2, 4, 8, None]
    lrs = [4*1e-4, 2*1e-4, 1*1e-4, None]

    cache = False
    if cache:
        train_errors = np.load(get_file_path('numpy', 'train_errors'))
        validation_errors = np.load(get_file_path('numpy', 'validation_errors'))
        test_errors = np.load(get_file_path('numpy', 'test_errors'))
    else:
        train_errors = []
        validation_errors = []
        test_errors = []

        for i, pore_id in enumerate(pore_ids):    
            args.pore_id = pore_id
            data = load_data()
            train_data, validation_data, test_data, train_loader, validation_loader, test_loader = shuffle_data(data) 
            train_errors.append([])
            validation_errors.append([])
            test_errors.append([])
            for j, regression in enumerate(regressions):
                args.width_hidden = width_hiddens[j]
                args.n_hidden = n_hiddens[j]
                args.lr = lrs[j]
                batch_forward = regression(True, train_data, train_loader)
                train_SMSE, _, _ = evaluate_errors(train_data, train_data, batch_forward)
                validation_SMSE, _, _, _, _ = evaluate_errors(validation_data, train_data, batch_forward)
                test_SMSE, _, _ = evaluate_errors(test_data, train_data, batch_forward)
                train_errors[i].append(train_SMSE)
                validation_errors[i].append(validation_SMSE)            
                test_errors[i].append(test_SMSE)
 
        train_errors = np.array(train_errors)
        validation_errors = np.array(validation_errors)
        test_errors= np.array(test_errors)
        np.save(get_file_path('numpy', 'train_errors'), train_errors)
        np.save(get_file_path('numpy', 'validation_errors'), validation_errors)
        np.save(get_file_path('numpy', 'test_errors'), test_errors)
 
    print(f"train_errors = {train_errors}")
    print(f"validation_errors = {validation_errors}")
    print(f"test_errors = {test_errors}")

    def set_plt():
        plt.xticks(X, shapes)
        plt.ylim((0., 0.0003))
        plt.tick_params(labelsize=18)
        plt.ylabel(f'SMSE', fontsize=20)
        plt.legend(fontsize=18, frameon=False)     

    shapes = ['shape A', 'shape B', 'shape C', 'shape D', 'shape E']

    X = np.arange(len(shapes))
    for i in range(len(regressions)):
        fig, ax = plt.subplots()
        ax.bar(X, train_errors[:, i], color='r', width = 0.22, label='Training')
        ax.bar(X + 0.25, validation_errors[:, i], color='b', width = 0.22, label='Validation')
        set_plt()

        plt.savefig(get_file_path('pdf', ['training', reg_names[i] + '_train_val']), bbox_inches='tight')
    
    fig, ax = plt.subplots()      
    ax.bar(X, test_errors[:, -1], color='g', width = 0.22, label='Test')
    set_plt()
    plt.savefig(get_file_path('pdf', ['training', reg_names[-1] + '_test']), bbox_inches='tight')


def energy_countours():
    '''
    Produce figure 'contour' in the manuscript
    '''
    def show_contours(axes_row):
        rot_bound, disp_bound = np.load(get_file_path('numpy', 'bounds'))
        x1_range = np.linspace(-rot_bound, rot_bound, 100)
        x2_range = np.linspace(-rot_bound, rot_bound, 100)
        x1, x2 = np.meshgrid(x1_range, x2_range)
        xx = np.vstack([x1.ravel(), x2.ravel()]).T
        print(f"zero deformation has enery {batch_forward(np.array([[0., 0., 0.]]))}")
        for i, disp in enumerate(np.linspace(-disp_bound, disp_bound, ncols_plt)):
            ax = axes_row[i]
            disps = disp * np.ones((len(xx), 1))
            inputs = np.hstack((xx, disps))
            out = batch_forward(inputs)
            out = out.reshape(x1.shape)
            print(f"disp = {disp}, min out = {np.min(out)}, max out = {np.max(out)}")
            ax.contourf(x1, x2, out, levels=100, cmap=cmap, vmin=0., vmax=max_energy)
            contours = np.linspace(0, max_energy, 26)
            ax.contour(x1, x2, out, contours, colors=['black']*len(contours), linewidths=0.3, vmin=0., vmax=max_energy)
            ax.set_aspect('equal', 'box')

    args.shape_tag = 'beam'
    pore_ids = ['poreA', 'poreB', 'poreC', 'poreD', 'poreE']

    ncols_plt = 5
    max_energy = 0.25
    cmap = 'seismic'
    fig, axes = plt.subplots(nrows=len(pore_ids), ncols=ncols_plt, sharex=True, sharey=True, figsize=(8, 8))

    for i, pore_id in enumerate(pore_ids):
        args.pore_id = pore_id
        data = load_data()
        train_data, validation_data, test_data, train_loader, validation_loader, test_loader = shuffle_data(data) 
        batch_forward = jax_gpr_surrogate(False, train_data, train_loader)
        show_contours(axes[i])

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.2, 0.03, 0.5])
    m = plt.cm.ScalarMappable(cmap=cmap)
    m.set_clim(0., max_energy)
    fig.colorbar(m, cax=cbar_ax)
    plt.savefig(get_file_path('pdf', ['contour', 'contour']), bbox_inches='tight')



def pred_true_plot():
    '''
    Produce figure 'pred_true' in the manuscript
    '''
    args.shape_tag = 'beam'
    pore_ids = ['poreA', 'poreB', 'poreC', 'poreD', 'poreE']
 
    for i, pore_id in enumerate(pore_ids):
        args.pore_id = pore_id
        data = load_data()
        train_data, validation_data, test_data, train_loader, validation_loader, test_loader = shuffle_data(data) 
        batch_forward = jax_gpr_surrogate(False, train_data, train_loader)
        _, scaled_true_vals, scaled_preds = evaluate_errors(test_data, train_data, batch_forward)

        plt.figure()
        ref_vals = np.linspace(0., 1., 10)
        plt.plot(ref_vals, ref_vals, '--', linewidth=2, color='black')
        plt.scatter(scaled_true_vals, scaled_preds, s=50, c='red')
        plt.xlabel("scaled true energy", fontsize=20)
        plt.ylabel("scaled predicted energy", fontsize=20)
        plt.tick_params(labelsize=18)
        plt.savefig(get_file_path('pdf', ['pred_true', args.pore_id]), bbox_inches='tight')


def pore():
    '''
    Produce figure 'pore' in the manuscript
    '''
    args.shape_tag = 'dns'
    args.dns_n_rows = 2
    args.dns_n_cols = 2
    args.damping_coeff = 0.
    args.description = 'pore'
    args.coef = 2
    dt = 1e-3
    ts = np.arange(0, 2*dt, dt)  
    uv_top_x = compute_uv_bc_vals(ts, bc_excitation_fixed, args.dns_n_rows)
    uv_top_y = compute_uv_bc_vals(ts, bc_excitation_fixed, args.dns_n_rows)
    uv_list = [None, None, uv_top_x, uv_top_y]
    pore_ids = ['poreA', 'poreB', 'poreC', 'poreD', 'poreE']
    for i, pore_id in enumerate(pore_ids):
        args.pore_id = pore_id
        dns_solve_implicit_euler(*get_shape_params(), uv_list, ts)


def show_gpr_params():
    '''
    Print optimized gpr params
    '''
    pore_ids = ['poreA', 'poreB', 'poreC', 'poreD', 'poreE']
    for i, pore_id in enumerate(pore_ids):
        args.pore_id = pore_id   
        pickle_path = get_file_path('pickle', 'gpr')
        with open(pickle_path, 'rb') as handle:
            params = pickle.load(handle)  
        print(params)  


def statics_dns():
    '''
    Produce figure 'statics_poreA' and 'statics_poreE' in the manuscript
    '''
    args.shape_tag = 'dns'
    args.dns_n_rows = 8
    args.dns_n_cols = 8
    args.damping_coeff = 10*1e-3
    args.coef = 1

    dt = 1e-2
    ts = np.arange(0, 101*dt, dt)

    amps = [0.1, -0.1]
    descriptions = ['statics_cmp', 'statics_ten']
    pore_ids = ['poreA', 'poreE']

    for i in range(len(pore_ids)):
        args.pore_id = pore_ids[i]
        for j in range(len(amps)):
            args.amp = amps[j]
            args.description = descriptions[j]

            compute = True
            if compute:
                uv_bottom_x = compute_uv_bc_vals(ts, bc_excitation_fixed, args.dns_n_rows)
                uv_bottom_y = compute_uv_bc_vals(ts, bc_excitation_fixed, args.dns_n_rows)    
                uv_top_x = compute_uv_bc_vals(ts, bc_excitation_fixed, args.dns_n_rows)
                uv_top_y = compute_uv_bc_vals(ts, bc_excitation_impulse, args.dns_n_rows)
                uv_list = [uv_bottom_x, uv_bottom_y, uv_top_x, uv_top_y]
                dns_solve_implicit_euler(*get_shape_params(), uv_list, ts)


def statics_gn():
    '''
    Produce figure 'statics_poreA' and 'statics_poreE' in the manuscript
    '''
    args.gn_n_cols = 8
    args.gn_n_rows = 8
    args.gn_damping = 2*1e-3
    args.coef = 1

    dt = 1e-3
    ts = np.arange(0, 1001*dt, dt)

    amps = [0.1, -0.1]
    descriptions = ['statics_cmp', 'statics_ten']
    pore_ids = ['poreA', 'poreE']

    for i in range(len(pore_ids)):
        args.pore_id = pore_ids[i]
        for j in range(len(amps)):
            args.shape_tag = 'beam'
            args.amp = amps[j]
            args.description = descriptions[j]

            uv_bottom_x = compute_uv_bc_vals(ts, bc_excitation_fixed, args.gn_n_rows)
            uv_bottom_y = compute_uv_bc_vals(ts, bc_excitation_fixed, args.gn_n_rows)    
            uv_top_x = compute_uv_bc_vals(ts, bc_excitation_fixed, args.gn_n_rows)
            uv_top_y = compute_uv_bc_vals(ts, bc_excitation_impulse, args.gn_n_rows)
            uv_list = [uv_bottom_x, uv_bottom_y, uv_top_x, uv_top_y]
            ys, hamitonians, kinetic_energies, graph = simulate(uv_list, ts)
            plot_dynamics(ys[::10], graph, args.gn_n_cols, args.gn_n_rows, [0, 100])


def wave_dns():
    args.shape_tag = 'dns'
    args.description = 'P_wave'
    args.dns_n_cols = 2
    args.dns_n_rows = 8
    args.damping_coeff = 0.
    args.coef = 10
    args.amp = 0.05

    dt = 5*1e-4
    ts = np.arange(0, 101*dt, dt)

    pore_ids = ['poreA', 'poreB', 'poreC', 'poreD', 'poreE']
    for i in range(len(pore_ids)):
        args.pore_id = pore_ids[i]

        uv_top_x = compute_uv_bc_vals(ts, bc_excitation_fixed, args.dns_n_rows)
        uv_top_y = compute_uv_bc_vals(ts, bc_excitation_impulse, args.dns_n_rows)
        uv_list = [None, None, uv_top_x, uv_top_y]
        dns_solve_implicit_euler(*get_shape_params(), uv_list, ts)


def wave_gn():
    args.shape_tag = 'beam'    
    args.description = 'P_wave'
    args.gn_n_cols = 2
    args.gn_n_rows = 8
    args.gn_damping = 0.
    args.coef = 10
    args.amp = 0.05

    dt = 5*1e-5
    ts = np.arange(0, 1001*dt, dt)

    pore_ids = ['poreA', 'poreB', 'poreC', 'poreD', 'poreE']
    for i, pore_id in enumerate(pore_ids):
        args.pore_id = pore_id 

        uv_top_x = compute_uv_bc_vals(ts, bc_excitation_fixed, args.gn_n_rows)
        uv_top_y = compute_uv_bc_vals(ts, bc_excitation_impulse, args.gn_n_rows)
        uv_list = [None, None, uv_top_x, uv_top_y]
        ys, hamitonians, kinetic_energies, graph = simulate(uv_list, ts)
        cross_energies = get_cross_energy(ys, [0, 1])
        energies = onp.stack((ts, hamitonians, kinetic_energies, cross_energies))
        onp.save(get_file_path('numpy', 'energy'), energies)
        save_frames = [0, 20, 40, 60, 100] if pore_id == 'poreA' else None
        plot_dynamics(ys[::10], graph, args.gn_n_cols, args.gn_n_rows, limit_ratio_x=1., save_frames=save_frames)


def plot_wave():
    args.description = 'P_wave'
    args.gn_n_cols = 2
    args.gn_n_rows = 8
    pore_ids = ['poreA', 'poreB', 'poreC', 'poreD', 'poreE']
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    markers = ['o', 's', '^', 'h', 'p']
    shape_tags = ['beam', 'dns']

    for i in range(len(shape_tags)):
        args.shape_tag = shape_tags[i]
        plt.figure()
        for j in range(len(pore_ids)):
            args.pore_id = pore_ids[j]
            ts, _, _, cross_energies =  onp.load(get_file_path('numpy', 'energy'))
            step = (len(ts) - 1) // 50
            plt.plot(ts[::step], cross_energies[::step], marker=markers[j],  markersize=2, linestyle="-", linewidth=1, color=colors[j])

        plt.xlabel("time", fontsize=20)
        plt.ylabel("kinetic energy", fontsize=20)
        plt.tick_params(labelsize=18)
        plt.savefig(get_file_path('pdf', ['disp', args.description]), bbox_inches='tight')


if __name__ == '__main__':
    # overview()
    # beam()
    # train_val_test()
    # energy_countours()
    # pred_true_plot()
    # pore()
    # show_gpr_params()
    # statics_dns()
    # statics_gn()
    # wave_dns()
    # wave_gn()
    plot_wave()
    # plt.show()

