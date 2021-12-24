'''
Instructions
'''

import numpy as onp
import jax.numpy as np
import jax
import fenics as fe
import pickle
import time
import datetime
import matplotlib.pyplot as plt
from src.arguments import args
from src.fem_commons import *
from src.dns import dns_solve_implicit_euler
from src.beam import fem_solve
from src.trainer import mlp_surrogate, jax_gpr_surrogate, load_data, shuffle_data, evaluate_errors
from src.utils import plot_energy, plot_dynamics, plot_disp
from src.graph_net import simulate, build_graph, build_bc_inds, hamiltonian, get_cross_energy, odeint, leapfrog

PLATFORM = jax.lib.xla_bridge.get_backend().platform


def overview():
    '''
    Produce figure 'overview' in the manuscript
    '''
    args.shape_tag = 'dns'
    args.pore_id = 'poreA'
    args.dns_n_rows = 4
    args.dns_n_cols = 4
    args.dns_damping = 5*1e-3
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
    args.dns_damping = 0.
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
    args.dns_damping = 10*1e-3
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
            uv_list = [uv_bottom_x, uv_bottom_y, None, uv_top_x, uv_top_y, None]
            ys, hamitonians, kinetic_energies, graph = simulate(uv_list, ts)
            plot_dynamics(ys[::10], graph, args.gn_n_cols, args.gn_n_rows, pdf_frames=[0, 100])


def wave_dns():
    args.shape_tag = 'dns'
    args.description = 'P_wave'
    args.dns_n_cols = 2
    args.dns_n_rows = 8
    args.dns_damping = 0.
    args.coef = 5
    args.amp = 0.1

    dt = 5*1e-4
    ts = np.arange(0, 101*dt, dt)

    pore_ids = ['poreA', 'poreB', 'poreC', 'poreD', 'poreE']
    # pore_ids = ['poreA']

    for i in range(len(pore_ids)):
        args.pore_id = pore_ids[i]

        uv_top_x = compute_uv_bc_vals(ts, bc_excitation_fixed, args.dns_n_rows)
        uv_top_y = compute_uv_bc_vals(ts, bc_excitation_impulse, args.dns_n_rows)
        uv_list = [None, None, uv_top_x, uv_top_y]
        dns_solve_implicit_euler(*get_shape_params(), uv_list, ts)
        ts, _, _, cross_energies =  onp.load(get_file_path('numpy', 'energy'))
        plot_disp(ts, cross_energies)


def wave_gn():
    args.shape_tag = 'beam'    
    args.description = 'P_wave'
    args.gn_n_cols = 2
    args.gn_n_rows = 8
    args.gn_damping = 0.
    args.coef = 5
    args.amp = 0.1

    dt = 5*1e-5
    ts = np.arange(0, 1001*dt, dt)

    pore_ids = ['poreA', 'poreB', 'poreC', 'poreD', 'poreE']
    # pore_ids = ['poreA']

    for i, pore_id in enumerate(pore_ids):
        args.pore_id = pore_id 

        uv_top_x = compute_uv_bc_vals(ts, bc_excitation_fixed, args.gn_n_rows)
        uv_top_y = compute_uv_bc_vals(ts, bc_excitation_impulse, args.gn_n_rows)
        # qw_top = compute_uv_bc_vals(ts, bc_excitation_fixed, args.gn_n_rows)
        uv_list = [None, None, None, uv_top_x, uv_top_y, None]
        ys, hamitonians, kinetic_energies, graph = simulate(uv_list, ts)
        cross_energies = get_cross_energy(ys, onp.arange(args.gn_n_cols))
        energies = onp.stack((ts, hamitonians, kinetic_energies, cross_energies))
        onp.save(get_file_path('numpy', 'energy'), energies)
        pdf_frames = onp.arange(0, len(ts), 10) if pore_id == 'poreA' else None
        plot_dynamics(ys[::10], graph, args.gn_n_cols, args.gn_n_rows, ((0.5, 0.5), (0.45, 0.1)), pdf_frames=pdf_frames)
        plot_disp(ts, cross_energies)


def plot_wave():
    '''
    Produce figure 'P_wave' in the manuscript
    '''
    args.description = 'P_wave'
    args.dns_n_cols = 2
    args.dns_n_rows = 8
    args.gn_n_cols = 2
    args.gn_n_rows = 8
    pore_ids = ['poreA', 'poreB', 'poreC', 'poreD', 'poreE']
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    markers = ['o', 's', '^', 'h', 'p']
    shape_tags = ['beam', 'dns']
    shapes = ['shape A', 'shape B', 'shape C', 'shape D', 'shape E']

    for i in range(len(shape_tags)):
        args.shape_tag = shape_tags[i]
        plt.figure(figsize=(8, 6))
        for j in range(len(pore_ids)):
            args.pore_id = pore_ids[j]
            ts, _, _, cross_energies =  onp.load(get_file_path('numpy', 'energy'))
            step = (len(ts) - 1) // 50
            truncate = int((len(ts) - 1) * 0.8)
            plt.plot(ts[:truncate:step], cross_energies[:truncate:step], marker=markers[j],  markersize=5, 
                linestyle="-", linewidth=1, color=colors[j], label=shapes[j])

        plt.xlabel("time [s]", fontsize=20)
        plt.ylabel("kinetic energy [MJ]", fontsize=20)
        plt.tick_params(labelsize=18)
        plt.legend(fontsize=18, frameon=False)
        plt.ylim(-0.03, 1.15)
        plt.savefig(get_file_path('pdf', ['disp', args.description]), bbox_inches='tight')


def walltime(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        func(*args, **kwargs)
        end_time = time.time()
        time_elapsed = end_time-start_time
        print(f"Time elapsed {time_elapsed} on platform {PLATFORM}") 
        return time_elapsed
    return wrapper


def evaluate_performance_gn():
    args.shape_tag = 'beam'  
    args.description = 'profile'  
    args.pore_id = 'poreA'
    args.gn_damping = 0.
    args.coef = 1
    args.amp = -0.1

    sizes = onp.array([2, 4, 8, 16, 32, 64, 128, 256])

    def prepare():
        dt = 1e-5
        ts = np.arange(0, 1001*dt, dt)   
        uv_top_x = compute_uv_bc_vals(ts, bc_excitation_fixed, args.gn_n_rows)
        uv_top_y = compute_uv_bc_vals(ts, bc_excitation_impulse, args.gn_n_rows)
        uv_list = [None, None, None, uv_top_x, uv_top_y, None]
        graph, ini_state, inds_lookup = build_graph()
        bc_inds_x_list, bc_inds_y_list = build_bc_inds(inds_lookup)
        compute_kinetic_energy, compute_hamiltonian, state_rhs = hamiltonian(graph)
        bcs = [bc_inds_x_list, bc_inds_y_list, uv_list, ini_state]
        return bcs, state_rhs, ini_state, ts

    all_computing_time = []
    for i in range(len(sizes)):
        args.gn_n_cols = sizes[i]
        args.gn_n_rows = sizes[i]

        bcs, state_rhs, ini_state, ts = prepare()
        # The first time jax_simulation runs, it needs to be jitted. We don't include this jitting time.
        walltime(odeint)(leapfrog, bcs, state_rhs, ini_state, ts)
        num_repeats = 5
        computing_time = []
        for j in range(num_repeats):
            print(f"\nsize {sizes[i]}, repeat {j}")
            time_elapsed = walltime(odeint)(leapfrog, bcs, state_rhs, ini_state, ts)
            computing_time.append(time_elapsed/(len(ts) + 1)*1000) 

        all_computing_time.append(computing_time)

    all_computing_time = np.array(all_computing_time).T
    now = datetime.datetime.now().strftime('%s%f')
    to_save = np.vstack((sizes[None, :], all_computing_time))
    np.save(get_file_path('numpy', [args.description, PLATFORM + '_' + now ]), to_save)
    print(f"wall time measurements: {all_computing_time}")


def evaluate_performance_dns():
    args.shape_tag = 'dns'  
    args.description = 'profile'  
    args.pore_id = 'poreA'
    args.dns_damping = 0.
    args.coef = 1
    args.amp = -0.1

    sizes = onp.array([2, 4, 8])
    resolutions = [20, 23, 20] 

    all_computing_time = []
    for i in range(len(sizes)):
        args.dns_n_cols = sizes[i]
        args.dns_n_rows = sizes[i]
        args.resolution = resolutions[i]
        dt = 1e-4
        ts = onp.arange(0, 101*dt, dt)
        uv_top_x = compute_uv_bc_vals(ts, bc_excitation_fixed, args.dns_n_rows)
        uv_top_y = compute_uv_bc_vals(ts, bc_excitation_impulse, args.dns_n_rows)
        uv_list = [None, None, uv_top_x, uv_top_y]
        num_repeats = 5
        computing_time = []
        for j in range(num_repeats):
            print(f"\nsize {sizes[i]}, repeat {j}")
            fe.set_log_level(50)
            time_elapsed = walltime(dns_solve_implicit_euler)(*get_shape_params(), uv_list, ts, False)
            computing_time.append(time_elapsed/(len(ts) + 1)*1000)

        all_computing_time.append(computing_time)

    all_computing_time = onp.array(all_computing_time).T
    now = datetime.datetime.now().strftime('%s%f')
    to_save = onp.vstack((sizes[None, :], all_computing_time))
    onp.save(get_file_path('numpy', [args.description, PLATFORM + '_' + now ]), to_save)
    print(f"wall time measurements: {all_computing_time}")


def plot_performance():
    '''
    Produce figure 'performance' in the manuscript
    '''

    def decode(performance_data):
        size = performance_data[0]
        cpt_time = performance_data[1:]
        mean = np.mean(cpt_time, axis=0)
        sd = np.std(cpt_time, axis=0)
        se = sd / np.sqrt(cpt_time.shape[0])
        return size, mean, se

    args.description = 'profile'
    args.shape_tag = 'beam'  
    gn_gpu_size, gn_gpu_mean, gn_gpu_se = decode(onp.load(get_file_path('numpy', [args.description, 'gpu_1640189219726683'])))
    gn_cpu_size, gn_cpu_mean, gn_cpu_se = decode(onp.load(get_file_path('numpy', [args.description, 'cpu_1640202074849026'])))

    args.shape_tag = 'dns'  
    dns_cpu_size, dns_cpu_mean, dns_cpu_se = decode(onp.load(get_file_path('numpy', [args.description, 'cpu_1640190501129076'])))

    tick_labels = ['2x2', '4x4', '8x8', '16x16', '32x32', '64x64', '128x128', '256x256']
    plt.figure(figsize=(10, 6))
    ref_vals = np.linspace(0., 1., 10)
    plt.plot(gn_gpu_size, gn_gpu_mean, linestyle='-', marker='o', markersize=10, linewidth=2, color='red', label='JAX (GPU)')
    plt.plot(gn_cpu_size, gn_cpu_mean, linestyle='-', marker='^', markersize=10, linewidth=2, color='orange', label='JAX (CPU)')
    plt.plot(dns_cpu_size, dns_cpu_mean, linestyle='-', marker='s', markersize=10, linewidth=2, color='blue', label='FEniCS (CPU)')
    # plt.errorbar(gn_gpu_size, gn_gpu_mean, gn_gpu_se)
    plt.xscale('log')
    plt.yscale('log')
    ax = plt.gca()
    ax.get_xaxis().set_tick_params(which='minor', size=0)
    plt.xticks(gn_gpu_size, tick_labels)
    plt.xlabel("structure size", fontsize=20)
    plt.ylabel("wall time (per step) [ms]", fontsize=20)
    plt.tick_params(labelsize=18)
    plt.legend(fontsize=18, frameon=False)   
    plt.savefig(get_file_path('pdf', 'performance'), bbox_inches='tight')


def hierarchy():
    args.shape_tag = 'beam'

    args.gn_n_cols = 17
    args.gn_n_rows = 65
    args.gn_damping = 1*1e-4

    args.amp = 0.05

    # 哭脸
    # args.deactivated_nodes = [(5, 13), (5, 14), (5, 15), (5, 16), (5, 17), (5, 18),
    #                      (22, 21), (22, 22), (22, 23),
    #                      (22, 10), (22, 9), (22, 8)]

    # args.deactivated_nodes = [(2, 2), (2, 6), (2, 10), (2, 14)]

    dt = 1e-3
    ts = np.arange(0, 2001*dt, dt)

    args.T = 0.2
    fix_bc = compute_uv_bc_vals(ts, bc_excitation_fixed, args.gn_n_rows) 
    sin_bc = compute_uv_bc_vals(ts, bc_excitation_sin, args.gn_n_rows)
    uv_bottom_x = fix_bc
    uv_bottom_y = fix_bc

    pore_ids = ['poreA', 'poreB', 'poreC', 'poreD', 'poreE']

    # pore_ids = ['poreA']
    modes = ['shear']
    defects = [0, 2]

    for i in range(len(pore_ids)):
        args.pore_id = pore_ids[i]
        for j in range(len(modes)):
            mode = modes[j]
            if mode == 'normal':
                uv_top_x = fix_bc
                uv_top_y = sin_bc
            else:
                uv_top_x = sin_bc
                uv_top_y = fix_bc  
            for k in range(len(defects)):
                args.deactivated_nodes = []
                dft = defects[k]
                args.description = 'hierarchy_holow_' + str(dft) +  '_' + mode
                print(f"pore_id = {args.pore_id}, description = {args.description}")
                if dft != 0:
                    for row in range(args.gn_n_rows):
                        for col in range(args.gn_n_cols):
                            if col % dft == 1 and row % dft == 1:
                                args.deactivated_nodes.append((col, row))

                uv_list = [None, uv_bottom_y, None, uv_top_x, uv_top_y, None]
                # uv_list = [None, None, None, uv_top_x, uv_top_y, None]

                ys, hamitonians, kinetic_energies, graph = simulate(uv_list, ts)
                cross_energies = get_cross_energy(ys, onp.arange(args.gn_n_cols)) 
                energies = onp.stack((ts, hamitonians, kinetic_energies, cross_energies))
                onp.save(get_file_path('numpy', 'energy'), energies)

                ts, hs, ks, cross_energies =  onp.load(get_file_path('numpy', 'energy'))
                plot_energy(ts, hs, ks)
                plot_disp(ts, cross_energies)
                if args.pore_id == 'poreA':
                    pdf_frames = onp.arange(0, len(ts), 10)
                    plot_dynamics(ys[::10], graph, args.gn_n_cols, args.gn_n_rows, limits=((1.5, 1.5), (0.18, 0.18)), pdf_frames=pdf_frames)


def S_wave_bulk():
    args.shape_tag = 'bulk'
    args.description = 'S_wave'
    args.bulk_n_cols = 17
    args.bulk_n_rows = 65
    args.dns_damping = 1e-3  
    args.amp = 0.05

    dt = 1e-2
    ts = np.arange(0, 201*dt, dt)

    args.T = 0.2
    fix_bc = compute_uv_bc_vals(ts, bc_excitation_fixed, args.bulk_n_rows) 
    sin_bc = compute_uv_bc_vals(ts, bc_excitation_sin, args.bulk_n_rows)
    uv_bottom_x = fix_bc
    uv_bottom_y = fix_bc
    uv_top_x = sin_bc
    uv_top_y = fix_bc 

    uv_list = [None, uv_bottom_y, uv_top_x, uv_top_y]
    dns_solve_implicit_euler(None, None, uv_list, ts)

    ts, _, _, cross_energies =  onp.load(get_file_path('numpy', 'energy'))
    plot_disp(ts, cross_energies)


def plot_S_wave():
    '''
    Produce figure 'S_wave' in the manuscript
    '''
    args.shape_tag = 'beam'
    args.gn_n_cols = 17
    args.gn_n_rows = 65
    args.bulk_n_cols = 17
    args.bulk_n_rows = 129
    pore_ids = ['poreA', 'poreB', 'poreC', 'poreD', 'poreE']
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    markers = ['o', 's', '^', 'h', 'p']
    shapes = ['shape A', 'shape B', 'shape C', 'shape D', 'shape E']
    defects = [0, 2]

    wave_travelled_time = []
    def get_criticial_time(ts, cross_energies):
        threshold = 0.05
        for i in range(len(ts)):
            if cross_energies[i] > threshold:
                return ts[i]
        raise ValueError(f"Unexpected return")

    for i in range(len(defects)):
        wave_travelled_time.append([])
        dft = defects[i]
        args.description = 'hierarchy_holow_' + str(dft) +  '_shear'
        plt.figure()
        for j in range(len(pore_ids)):
            args.pore_id = pore_ids[j]
            ts, _, _, cross_energies = onp.load(get_file_path('numpy', 'energy'))
            step = (len(ts) - 1) // 50
            truncate = int((len(ts) - 1) * 0.5) if dft == 0  else int((len(ts) - 1) * 0.65)
            plt.plot(ts[:truncate:step], cross_energies[:truncate:step], marker=markers[j],  markersize=5, 
                linestyle="-", linewidth=1, color=colors[j], label=shapes[j])

            wave_travelled_time[i].append(get_criticial_time(ts, cross_energies))

        plt.xlabel("time [s]", fontsize=20)
        plt.ylabel("kinetic energy [MJ]", fontsize=20)
        plt.tick_params(labelsize=18)
        plt.legend(fontsize=18, frameon=False)
        plt.ylim(-0.1, 2.)
        plt.savefig(get_file_path('pdf', ['disp', args.description]), bbox_inches='tight')

    wave_travelled_dist = (args.gn_n_rows - 1) * args.L0
    wave_travelled_vel = wave_travelled_dist / np.array(wave_travelled_time)


    args.shape_tag = 'bulk'
    args.description = 'S_wave'
    ts, _, _, cross_energies = onp.load(get_file_path('numpy', 'energy'))
    # step = (len(ts) - 1) // 50
    # truncate =  int((len(ts) - 1) * 0.25)
    # plt.figure(figsize=(8, 6))
    # plt.plot(ts[:truncate:step], cross_energies[:truncate:step], marker='o',  markersize=5, 
    #     linestyle="-", linewidth=1, color='black', label='bulk')
    # plt.xlabel("time [s]", fontsize=20)
    # plt.ylabel("kinetic energy [MJ]", fontsize=20)
    # plt.tick_params(labelsize=18)  
    # plt.legend(fontsize=18, frameon=False)     
    # # plt.ylim(-0.03, 1.15)
    # plt.savefig(get_file_path('pdf', 'disp'), bbox_inches='tight')

    bulk_vel = wave_travelled_dist / get_criticial_time(ts, cross_energies)
    print(f"bulk material S wave velocity = {bulk_vel}")

    plt.figure()
    dummy_x = np.arange(len(shapes))
    plt.plot(dummy_x, wave_travelled_vel[0], linestyle='-', marker='o', markersize=10, linewidth=2, color='red', label='dense')
    plt.plot(dummy_x, wave_travelled_vel[1], linestyle='-', marker='s', markersize=10, linewidth=2, color='blue', label='sparse')    
    # plt.plot(dummy_x, np.ones(len(dummy_x))*bulk_vel, linestyle='-', linewidth=2, color='black', label='bulk')
    # plt.ylim(0, 250)
    ax = plt.gca()
    ax.get_xaxis().set_tick_params(which='minor', size=0)
    plt.xticks(dummy_x, shapes)
    plt.ylabel("wave velocity [m/s]", fontsize=20)
    plt.tick_params(labelsize=18)
    plt.legend(fontsize=18, frameon=False)   
    plt.savefig(get_file_path('pdf', 'S_wave'), bbox_inches='tight')


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
    # plot_wave()
    # evaluate_performance_dns()
    # evaluate_performance_gn()
    # plot_performance()
    # hierarchy()
    # S_wave_bulk()
    plot_S_wave()
    plt.show()
