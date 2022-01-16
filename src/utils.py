import numpy as onp
import jax
import jax.numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from src.arguments import args
from celluloid import Camera
from matplotlib import collections  as mc
import time
import meshio
import shutil
import os
from src.fem_commons import *


def angle_to_rot_mat(theta):
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])


def rotate_vector(theta, vector):
    rot_mat = angle_to_rot_mat(theta)
    return np.dot(rot_mat, vector)


def get_edge_line(sender_ref_x, receiver_ref_x, sender_crt_x, receiver_crt_x, sender_q, receiver_q):
    sender_u = rotate_vector(sender_q, receiver_ref_x - sender_ref_x)
    receiver_u = rotate_vector(receiver_q, receiver_ref_x - sender_ref_x)

    crt_arm = receiver_crt_x - sender_crt_x
    crt_arm_norm = np.linalg.norm(crt_arm**2)

    theta = np.arctan2(crt_arm[1], crt_arm[0])
    A = np.array([[-2*np.pi*np.sin(theta)*sender_u[1] - 2*np.pi*np.cos(theta)*sender_u[0], 
                   -np.pi*np.sin(theta)*sender_u[1] - np.pi*np.cos(theta)*sender_u[0]],
                  [-2*np.pi*np.sin(theta)*receiver_u[1] - 2*np.pi*np.cos(theta)*receiver_u[0], 
                    np.pi*np.sin(theta)*receiver_u[1] + np.pi*np.cos(theta)*receiver_u[0]]])

    rhs = np.array([-crt_arm[0]*sender_u[1] + crt_arm[1]*sender_u[0], -crt_arm[0]*receiver_u[1] + crt_arm[1]*receiver_u[0]])
    c, d = np.linalg.solve(A, rhs)

    num_lines_per_edge = 4
    t = np.linspace(0., 1., num_lines_per_edge + 1)

    lines_x0 = sender_crt_x[0] + crt_arm[0]*t - np.sin(theta)*(c*np.sin(2*np.pi*t) + d*np.sin(np.pi*t)) 
    lines_x1 = sender_crt_x[1] + crt_arm[1]*t + np.cos(theta)*(c*np.sin(2*np.pi*t) + d*np.sin(np.pi*t)) 

    lines_x = np.stack((lines_x0, lines_x1)).T
    lines_start = lines_x[:-1]
    lines_end = lines_x[1:]
    return lines_start, lines_end


get_edge_lines = jax.jit(jax.vmap(get_edge_line, in_axes=(0, 0, 0, 0, 0, 0), out_axes=(0, 0)))


def get_cross_line(crt_x, q, L0):
    bar1 = np.array([0.38*L0, 0.])
    bar2 = np.array([0., 0.38*L0])
    start1 = crt_x - rotate_vector(q, bar1)
    start2 = crt_x - rotate_vector(q, bar2)
    end1 = crt_x + rotate_vector(q, bar1)
    end2 = crt_x + rotate_vector(q, bar2)
    return np.stack((start1, start2)), np.stack((end1, end2))

get_cross_lines = jax.jit(jax.vmap(get_cross_line, in_axes=(0, 0, None), out_axes=(0, 0)))


def get_unique(senders, receivers):
    '''
    For plotting purposes, we remove the duplicated pairs between
    senders and receivers.
    '''
    senders = list(senders)
    receivers = list(receivers)
    for i in range(len(senders)):
        if senders[i] > receivers[i]:
            tmp = senders[i]
            senders[i] = receivers[i]
            receivers[i] = tmp

    combined = onp.array([senders, receivers]).T 
    combined_unique = onp.unique(combined, axis=0)
    senders_unique, receivers_unique = combined_unique.T
    return np.array(senders_unique), np.array(receivers_unique)


def process_lines(starts, ends):
    starts, ends = starts.reshape(-1, 2), ends.reshape(-1, 2)
    lines = np.transpose(np.stack((starts, ends)), axes=(1, 0, 2))
    return lines


def ax_add_helper(ax, edge_lines, cross_lines):
    edge_lc = mc.LineCollection(edge_lines, colors='black',  linewidths=1)
    ax.add_collection(edge_lc)
    cross_lc = mc.LineCollection(cross_lines, colors='grey',  linewidths=3)
    ax.add_collection(cross_lc)


def ax_set_limits(ax, xlim, ylim):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')


def plot_dynamics(ys, graph, gn_n_cols, gn_n_rows, limits=((0.24, 0.24), (0.24, 0.24)), pdf_frames=None):
    '''
    Save configurations of cross-spring system to local png files and convert them to an mp4 file.
    '''
    L0 = args.L0

    xlim = (-limits[0][0]*(gn_n_cols - 1)*L0, (1 + limits[0][1])*(gn_n_cols - 1)*L0)
    ylim = (-limits[1][0]*(gn_n_rows - 1)*L0, (1 + limits[1][1])*(gn_n_rows - 1)*L0)

    senders, receivers = get_unique(graph.senders, graph.receivers)
    sender_ref_xs = graph.nodes['ref_state'][senders][:, :2]
    receiver_ref_xs = graph.nodes['ref_state'][receivers][:, :2]

    tmp_root_path = f'data/png/tmp/'
    shutil.rmtree(tmp_root_path, ignore_errors=True)
    os.mkdir(tmp_root_path)

    for i in range(len(ys)):
        if i % 20 == 0:
            print(f"i = {i}")

        y = ys[i]
        sender_crt_xs = y[senders][:, :2]
        receiver_crt_xs = y[receivers][:, :2]
        sender_qs = y[senders][:, 2]
        receiver_qs = y[receivers][:, 2]
        edge_starts, edge_ends = get_edge_lines(sender_ref_xs, receiver_ref_xs, sender_crt_xs, receiver_crt_xs, sender_qs, receiver_qs)
        edge_lines = process_lines(edge_starts, edge_ends)
        crt_xs = y[:, :2]
        qs = y[:, 2]
        cross_starts, cross_ends = get_cross_lines(crt_xs, qs, L0)
        cross_lines = process_lines(cross_starts, cross_ends)

        fig, ax = plt.subplots(figsize=(xlim[1]-xlim[0], ylim[1]-ylim[0]))
        ax_set_limits(ax, xlim, ylim)
        ax_add_helper(ax, edge_lines, cross_lines)    
        fig.savefig(tmp_root_path + f'{i:05}.png', bbox_inches='tight')

        if pdf_frames is not None:
            if i in pdf_frames:
                fig.savefig(get_file_path('pdf', ['graph', f"{args.description}_{args.pore_id}_{i:03}"]), bbox_inches='tight')

        plt.close(fig)

    # The command -pix_fmt yuv420p is to ensure preview of video on Mac OS is enabled
    # https://apple.stackexchange.com/questions/166553/why-wont-video-from-ffmpeg-show-in-quicktime-imovie-or-quick-preview
    # The command -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" is to solve the following "not-divisible-by-2" problem
    # https://stackoverflow.com/questions/20847674/ffmpeg-libx264-height-not-divisible-by-2
    # -y means always overwrite
    os.system('ffmpeg -y -framerate 25 -i data/png/tmp/%05d.png -pix_fmt yuv420p -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" ' + get_file_path('mp4'))


def plot_energy(ts, hamiltonians, kinetic_energies, inverval=50):
    # plt.figure(num=10, figsize=(6, 6))
    assert len(hamiltonians) == len(kinetic_energies) and len(ts) == len(hamiltonians), f"{len(hamiltonians)}, {len(kinetic_energies)} {len(ts)}"
    potential_energies = hamiltonians - kinetic_energies
    step = (len(ts) - 1) // inverval
    plt.figure()
    plt.plot(ts[::step], hamiltonians[::step], marker='o',  markersize=4, linestyle="-", linewidth=1, color='black', label='total')
    plt.plot(ts[::step], kinetic_energies[::step], marker='o',  markersize=4, linestyle="-", linewidth=1, color='red', label='kinetic')
    plt.plot(ts[::step], potential_energies[::step], marker='o',  markersize=4, linestyle="-", linewidth=1, color='blue', label='potential')    
    plt.xlabel("time [s]", fontsize=20)
    plt.ylabel("energy [MJ]", fontsize=20)
    plt.tick_params(labelsize=18)
    plt.legend(fontsize=18, frameon=False)
    plt.savefig(get_file_path('pdf', 'energy'), bbox_inches='tight')


def plot_disp(ts, ks, inverval=50):
    assert len(ts) == len(ks), f"{len(ts), {len(ks)}}"
    step = (len(ts) - 1) // inverval
    plt.figure()
    plt.plot(ts[::step], ks[::step], marker='o',  markersize=2, linestyle="-", linewidth=1, color='black')
    plt.xlabel("time [s]", fontsize=20)
    plt.ylabel("kinetic energy [MJ]", fontsize=20)
    plt.tick_params(labelsize=18)
    plt.savefig(get_file_path('pdf', 'disp'), bbox_inches='tight')


def plot_force(ts, hamiltonians, kinetic_energies, inverval=50):
    '''
    This is a buggy version that doesn't consider dissipation - to be fixed
    '''
    step = (len(ts) - 1) // inverval
    ts = ts[::step]
    hamiltonians = hamiltonians[::step]
    kinetic_energies = kinetic_energies[::step]
    forces = np.hstack((0., np.diff(hamiltonians - kinetic_energies)))
    plt.figure()
    plt.plot(ts, forces, marker='o',  markersize=2, linestyle="-", linewidth=1, color='black')
    plt.xlabel("TBD")
    plt.ylabel("force")
    plt.tick_params(labelsize=14)
    # plt.savefig(file_path)


def walltime(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        func(*args, **kwargs)
        end_time = time.time()
        print(f"Time elapsed {end_time-start_time}") 
    return wrapper



###########################################################################
# output to stl files for 3D printing

def output_stl():
    mesh = meshio.read(f'data/pvd/meshes/dns/poreA_mesh000000.vtu')
    mesh.write(f'data/stl/dns.stl')


if __name__ == '__main__':
    output_stl()
