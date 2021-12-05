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
# plt.style.use('seaborn-pastel')


def angle_to_rot_mat(theta):
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])


def rotate_vector(theta, vector):
    rot_mat = angle_to_rot_mat(theta)
    return np.dot(rot_mat, vector)

rotate_vector_thetas = jax.vmap(jax.vmap(jax.vmap(rotate_vector, in_axes=(0, None)), in_axes=(0, None)), in_axes=(None, 0))


def get_line(sender_ref_x, receiver_ref_x, sender_crt_x, receiver_crt_x, sender_q, receiver_q):
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


get_lines = jax.jit(jax.vmap(get_line, in_axes=(0, 0, 0, 0, 0, 0), out_axes=(0, 0)))


def plot_dynamics(ys, graph):
    L0 = args.L0
 
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)

    plt.xlim(-L0, args.n_col * L0)
    plt.ylim(-L0, args.n_row * L0)
    ax.set_aspect('equal', adjustable='box')

    camera = Camera(fig)

    sender_ref_xs = graph.nodes['ref_state'][graph.senders][:, :2]
    receiver_ref_xs = graph.nodes['ref_state'][graph.receivers][:, :2]

    for i in range(len(ys)):
        if i % 20 == 0:
            print(f"i = {i}")

        y = ys[i]

        sender_crt_xs = y[graph.senders][:, :2]
        receiver_crt_xs = y[graph.receivers][:, :2]
        sender_qs = y[graph.senders][:, 2]
        receiver_qs = y[graph.receivers][:, 2]

        starts, ends = get_lines(sender_ref_xs, receiver_ref_xs, sender_crt_xs, receiver_crt_xs, sender_qs, receiver_qs)

        # print(sender_ref_xs[0])
        # print(receiver_ref_xs[0])
        # print(starts.shape)
        # print(starts[0])
        # print(ends[0])
        # exit()

        starts = starts.reshape(-1, 2)
        ends = ends.reshape(-1, 2)
        lines = np.transpose(np.stack((starts, ends)), axes=(1, 0, 2))
        lc = mc.LineCollection(lines, colors='black',  linewidths=2)
        ax.add_collection(lc)

        camera.snap()

    anim = camera.animate(interval=20)

    anim.save(f'data/mp4/test_{args.case_id}.mp4', writer='ffmpeg', dpi=300)


def plot_energy(energy, file_path):
    plt.figure(num=10, figsize=(6, 6))
    plt.plot(np.arange(1, len(energy) + 1, 1), energy, marker='o',  markersize=2, linestyle="-", linewidth=1, color='blue')
    plt.xlabel("Time steps")
    plt.ylabel("Energy")
    plt.savefig(file_path)


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
    # plot_dynamics(None)
    output_stl()
