import jraph
import jax
import jax.numpy as np
from src.arguments import args
from src.trainer import jax_gpr_surrogate
from src.utils import plot_dynamics, plot_energy
from functools import partial
from matplotlib import pyplot as plt


def unpack_state(state):
    x = state[..., 0:2]
    q = state[..., 2:3]
    v = state[..., 3:5]
    w = state[..., 5:6]
    return x, q, v, w


@partial(jax.jit, static_argnums=(2,))
def leapfrog(state, t_crt, f, *diff_args):
    y_prev, t_prev = state
    h = t_crt - t_prev

    # x_prev, q_prev are at time step n
    # v_prev, w_prev are at time step n-1/2
    x_prev, q_prev, v_prev, w_prev = unpack_state(y_prev)

    rhs = f(y_prev, t_prev, *diff_args)
    rhs_v = rhs[:, 3:5]
    rhs_w = rhs[:, 5:6]

    # v_crt, w_crt are at time step n+1/2
    v_crt = v_prev + h * rhs_v
    w_crt = w_prev + h * rhs_w

    # x_crt, q_crt are at time step n+1
    x_crt = x_prev + h * v_crt
    q_crt = q_prev + h * w_crt

    y_crt = np.hstack((x_crt, q_crt, v_crt, w_crt))
    return (y_crt, t_crt), y_crt


def activation(alpha):
    coef = 10
    return np.where(alpha < 1./coef, coef*alpha, 1.)


def odeint(stepper, bcs, f, y0, ts, *diff_args):

    def stepper_partial(state, t_crt):
        (y_crt, t_crt), _ = stepper(state, t_crt, f, *diff_args)
        y_crt = jax.ops.index_update(y_crt, bc_inds, bc_vals)
        return (y_crt, t_crt), y_crt

    # _, ys = jax.lax.scan(stepper_partial, (y0, ts[0]), ts[1:])

    ys = []
    state = (y0, ts[0])
    for (i, t_crt) in enumerate(ts[1:]):

        bc_inds, bc_vals = bcs
        bc_vals = (1 - activation(ts[i] / ts[-1]) * 0.1) * bc_vals

        state, y = stepper_partial(state, t_crt)
        if i % 20 == 0:
            print(f"step {i}")
            if not np.all(np.isfinite(y)):
                print(f"Found np.inf or np.nan in y - stop the program")             
                exit()
        ys.append(y)
    ys = np.array(ys)
    return ys


def build_graph():
    L0 = args.L0
    n_row = 16
    n_col = 16
    n_node = np.array([n_row*n_col])
    n_edge = 2 * np.array([(n_row - 1)*n_col + (n_col - 1)*n_row])

    args.n_col = n_col
    args.n_row = n_row

    senders = []
    receivers = []
    crt_state = []
    ref_state = []
    # compression = 0.1
    compression = 0.

    for row in range(n_row):
        for col in range(n_col):
            ref_state.append(np.array([col*L0, row*L0, 0., 0., 0., 0.]))
            crt_state.append(np.array([col*L0, row*L0*(1 - compression), 0., 0., 0., 0.]))
            receiver = row*n_col + col
            local_senders = [(col + 1, row), (col, row + 1), (col - 1, row), (col, row - 1)]
            for local_col, local_row in local_senders:
                if  local_col >= 0 and local_col <= n_col - 1 and local_row >= 0 and local_row <= n_row - 1:
                    senders.append(local_row*n_col + local_col)
                    receivers.append(receiver)
                    
    assert len(senders) == n_edge[0], f"Building edges wrong!"
    print(f"Total number nodes = {n_node[0]}, total number of edges = {n_edge[0]}")
    crt_state = np.stack(crt_state)
    ref_state = np.stack(ref_state)
    ini_state = np.array(crt_state)
    senders = np.array(senders)
    receivers = np.array(receivers)

    mass = np.load(f"data/numpy/unit/mass_{args.case_id}.npy") * np.ones(n_node)
    inertia = np.load(f"data/numpy/unit/inertia_{args.case_id}.npy") * np.ones(n_node)

    node_features = {"ref_state": ref_state, "crt_state": crt_state, "mass": mass, "inertia": inertia}
    graph = jraph.GraphsTuple(nodes=node_features, edges={}, senders=senders, receivers=receivers,
        n_node=n_node, n_edge=n_edge, globals={})

    # bc_nodes = [0, 1, 2, 3, 12, 13, 14, 15] + 
    bc_nodes = [i for i in range(n_col)] + [int(n_node - i - 1) for i in range(n_col)]
    bc_inds = jax.ops.index[bc_nodes*2, [1]*len(bc_nodes) + [4]*len(bc_nodes)]
    bc_vals = ini_state[bc_inds]
    bcs = (bc_inds, bc_vals)

    return graph, ini_state, bcs


def update_graph():
    batch_forward = jax_gpr_surrogate(False)

    def update_edge_fn(edges, senders, receivers, globals_):
        sender_ref_x, sender_ref_q, _, _ = unpack_state(senders["ref_state"])
        sender_crt_x, sender_crt_q, _, _ = unpack_state(senders["crt_state"])
        receiver_ref_x, receiver_ref_q, _, _ = unpack_state(receivers["ref_state"])
        receiver_crt_x, receiver_crt_q, _, _ = unpack_state(receivers["crt_state"])

        ref_direction = receiver_ref_x - sender_ref_x
        crt_direction = receiver_crt_x - sender_crt_x
 
        ref_dist = np.linalg.norm(ref_direction, axis=-1)
        crt_dist = np.linalg.norm(crt_direction, axis=-1)

        edge_q = np.arcsin(np.cross(ref_direction, crt_direction) / (ref_dist * crt_dist))
        sender_delta_q = sender_crt_q.reshape(-1) - sender_ref_q.reshape(-1) - edge_q
        receiver_delta_q = receiver_crt_q.reshape(-1) - receiver_ref_q.reshape(-1) - edge_q
        delta_dist = crt_dist - ref_dist
        inputs = np.stack((sender_delta_q, receiver_delta_q, delta_dist)).T

        potential_energy = 0.5 * batch_forward(inputs)

        return {"potential_energy": potential_energy}

    def update_node_fn(nodes, sent_edges, received_edges, globals_):
        #TODO: delete unused variables
        _, _, crt_v, crt_w = unpack_state(nodes["crt_state"])
        kinetic_energy_per_node = 0.5 * nodes["mass"] * np.sum(crt_v**2, -1) + 0.5 * nodes["inertia"] * np.sum(crt_w**2, -1)

        return {"kinetic_energy": kinetic_energy_per_node}

    def update_global_fn(nodes, edges, globals_):
        hamiltonian = nodes["kinetic_energy"] + edges["potential_energy"]
        return {"hamiltonian": hamiltonian}

    net_fn = jraph.GraphNetwork(update_edge_fn=update_edge_fn,
                                update_node_fn=update_node_fn,
                                update_global_fn=update_global_fn)

    return net_fn


def hamiltonian(graph):
    net_fn = update_graph()

    def compute_hamiltonian(y):
        graph.nodes["crt_state"] = y
        new_graph = net_fn(graph)
        return new_graph.globals["hamiltonian"][0]

    grad_hamiltonian = jax.grad(compute_hamiltonian)

    def state_rhs(y, t, *diff_args):
        x, q, v, w = unpack_state(y)
        damping = 1e-1
        grads = grad_hamiltonian(y)
        mass = graph.nodes["mass"]
        inertia = graph.nodes["inertia"]

        v_rhs = (-damping * v - grads[:, :2]) / mass[:, None]
        w_rhs = (-damping * w - grads[:, 2:3]) / inertia[:, None]

        rhs = np.hstack((np.zeros(grads[:, 3:].shape), v_rhs, w_rhs))
        return rhs

    # nodes, edges, receivers, senders, globals_, n_node, n_edge = new_graph
    # print(new_graph)
    # print(compute_hamiltonian(crt_state))

    return compute_hamiltonian, state_rhs


def main():
    args.case_id = 'poreA'
    args.num_samples = 1000

    graph, ini_state, bcs = build_graph()
    compute_hamiltonian, state_rhs = hamiltonian(graph)
    vmap_hamitonian = jax.jit(jax.vmap(compute_hamiltonian))

    dt = 1e-3
    # ts = np.arange(0, 20001*dt, dt)
    ts = np.arange(0, 10001*dt, dt)
    ys = odeint(leapfrog, bcs, state_rhs, ini_state, ts)

    # print(ys[0])
    # print(ys[-2:])

    hs = vmap_hamitonian(ys[::10])

    print(hs.shape)
    plot_energy(hs, f"data/pdf/hs_{args.case_id}.pdf")
    plot_dynamics(ys[::10], graph)


if __name__ == '__main__':
    main()
    # plt.show()
