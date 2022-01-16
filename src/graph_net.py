import jraph
import jax
import jax.numpy as np
from src.arguments import args
from src.trainer import jax_gpr_surrogate, regression
from src.utils import plot_dynamics, plot_energy, plot_disp
from functools import partial
from matplotlib import pyplot as plt
from src.fem_commons import *

INVALID = -1

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


def get_bcs(time_index, bcs):
    bc_inds_x_list, bc_inds_y_list, uv_list, ini_state = bcs
    bc_inds_x = []
    bc_inds_y = []
    bc_vals = []
    for bc_index in range(len(uv_list)):
        if uv_list[bc_index] is not None:
            disps, vels = uv_list[bc_index]
            bc_val_disp = disps[time_index]
            bc_val_vel = vels[time_index]
            bc_inds_x = bc_inds_x + bc_inds_x_list[bc_index]
            bc_inds_y = bc_inds_y + bc_inds_y_list[bc_index]
            val_len = len(bc_inds_x_list[bc_index])//2
            bc_vals = bc_vals + [float(bc_val_disp)]*val_len + [float(bc_val_vel)]*val_len

    bc_inds = jax.ops.index[bc_inds_x, bc_inds_y]
    bc_vals = ini_state[bc_inds] + np.array(bc_vals)

    return bc_inds, bc_vals


def build_bc_inds(inds_lookup):
    '''
    Four bcs are considered:
    x control on bottom edge: u and v
    y control on bottom edge: u and v
    x control on top edge: u and v
    y control on top edge: u and v
    '''    
    bc_inds_x_list = [[i for i in range(args.gn_n_cols)]*2]*3 + \
                     [[int(args.gn_n_rows*args.gn_n_cols - i - 1) for i in range(args.gn_n_cols)]*2]*3

    bc_inds_x_list = [[inds_lookup[i] for i in bc_inds_x if inds_lookup[i] != INVALID] for bc_inds_x in bc_inds_x_list]

    shapes = [len(bc_inds_x) // 2 for bc_inds_x in bc_inds_x_list]

    bc_inds_y_list = [[0]*shapes[0] + [3]*shapes[0], 
                      [1]*shapes[1] + [4]*shapes[1],
                      [2]*shapes[2] + [5]*shapes[2],
                      [0]*shapes[3] + [3]*shapes[3], 
                      [1]*shapes[4] + [4]*shapes[4],
                      [2]*shapes[5] + [5]*shapes[5]] 

    return bc_inds_x_list, bc_inds_y_list    


def odeint(stepper, bcs, f, y0, ts, *diff_args):

    def stepper_partial(state, t_crt):
        (y_crt, t_crt), _ = stepper(state, t_crt, f, *diff_args)
        y_crt = jax.ops.index_update(y_crt, bc_inds, bc_vals)
        return (y_crt, t_crt), y_crt

    # _, ys = jax.lax.scan(stepper_partial, (y0, ts[0]), ts[1:])

    ys = []
    state = (y0, ts[0])
    for (i, t_crt) in enumerate(ts[1:]):
        bc_inds, bc_vals = get_bcs(i + 1, bcs)
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
    gn_n_rows = args.gn_n_rows
    gn_n_cols = args.gn_n_cols

    senders = []
    receivers = []
    crt_state = []
    ref_state = []

    def valid_node(this_col, this_row):
        in_matrix = this_col >= 0 and this_col <= gn_n_cols - 1 and this_row >= 0 and this_row <= gn_n_rows - 1
        defect = (this_col, this_row) in args.deactivated_nodes
        return in_matrix and not defect
    
    # TODO: the double for-loop is slow for large size of structure - need to optimize the code
    inds_lookup = []
    valid_node_num = 0
    for row in range(gn_n_rows):
        for col in range(gn_n_cols):
            if valid_node(col, row):
                inds_lookup.append(valid_node_num)
                valid_node_num += 1
                ref_state.append(np.array([col*L0, row*L0, 0., 0., 0., 0.]))
                crt_state.append(np.array([col*L0, row*L0, 0., 0., 0., 0.]))
                receiver = row*gn_n_cols + col
                local_senders = [(col + 1, row), (col, row + 1), (col - 1, row), (col, row - 1)]
                for local_col, local_row in local_senders:
                    if valid_node(local_col, local_row):
                        senders.append(local_row*gn_n_cols + local_col)
                        receivers.append(receiver)
            else:
                inds_lookup.append(INVALID)

    senders = [inds_lookup[i] for i in senders]
    receivers = [inds_lookup[i] for i in receivers]
        
    n_node = np.array([gn_n_rows*gn_n_cols - len(args.deactivated_nodes)])
    n_edge = np.array([len(senders)])
    # n_edge = 2 * np.array([(gn_n_rows - 1)*gn_n_cols + (gn_n_cols - 1)*gn_n_rows])

    assert valid_node_num == n_node[0], f"Building nodes wrong!"
    assert len(senders) == n_edge[0], f"Building edges wrong!"

    print(f"Total number nodes = {n_node[0]}, total number of edges = {n_edge[0]}")
    crt_state = np.stack(crt_state)
    ref_state = np.stack(ref_state)
    ini_state = np.array(crt_state)
    senders = np.array(senders)
    receivers = np.array(receivers)

    mass = np.load(get_file_path('numpy', 'mass')) * np.ones(n_node)
    inertia = np.load(get_file_path('numpy', 'inertia')) * np.ones(n_node)
 
    node_features = {"ref_state": ref_state, "crt_state": crt_state, "mass": mass, "inertia": inertia}
    graph = jraph.GraphsTuple(nodes=node_features, edges={}, senders=senders, receivers=receivers,
        n_node=n_node, n_edge=n_edge, globals={})

    return graph, ini_state, inds_lookup


def update_graph():
    batch_forward = regression(jax_gpr_surrogate, False)

    def update_edge_fn(edges, senders, receivers, globals_):
        del edges, globals_
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
        del sent_edges, received_edges, globals_
        #TODO: delete unused variables
        _, _, crt_v, crt_w = unpack_state(nodes["crt_state"])
        kinetic_energy_per_node = 0.5 * args.density * nodes["mass"] * np.sum(crt_v**2, -1) + \
                                  0.5 * args.density * nodes["inertia"] * np.sum(crt_w**2, -1)
        return {"kinetic_energy": kinetic_energy_per_node}

    def update_global_fn(nodes, edges, globals_):
        del globals_
        total_kinetic_energy = nodes["kinetic_energy"]
        total_potential_energy = edges["potential_energy"]
        total_hamiltonian = total_kinetic_energy + total_potential_energy
        return {"total_hamiltonian": total_hamiltonian, 
                "total_kinetic_energy": total_kinetic_energy, 
                "total_potential_energy": total_potential_energy}

    net_fn = jraph.GraphNetwork(update_edge_fn=update_edge_fn,
                                update_node_fn=update_node_fn,
                                update_global_fn=update_global_fn)

    return net_fn


def hamiltonian(graph):
    net_fn = update_graph()

    def compute_hamiltonian(y):
        graph.nodes["crt_state"] = y
        new_graph = net_fn(graph)
        return new_graph.globals["total_hamiltonian"][0]

    def compute_kinetic_energy(y):
        graph.nodes["crt_state"] = y
        new_graph = net_fn(graph)
        return new_graph.globals["total_kinetic_energy"][0]    

    grad_hamiltonian = jax.grad(compute_hamiltonian)

    def state_rhs(y, t, *diff_args):
        x, q, v, w = unpack_state(y)
        grads = grad_hamiltonian(y)
        mass = graph.nodes["mass"]
        inertia = graph.nodes["inertia"]

        v_rhs = (-args.gn_damping * v - grads[:, :2]) / mass[:, None] / args.density
        w_rhs = (-args.gn_damping * w - grads[:, 2:3]) / inertia[:, None] / args.density

        rhs = np.hstack((np.zeros(grads[:, 3:].shape), v_rhs, w_rhs))

        return rhs

    return compute_kinetic_energy, compute_hamiltonian, state_rhs


def get_cross_energy(ys, inds):
    # TODO: mass is actually volume
    mass = np.load(get_file_path('numpy', 'mass'))
    inertia = np.load(get_file_path('numpy', 'inertia'))
    ks = 0.
    for index in inds:
        ks = ks + 0.5*args.density*mass*np.sum(ys[:, index, 3:5]**2, axis=-1) + 0.5*inertia*args.density*ys[:, index, 5]**2
    return ks


def simulate(uv_list, ts, y0=None):
    graph, ini_state, inds_lookup = build_graph()
    bc_inds_x_list, bc_inds_y_list = build_bc_inds(inds_lookup)
    compute_kinetic_energy, compute_hamiltonian, state_rhs = hamiltonian(graph)
    bcs = [bc_inds_x_list, bc_inds_y_list, uv_list, ini_state]

    if y0 is not None:
        ys_ = odeint(leapfrog, bcs, state_rhs, y0, ts)
        ys = np.vstack((y0[None, :], ys_))
    else:
        ys_ = odeint(leapfrog, bcs, state_rhs, ini_state, ts)
        ys = np.vstack((ini_state[None, :], ys_))        

    vmap_hamitonian = jax.jit(jax.vmap(compute_hamiltonian))
    vmap_kinetic_energy = jax.jit(jax.vmap(compute_kinetic_energy))
    hamitonians = vmap_hamitonian(ys)
    kinetic_energies = vmap_kinetic_energy(ys) 
    return ys, hamitonians, kinetic_energies, graph


 

