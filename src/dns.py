import fenics as fe
import numpy as onp
import os
import matplotlib.pyplot as plt
from src.arguments import args
from src.fem_commons import *


def build_mesh_and_boundaries_porous(c1, c2):
    L0 = args.L0 

    shape_key = 'dns'
    xml_path = get_file_path('xml', ['meshes', shape_key])
    if not os.path.isfile(xml_path) or args.overwrite_mesh:
        mesh = build_mesh(c1, c2, shape_key, True)
    else:
        mesh = fe.Mesh(xml_path)
 
    class Bottom(fe.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and x[1] < 1e-8

    class Top(fe.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and x[1] > args.dns_n_rows * L0 - 1e-8
            # return  x[1] > args.dns_n_rows * L0 - L0 / 2.

    class LeftCorner(fe.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and x[1] < 1e-8 and x[0] < L0/2   

    class Cross(fe.SubDomain):
        def inside(self, x, on_boundary):
           return x[1] < L0

    bottom = Bottom()
    top = Top()
    left_corner = LeftCorner()
    cross = Cross()

    # boundaries = fe.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    # boundaries.set_all(0)
    # ds = fe.Measure('ds')(subdomain_data=boundaries)
    domains = fe.MeshFunction("size_t", mesh, mesh.topology().dim())
    domains.set_all(0)
    dx = fe.Measure('dx')(domain=mesh, subdomain_data=domains)
    cross.mark(domains, 1)
    return mesh, (bottom, top, left_corner), dx


def build_mesh_and_boundaries_bulk():
    L0 = args.L0 

    # diagonal = 'crossed'
    bulk_resolution = 1
    mesh = fe.RectangleMesh(fe.Point(0., 0.), fe.Point(args.bulk_n_cols * L0, args.bulk_n_rows * L0), 
        bulk_resolution*args.bulk_n_cols, bulk_resolution*args.bulk_n_rows)

    class Bottom(fe.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and x[1] < 1e-8

    class Top(fe.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and x[1] > args.bulk_n_rows * L0 - 1e-8

    class LeftCorner(fe.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and  x[1] < 1e-8 and x[0] < 1e-8  

    class Cross(fe.SubDomain):
        def inside(self, x, on_boundary):
           return x[1] < L0 + 1e-8

    # TODO: clearly redundant code
    bottom = Bottom()
    top = Top()
    left_corner = LeftCorner()
    cross = Cross()

    domains = fe.MeshFunction("size_t", mesh, mesh.topology().dim())
    domains.set_all(0)
    dx = fe.Measure('dx')(domain=mesh, subdomain_data=domains)
    cross.mark(domains, 1)
       
    return mesh, (bottom, top, left_corner), dx


def build_bcs(uv_list, uv_bcs):
    bcs = []
    for edge_index, uv in enumerate(uv_list):
        if uv is not None:
            bcs = bcs + uv_bcs[edge_index]
    return bcs


def apply_bcs(time_index, uv_list, uv_loads):
    for edge_index, uv in enumerate(uv_list):
        if uv is not None:
            disps, vels = uv
            u_load, v_load = uv_loads[edge_index]
            u_load.disp = disps[time_index]
            v_load.vel = vels[time_index]


def dns_solve_implicit_euler(c1, c2, uv_list, ts, save_files=True):
    if args.shape_tag == 'dns':
        mesh, (bottom, top, left_corner), dx = build_mesh_and_boundaries_porous(c1, c2)
    if args.shape_tag == 'bulk':
        mesh, (bottom, top, left_corner), dx = build_mesh_and_boundaries_bulk()

    U = fe.VectorElement('CG', mesh.ufl_cell(), 1)  
    W = fe.VectorElement('CG', mesh.ufl_cell(), 1)
    M = fe.FunctionSpace(mesh, U * W)
    E = fe.FunctionSpace(mesh, 'DG', 0)

    # m_delta = fe.TrialFunctions(M)
    m_test = fe.TestFunctions(M)
    m_crt = fe.Function(M)
    m_pre = fe.Function(M)
    u_test, v_test = m_test
    u_crt, v_crt = fe.split(m_crt)
    u_pre, v_pre = fe.split(m_pre)

    # print(len(m_crt.vector()))

    uv_bcs = []
    uv_loads = []
    edges = [bottom, top]
    for edge in edges:
        for i in range(2):
            u_load = fe.Expression("disp", disp=0., degree=1)
            u_BC = fe.DirichletBC(M.sub(0).sub(i), u_load,  edge) # method='pointwise'
            v_load = fe.Expression("vel", vel=0., degree=1)
            v_BC = fe.DirichletBC(M.sub(1).sub(i), v_load,  edge)
            uv_bcs.append([u_BC, v_BC])
            uv_loads.append([u_load, v_load])

    bcs = build_bcs(uv_list, uv_bcs)
    dt = float(ts[1] - ts[0])
    theta = 0.5
    u_rhs = theta*u_pre + (1 - theta)*u_crt
    v_rhs = theta*v_pre + (1 - theta)*v_crt

    if args.dns_dynamics:

        strain_energy_density, PK_stress = NeoHookeanEnergy(u_rhs)
        if args.dns_damping == 0.:
            v_res = args.density * fe.dot(v_crt - v_pre, v_test) * fe.dx + dt * fe.inner(PK_stress, fe.grad(v_test)) * fe.dx
        else:
            v_res = args.density * fe.dot(v_crt - v_pre, v_test) * fe.dx + dt * fe.inner(PK_stress, fe.grad(v_test)) * fe.dx + \
                    dt * args.dns_damping * fe.dot(v_rhs, v_test) * fe.dx       
        u_res = fe.dot(u_crt - u_pre, u_test) * fe.dx - dt * fe.dot(v_rhs, u_test) * fe.dx
    else:
        strain_energy_density, PK_stress = NeoHookeanEnergy(u_crt)
        u_res = fe.inner(PK_stress, fe.grad(u_test)) * fe.dx
        v_res = fe.dot(v_crt, v_test) * fe.dx

    xdmf_file = fe.XDMFFile(get_file_path('xdmf'))
    xdmf_file.parameters["functions_share_mesh"] = True
    u_vtk_file = fe.File(get_file_path('pvd', ['sols', 'u']))

    kinetic_energy = 0.5 * args.density * fe.dot(v_rhs, v_rhs) * fe.dx
    strain_energy = strain_energy_density * fe.dx
    nondissipative_energy = strain_energy + kinetic_energy
    cross_energy = 0.5 * args.density * fe.dot(v_rhs, v_rhs) * dx(1)
    nondissipative_energies = [0.]
    kinetic_energies = [0.]
    cross_energies = [0.]
    for i, t in enumerate(ts[1:]):
        print(f"\nStep {i}")

        if i % 1 == 0 and save_files:
            e_plot = fe.project(strain_energy_density, E)
            u_plot, v_plot = m_pre.split()
            u_plot.rename("u", "u")
            v_plot.rename("v", "v")
            e_plot.rename("e", "e")
            xdmf_file.write(u_plot, i)
            xdmf_file.write(v_plot, i)
            xdmf_file.write(e_plot, i)
            u_vtk_file << u_plot

        apply_bcs(i + 1, uv_list, uv_loads)

        solver_parameters = {'newton_solver': {'maximum_iterations': 20}}
        fe.solve(v_res + u_res == 0, m_crt, bcs, solver_parameters=solver_parameters)
        m_pre.assign(m_crt)

        nondissipative_energy_val = fe.assemble(nondissipative_energy)
        kinetic_energy_val = fe.assemble(kinetic_energy)
        cross_energy_val = fe.assemble(cross_energy)
        nondissipative_energies.append(nondissipative_energy_val)
        kinetic_energies.append(kinetic_energy_val)
        cross_energies.append(cross_energy_val)
        print(f"nondissipative_energy_val = {nondissipative_energy_val}")
        print(f"kinetic_energy_val = {kinetic_energy_val}")
        print(f"cross_energy_val = {cross_energy_val}")

    if save_files:
        nondissipative_energies = onp.array(nondissipative_energies)
        kinetic_energies = onp.array(kinetic_energies)
        cross_energies = onp.array(cross_energies)
        energies = onp.stack((ts, nondissipative_energies, kinetic_energies, cross_energies))
        onp.save(get_file_path('numpy', 'energy'), energies)


def main():
    pore_ids = ['poreA', 'poreB', 'poreC', 'poreD', 'poreE']
    pore_index = 0
    args.pore_id = pore_ids[pore_index]
    args.overwrite_mesh = False

    # args.shape_tag= 'bulk'

    args.shape_tag = 'dns'
    args.dns_n_rows = 1
    args.dns_n_cols = 1
    args.dns_damping = 0.
    args.description = ''
    args.coef = 10

    dt = 1e-4
    ts = np.arange(0, 11*dt, dt)

    uv_top_x = compute_uv_bc_vals(ts, bc_excitation_fixed, args.dns_n_rows)
    uv_top_y = compute_uv_bc_vals(ts, bc_excitation_impulse, args.dns_n_rows)
    uv_list = [None, None, uv_top_x, None]

    dns_solve_implicit_euler(*get_shape_params(), uv_list, ts)


if __name__ == '__main__':
    main()
