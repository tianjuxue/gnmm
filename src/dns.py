import fenics as fe
import numpy as onp
import os
from src.arguments import args
from src.fem_commons import *


def dns_solve(c1, c2):
    L0 = args.L0 
    args.shape_tag= 'dns'
    args.resolution = 5
    args.dns_n_rows = 5
    args.dns_n_cols = 5

    xml_mesh_file = f'data/xml/meshes/{args.shape_tag}/{args.case_id}_mesh.xml'
    if os.path.isfile(xml_mesh_file):
        mesh = fe.Mesh(xml_mesh_file)
    else:
        mesh = build_mesh(c1, c2, True)
 

    V = fe.VectorFunctionSpace(mesh, 'P', 1)
    E = fe.FunctionSpace(mesh, 'DG', 0)

    du = fe.TrialFunction(V)
    v = fe.TestFunction(V)

    u_crt = fe.Function(V)
    u_pre = fe.Function(V, name="u")

    vel_crt = fe.Function(V)
    vel_pre = fe.Function(V, name="v")

    class Bottom(fe.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and x[1] < 1e-8

    class Top(fe.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and x[1] > args.dns_n_rows * L0 - 1e-8

    bottom = Bottom()
    top = Top()
    boundaries = fe.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    boundaries.set_all(0)
    ds = fe.Measure('ds')(subdomain_data=boundaries)
    
    u_load_top = fe.Expression("disp", disp=0., degree=1)
    u_BC_top = fe.DirichletBC(V.sub(1), u_load_top,  top)
    u_BC_bottom = fe.DirichletBC(V.sub(1), fe.Constant(0.),  bottom)
    u_bcs = [u_BC_top, u_BC_bottom]

    vel_load_top = fe.Expression("vel", vel=0., degree=1)
    vel_BC_top = fe.DirichletBC(V.sub(1), vel_load_top,  top)
    vel_BC_bottom = fe.DirichletBC(V.sub(1), fe.Constant(0.),  bottom)
    vel_bcs = [vel_BC_top, vel_BC_bottom]

    energy_density, PK_stress = NeoHookeanEnergy(u_pre)
    total_energy = energy_density * fe.dx 

    rho_0 = 1.

    dt = 1e-4
    ts = onp.arange(0, 1001*dt, dt)
    coef = 2
    alpha = ts / ts[-1]
    bc_activation = onp.where(alpha < 1./coef, coef*alpha, 1.)
    disps = -0.05 * bc_activation * args.dns_n_rows * L0
    vels = onp.hstack((0., onp.diff(disps)))

    vel_res = rho_0 * fe.dot(vel_crt - vel_pre, v) * fe.dx + dt * fe.inner(PK_stress, fe.grad(v)) * fe.dx
    u_res = fe.dot(u_crt - u_pre, v) * fe.dx - dt * fe.dot(vel_crt, v) * fe.dx


    # vel_left = rho_0 * fe.dot(du, v) * fe.dx 
    # vel_right = rho_0 * fe.dot(vel_pre, v) * fe.dx - dt * fe.inner(PK_stress, fe.grad(v)) * fe.dx
    # u_left = fe.dot(du, v) * fe.dx 
    # u_right =  fe.dot(u_pre, v) * fe.dx  + dt * fe.dot(vel_crt, v) * fe.dx


    xdmf_file = fe.XDMFFile(f'data/xdmf/{args.shape_tag}/{args.case_id}/u.xdmf')
    xdmf_file.parameters["functions_share_mesh"] = True

    for i, t in enumerate(ts[1:]):
        print(f"\nStep {i}")

        if i % 10 == 0:
            xdmf_file.write(u_pre, i)
            xdmf_file.write(vel_pre, i)

        u_load_top.disp = disps[i + 1]
        vel_load_top.disp = vels[i + 1]

        fe.solve(vel_res == 0, vel_crt, vel_bcs)
        fe.solve(u_res == 0, u_crt, u_bcs)

        # fe.solve(vel_left == vel_right, vel_crt, vel_bcs)
        # fe.solve(u_left == u_right, u_crt, u_bcs)

        vel_pre.assign(vel_crt)
        u_pre.assign(u_crt)


def main():
    shape_params = [(0., 0.), (-0.2, 0.2)]
    args.case_id = 'poreA'
    dns_solve(*shape_params[0])


if __name__ == '__main__':
    main()
