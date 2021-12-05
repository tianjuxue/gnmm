import fenics as fe
import numpy as onp
import os
import matplotlib.pyplot as plt
from src.arguments import args
from src.fem_commons import *


def file_path(file_type):
    if args.shape_tag == 'dns':
        return f'data/{file_type}/{args.shape_tag}/{args.case_id}/u.xdmf'
    elif args.shape_tag == 'bulk':
        return f'data/{file_type}/{args.shape_tag}/u.xdmf'
    else:
        raise ValueError(f"Unknown shape_tag {args.shape_tag}")


def build_mesh_and_boundaries_porous(c1, c2):
    L0 = args.L0 
    args.shape_tag = 'dns'
    args.resolution = 4
    args.dns_n_rows = 5
    args.dns_n_cols = 5

    xml_mesh_file = f'data/xml/meshes/{args.shape_tag}/{args.case_id}_mesh.xml'

    if not os.path.isfile(xml_mesh_file) or args.overwrite_mesh:
        mesh = build_mesh(c1, c2, True)
    else:
        mesh = fe.Mesh(xml_mesh_file)
 
    class Bottom(fe.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and x[1] < 1e-8

    class Top(fe.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and x[1] > args.dns_n_rows * L0 - 1e-8

    class LeftCorner(fe.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and  x[1] < 1e-8 and x[0] < L0/2      

    bottom = Bottom()
    top = Top()
    left_corner = LeftCorner()
    # boundaries = fe.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    # boundaries.set_all(0)
    # ds = fe.Measure('ds')(subdomain_data=boundaries)

    return mesh, (bottom, top, left_corner)


def build_mesh_and_boundaries_bulk():
    L0 = args.L0 
    args.shape_tag= 'bulk'
    args.dns_n_rows = 5
    args.dns_n_cols = 5    

    # diagonal = 'crossed'
    mesh = fe.RectangleMesh(fe.Point(0., 0.), fe.Point(args.dns_n_cols * L0, args.dns_n_rows * L0), 40, 40)

    class Bottom(fe.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and x[1] < 1e-8

    class Top(fe.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and x[1] > args.dns_n_rows * L0 - 1e-8

    class LeftCorner(fe.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and  x[1] < 1e-8 and x[0] < 1e-8  

    bottom = Bottom()
    top = Top()
    left_corner = LeftCorner()

    return mesh, (bottom, top, left_corner)



def boundary_excitation_impulse(dt, t_steps, coef):
    ts = onp.arange(0, (t_steps + 1)*dt, dt)
    alpha = ts / ts[-1]
    bc_activation = onp.where(alpha < 1./coef, coef*alpha, 1.)
    disps = -0.1 * bc_activation * args.dns_n_rows * args.L0 
    vels = onp.hstack((0., onp.diff(disps)))
    return ts, disps, vels


def boundary_excitation_sin(dt, t_steps, coef):
    ts = onp.arange(0, (t_steps + 1)*dt, dt)
    alpha = ts / ts[-1]
    bc_activation = onp.where(alpha < 1./coef, onp.sin(2*onp.pi*coef*alpha), 0.)
    disps = -0.1 * bc_activation * args.dns_n_rows * args.L0 
    vels = onp.hstack((0., onp.diff(disps)))
    return ts, disps, vels



def dns_solve_viscoelasticity(c1, c2):
    '''
    这个函数介绍了viscoelasticity的解法，仅做参考意义，并未在gnmm这个项目中实际使用
    '''

    # mesh, (bottom, top, left_corner) = build_mesh_and_boundaries_bulk()    
    mesh, (bottom, top, left_corner) = build_mesh_and_boundaries_porous(c1, c2)    

    U = fe.VectorElement('CG', mesh.ufl_cell(), 1)  
    W = fe.VectorElement('CG', mesh.ufl_cell(), 1)
    Q = fe.TensorElement('DG', mesh.ufl_cell(), 0)
    M = fe.FunctionSpace(mesh, fe.MixedElement(U, W, Q))

    m_test = fe.TestFunctions(M)
    m_crt = fe.Function(M)
    m_pre = fe.Function(M)

    u_test, v_test, Fi_test = m_test
    u_crt, v_crt, Fi_crt = fe.split(m_crt)
    u_pre, v_pre, Fi_pre = fe.split(m_pre)

    m_crt.assign(fe.Constant((0., 0., 0., 0., 1., 0., 0., 1.)))
    m_pre.assign(fe.Constant((0., 0., 0., 0., 1., 0., 0., 1.)))

    u_load_top = fe.Expression("disp", disp=0., degree=1)
    u_BC_top_y = fe.DirichletBC(M.sub(0).sub(1), u_load_top,  top)
    u_BC_top_x = fe.DirichletBC(M.sub(0).sub(0), fe.Constant(0.),  top)
    u_BC_bottom_y = fe.DirichletBC(M.sub(0).sub(1), fe.Constant(0.), bottom)
    u_BC_bottom_x = fe.DirichletBC(M.sub(0).sub(0), fe.Constant(0.), bottom)
    u_BC_corner_x = fe.DirichletBC(M.sub(0).sub(0), fe.Constant(0.), left_corner, method='pointwise')
    v_load_top = fe.Expression("vel", vel=0., degree=1)
    v_BC_top_y = fe.DirichletBC(M.sub(1).sub(1), v_load_top, top)
    v_BC_top_x = fe.DirichletBC(M.sub(1).sub(0), fe.Constant(0.),  top)
    v_BC_bottom_y = fe.DirichletBC(M.sub(1).sub(1), fe.Constant(0.), bottom)
    v_BC_bottom_x = fe.DirichletBC(M.sub(1).sub(0), fe.Constant(0.), bottom)
    v_BC_corner_x = fe.DirichletBC(M.sub(1).sub(0), fe.Constant(0.), left_corner, method='pointwise')
    # bcs = [u_BC_top_y, u_BC_bottom_y, u_BC_corner_x, v_BC_top_y, v_BC_bottom_y, v_BC_corner_x]
    bcs = [u_BC_top_y, u_BC_top_x, u_BC_bottom_y, u_BC_bottom_x, v_BC_top_y, v_BC_top_x, v_BC_bottom_y, v_BC_bottom_x]

    # debug惨痛教训：fe.variable的位置要放对了
    Fi_crt = fe.variable(Fi_crt)
    theta = 0.5
    u_rhs = theta*u_pre + (1 - theta)*u_crt
    v_rhs = theta*v_pre + (1 - theta)*v_crt
    Fi_rhs = theta*Fi_pre + (1 - theta)*Fi_crt

    ts, disps, vels = boundary_excitation_impulse(dt=1e-2, t_steps=100, coef=10)
    dt = ts[1] - ts[0]

    kappa = 10.
    mu = 1.
    beta = 0.5 
    tau = 0.1
    damping_coeff = 0.001
    density = 1e-3

    def Uiso(F, mu):
        C = F.T*F 
        Ic = fe.tr(C) 
        J  = fe.det(F)
        Jinv = J**(-2 / 3)
        return (mu/2)*(Jinv * (Ic + 1) - 3)

    def Uvol(F, kappa):
        J = fe.det(F)
        return (kappa/2)*(J-1)**2

    def Phi(Fe, Di, eta):
        return 0.5*eta*fe.inner(Di, Di)*fe.det(Fe)

    dim = u_crt.geometric_dimension()
    I = fe.Identity(dim)
    F_rhs = DeformationGradient(u_rhs)
    F_rhs = fe.variable(F_rhs)
    Fe_rhs = F_rhs * fe.inv(Fi_rhs)

    Fidot = (Fi_crt - Fi_pre) / dt
    Di = (Fe_rhs*Fidot) * fe.inv(F_rhs)
    strain_energy_density = Uvol(F_rhs, kappa) + Uiso(F_rhs, mu) + Uiso(Fe_rhs, beta*mu)
    psi = strain_energy_density + dt*Phi(Fe_rhs, Di, 0.5*tau*beta*mu) + dt*Uvol(Fi_rhs, kappa)

    PK_stress = fe.diff(psi, F_rhs)
    vis_stress = fe.diff(psi, Fi_crt)

    kinetic_energy = 0.5 * density * fe.dot(v_rhs, v_rhs) * fe.dx
    strain_energy = strain_energy_density * fe.dx
    nondissipative_energy = strain_energy + kinetic_energy

    mode = 'default'
    if mode == 'default': # 既有viscosity，也有inertia
        v_res = density * fe.dot(v_crt - v_pre, v_test) * fe.dx + dt * fe.inner(PK_stress, fe.grad(v_test)) * fe.dx
        u_res = fe.dot(u_crt - u_pre, u_test) * fe.dx - dt * fe.dot(v_rhs, u_test) * fe.dx
        Fi_res = fe.inner(vis_stress, Fi_test) * fe.dx
    elif mode == 'no_viscosity': # 没有viscosity，有inertia
        v_res = density * fe.dot(v_crt - v_pre, v_test) * fe.dx + dt * fe.inner(PK_stress, fe.grad(v_test)) * fe.dx
        u_res = fe.dot(u_crt - u_pre, u_test) * fe.dx - dt * fe.dot(v_rhs, u_test) * fe.dx
        Fi_res = fe.inner(Fi_crt - I, Fi_test) * fe.dx      
    elif mode == 'no_inertia': # 有viscosity，没有inertia
        u_res = fe.inner(PK_stress, fe.grad(u_test)) * fe.dx
        v_res = density * fe.dot(v_crt - v_pre, v_test) * fe.dx 
        Fi_res = fe.inner(Fi_crt - I, Fi_test) * fe.dx
    elif mode == 'quasi_static': # 既没有viscosity，也没有inertia
        u_res = fe.inner(PK_stress, fe.grad(u_test)) * fe.dx
        v_res = density * fe.dot(v_crt - v_pre, v_test) * fe.dx 
        Fi_res = fe.inner(Fi_crt - I, Fi_test) * fe.dx
    else:
         raise ValueError(f'Unknown mode {mode}')

    xdmf_file_path = file_path('xdmf')
    xdmf_file = fe.XDMFFile(xdmf_file_path)
    xdmf_file.parameters["functions_share_mesh"] = True

    nondissipative_energies = []

    for i, t in enumerate(ts[1:]):
        print(f"\nStep {i}")

        if i % 1 == 0:
            u_plot, v_plot, Fi_plot = m_pre.split()
            u_plot.rename("u", "u")
            v_plot.rename("v", "v")
            Fi_plot.rename("Fi", "Fi")
            xdmf_file.write(u_plot, i)
            xdmf_file.write(v_plot, i)
            xdmf_file.write(Fi_plot, i)

        u_load_top.disp = disps[i + 1]
        v_load_top.vel = vels[i + 1]

        fe.solve(v_res + u_res + Fi_res == 0, m_crt, bcs)
        m_pre.assign(m_crt)

        nondissipative_energy_val = fe.assemble(nondissipative_energy)
        nondissipative_energies.append(nondissipative_energy_val)
        print(f"nondissipative_energy_val = {nondissipative_energy_val}")

    fig = plt.figure()
    plt.plot(nondissipative_energies, linestyle='-', marker='o', color='black')
    plt.tick_params(labelsize=14)
    plt.show()


def dns_solve_implicit_euler(c1, c2):
    mesh, (bottom, top, left_corner) = build_mesh_and_boundaries_porous(c1, c2)
    # mesh, (bottom, top, left_corner) = build_mesh_and_boundaries_bulk()

    U = fe.VectorElement('CG', mesh.ufl_cell(), 1)  
    W = fe.VectorElement('CG', mesh.ufl_cell(), 1)
    M = fe.FunctionSpace(mesh, U * W)

    # m_delta = fe.TrialFunctions(M)
    m_test = fe.TestFunctions(M)
    m_crt = fe.Function(M)
    m_pre = fe.Function(M)
    u_test, v_test = m_test
    u_crt, v_crt = fe.split(m_crt)
    u_pre, v_pre = fe.split(m_pre)

    u_load_top = fe.Expression("disp", disp=0., degree=1)
    u_BC_top_y = fe.DirichletBC(M.sub(0).sub(1), u_load_top,  top)
    u_BC_top_x = fe.DirichletBC(M.sub(0).sub(0), fe.Constant(0.),  top)
    u_BC_bottom_y = fe.DirichletBC(M.sub(0).sub(1), fe.Constant(0.), bottom)
    u_BC_bottom_x = fe.DirichletBC(M.sub(0).sub(0), fe.Constant(0.), bottom)
    u_BC_corner_x = fe.DirichletBC(M.sub(0).sub(0), fe.Constant(0.), left_corner, method='pointwise')
    v_load_top = fe.Expression("vel", vel=0., degree=1)
    v_BC_top_y = fe.DirichletBC(M.sub(1).sub(1), v_load_top, top)
    v_BC_top_x = fe.DirichletBC(M.sub(1).sub(0), fe.Constant(0.),  top)
    v_BC_bottom_y = fe.DirichletBC(M.sub(1).sub(1), fe.Constant(0.), bottom)
    v_BC_bottom_x = fe.DirichletBC(M.sub(1).sub(0), fe.Constant(0.), bottom)
    v_BC_corner_x = fe.DirichletBC(M.sub(1).sub(0), fe.Constant(0.), left_corner, method='pointwise')
    # bcs = [u_BC_top_y, u_BC_bottom_y, u_BC_corner_x, v_BC_top_y, v_BC_bottom_y, v_BC_corner_x]
    bcs = [u_BC_top_y, u_BC_top_x, u_BC_bottom_y, u_BC_bottom_x, v_BC_top_y, v_BC_top_x, v_BC_bottom_y, v_BC_bottom_x]
 
    theta = 0.5
    u_rhs = theta*u_pre + (1 - theta)*u_crt
    v_rhs = theta*v_pre + (1 - theta)*v_crt
    strain_energy_density, PK_stress = NeoHookeanEnergy(u_rhs)

    ts, disps, vels = boundary_excitation_impulse(dt=1e-2, t_steps=100, coef=2)
    dt = ts[1] - ts[0]

    args.damping_coeff = 1e-1

    v_res = args.density * fe.dot(v_crt - v_pre, v_test) * fe.dx + dt * fe.inner(PK_stress, fe.grad(v_test)) * fe.dx + \
            dt * args.damping_coeff * fe.dot(v_rhs, v_test) * fe.dx
    u_res = fe.dot(u_crt - u_pre, u_test) * fe.dx - dt * fe.dot(v_rhs, u_test) * fe.dx

    xdmf_file_path = file_path('xdmf')
    xdmf_file = fe.XDMFFile(xdmf_file_path)
    xdmf_file.parameters["functions_share_mesh"] = True

    kinetic_energy = 0.5 * args.density * fe.dot(v_rhs, v_rhs) * fe.dx
    strain_energy = strain_energy_density * fe.dx
    nondissipative_energy = strain_energy + kinetic_energy

    nondissipative_energies = []
    for i, t in enumerate(ts[1:]):
        print(f"\nStep {i}")

        if i % 1 == 0:
            u_plot, v_plot = m_pre.split()
            u_plot.rename("u", "u")
            v_plot.rename("v", "v")
            xdmf_file.write(u_plot, i)
            xdmf_file.write(v_plot, i)

        u_load_top.disp = disps[i + 1]
        v_load_top.vel = vels[i + 1]

        fe.solve(v_res + u_res == 0, m_crt, bcs)
        m_pre.assign(m_crt)

        nondissipative_energy_val = fe.assemble(nondissipative_energy)
        nondissipative_energies.append(nondissipative_energy_val)
        print(f"nondissipative_energy_val = {nondissipative_energy_val}")

    fig = plt.figure()
    plt.plot(nondissipative_energies, linestyle='-', marker='o', color='black')
    plt.tick_params(labelsize=14)
    # plt.xlabel('Optimization step', fontsize=14)
    # plt.ylabel('Objective value', fontsize=14)
    plt.show()


def main():
    shape_params = [(0., 0.), (-0.2, 0.2), (-0.2, 0.)]
    args.case_id = 'poreA'
    args.overwrite_mesh = True
    dns_solve_implicit_euler(*shape_params[0])
    # dns_solve_viscoelasticity(*shape_params[0])

if __name__ == '__main__':
    main()
