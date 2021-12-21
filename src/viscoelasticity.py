def dns_solve_viscoelasticity(c1, c2):
    '''
    这个函数介绍了viscoelasticity的解法，仅做参考意义，并未在gnmm这个项目中实际使用
    '''
    mesh, (bottom, top, left_corner) = build_mesh_and_boundaries_bulk()    
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

    ts, disps, vels = boundary_excitation_impulse(dt=1e-2, t_steps=1000, coef=10)
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

    xdmf_file_path = get_file_path('xdmf')
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

