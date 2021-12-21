from src.arguments import args
import fenics as fe
import numpy as onp
import jax.numpy as np
import mshr


def get_file_path(file_type, keyword=None):
    if file_type == 'xdmf':
        if args.shape_tag == 'dns':
            return f'data/{file_type}/{args.shape_tag}/{args.pore_id}/resolution_{args.resolution}' + \
                   f'_size_{args.dns_n_cols}x{args.dns_n_rows}_description_{args.description}_u.xdmf'
        if args.shape_tag == 'bulk':
            return f'data/{file_type}/{args.shape_tag}/u.xdmf'
        if args.shape_tag == 'beam':
            return f'data/{file_type}/{args.shape_tag}/{args.pore_id}/u.xdmf'

    if file_type == 'pdf':
        if args.shape_tag == 'dns':
            if keyword == 'energy' or keyword == 'disp':
                return f"data/{file_type}/{args.shape_tag}/{keyword}/{args.description}_{args.pore_id}.pdf"
        if args.shape_tag == 'bulk':
            return f"data/{file_type}/{args.shape_tag}/energy.pdf"
        if args.shape_tag == 'beam':
            if keyword == 'energy' or keyword == 'disp':
                return f"data/{file_type}/{args.shape_tag}/{keyword}/{args.description}_{args.pore_id}.pdf"
            if hasattr(keyword, '__len__'):
                topic, name = keyword
                return f"data/{file_type}/{args.shape_tag}/{topic}/{name}.pdf"

    if file_type == 'xml':
        mesh_or_sol, var_name = keyword
        if args.shape_tag == 'dns':
            return f'data/{file_type}/{mesh_or_sol}/{args.shape_tag}/{args.pore_id}_resolution_{args.resolution}' + \
                   f'_size_{args.dns_n_cols}x{args.dns_n_rows}.xml'
        if args.shape_tag == 'beam':
            return f'data/{file_type}/{mesh_or_sol}/{args.shape_tag}/{args.pore_id}_{var_name}_resolution_{args.resolution}.xml'

    if file_type == 'pvd':
        mesh_or_sol, var_name = keyword
        if args.shape_tag == 'beam':
            return f'data/{file_type}/{mesh_or_sol}/{args.shape_tag}/{args.pore_id}/{var_name}.pvd'
        if args.shape_tag == 'dns':
            return f'data/{file_type}/{mesh_or_sol}/{args.shape_tag}/{args.pore_id}_resolution_{args.resolution}_' + \
                   f'size_{args.dns_n_cols}x{args.dns_n_rows}_description_{args.description}/{var_name}.pvd'
    
    if file_type == 'numpy':
        if args.shape_tag == 'dns':
            if keyword == 'energy':
                return  f'data/{file_type}/{args.shape_tag}/{args.pore_id}_resolution_{args.resolution}' + \
                        f'_size_{args.dns_n_cols}x{args.dns_n_rows}_{keyword}_description_{args.description}.npy'
        if args.shape_tag == 'beam':
            if keyword == 'data':
                return f'data/{file_type}/{args.shape_tag}/energy_sample_{args.num_samples}_resolution_{args.resolution}_{args.pore_id}'
            if keyword in ['mass', 'inertia', 'bounds']:
                return f"data/{file_type}/{args.shape_tag}/{args.pore_id}_{keyword}.npy"
            if keyword in ['train_errors', 'validation_errors', 'test_errors']:
                return f'data/{file_type}/{args.shape_tag}/{keyword}.npy'
            if keyword == 'energy':
                return  f'data/{file_type}/{args.shape_tag}/{args.pore_id}_resolution_{args.resolution}' + \
                        f'_size_{args.gn_n_cols}x{args.gn_n_rows}_{keyword}_description_{args.description}.npy'            

    if file_type == 'pickle':
        return f"data/{file_type}/jax_{keyword}_resolution_{args.resolution}_{args.pore_id}.pkl" 

    if file_type == 'mp4':
        return f'data/{file_type}/{args.description}_{args.pore_id}.mp4'

    raise ValueError(f"Fail to return a file path! file_type = {file_type}, keyword = {keyword}")



def bc_excitation_impulse(t_frac):
    coef = args.coef
    return np.where(t_frac < 1./coef, coef*t_frac, 1.)


def bc_excitation_fixed(t_frac):
    return np.zeros_like(t_frac)


def compute_uv_bc_vals(ts, bc_activation_fn, n_rows):
    t_frac = ts / ts[-1]
    disps = -args.amp * bc_activation_fn(t_frac) * n_rows * args.L0 
    vels = np.hstack((0., np.diff(disps)))
    return disps, vels


def get_shape_params():
    shape_params = [(0., 0.), (-0.05, 0.), (-0.1, 0.), (-0.15, 0.), (-0.2, 0.)]
    pore_ids = ['poreA', 'poreB', 'poreC', 'poreD', 'poreE']
    for i, pore_id in enumerate(pore_ids):
        if pore_id == args.pore_id:
            return shape_params[i]

    raise ValueError(f"No matching pore_id!")


def build_mesh(c1, c2, shape_key, save_mesh):
    porosity = args.porosity
    L0 = args.L0
    n_radial_points = 100

    def coords_fn(theta):
        r0 = L0 * onp.sqrt(2 * porosity) / onp.sqrt(onp.pi * (2 + c1**2 + c2**2))
        return r0 * (1 + c1 * fe.cos(4 * theta) + c2 * fe.cos(8 * theta))

    def build_base_pore():
        # Remark(Tianju): mshr may not work well if shift is not added, when jax.numpy is used.
        shift = 0.
        thetas = [float(i) * 2 * onp.pi / n_radial_points + shift for i in range(n_radial_points)]
        radii = [coords_fn(theta) for theta in thetas]
        points = [(rtheta * onp.cos(theta), rtheta * onp.sin(theta)) for rtheta, theta in zip(radii, thetas)]
        return onp.array(points), onp.array(radii), onp.array(thetas)

    def build_pore_polygon(offset):
        base_pore_points, radii, thetas = build_base_pore()
        points = [fe.Point(p[0] + offset[0], p[1] + offset[1]) for p in base_pore_points]
        pore = mshr.Polygon(points)
        return pore

    def helper(lower_corner, upper_corner, n_rows, n_cols):
        material_domain = mshr.Rectangle(lower_corner, upper_corner)  
        for i in range(n_cols + 1):
            for j in range(n_rows + 1):
                pore = build_pore_polygon((i * L0, j * L0))
                material_domain = material_domain - pore
        res_tmp = 1 if shape_key == 'beam' else onp.sqrt(n_rows * n_cols)
        mesh = mshr.generate_mesh(material_domain, args.resolution * res_tmp)
        return mesh

    if shape_key == 'beam':
        mesh = helper(lower_corner=fe.Point(0.5 * L0, 0.), 
                      upper_corner=fe.Point(1.5 * L0, L0), 
                      n_rows=1, 
                      n_cols=2)
    elif shape_key == 'unit':
        mesh = helper(lower_corner=fe.Point(0., 0.), 
                      upper_corner=fe.Point(L0, L0), 
                      n_rows=1, 
                      n_cols=1)      
    elif shape_key == 'dns':
        mesh = helper(lower_corner=fe.Point(0., 0.), 
                      upper_corner=fe.Point(args.dns_n_cols*L0, args.dns_n_rows*L0), 
                      n_rows=args.dns_n_rows, 
                      n_cols=args.dns_n_cols)         
    else:
        raise ValueError(f"Unknown shape: {shape_key}")

    if save_mesh:
        mesh_file = fe.File(get_file_path('pvd', ['meshes', shape_key]))
        mesh.rename("mesh", "mesh")
        mesh_file << mesh
        xml_path = get_file_path('xml', ['meshes', shape_key])
        fe.File(xml_path) << mesh

    return mesh


def DeformationGradient(u):
    I = fe.Identity(u.geometric_dimension())
    return I + fe.grad(u)


def NeoHookeanEnergy(u):
    young_modulus = args.young_modulus
    poisson_ratio = args.poisson_ratio
    shear_mod = young_modulus / (2 * (1 + poisson_ratio))
    bulk_mod = young_modulus / (3 * (1 - 2*poisson_ratio))
    F = DeformationGradient(u)
    F = fe.variable(F)
    J = fe.det(F)
    Jinv = J**(-2 / 3)
    I1 = fe.tr(F.T * F)
    energy = ((shear_mod / 2) * (Jinv * (I1 + 1) - 3) +
              (bulk_mod / 2) * (J - 1)**2) 
    first_pk_stress = fe.diff(energy, F)
    return energy, first_pk_stress

