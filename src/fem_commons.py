from src.arguments import args
import fenics as fe
import numpy as onp
import mshr


def build_mesh(c1, c2, save_mesh):
    porosity = args.porosity
    L0 = args.L0
    n_radial_points = 100

    def coords_fn(theta):
        r0 = L0 * onp.sqrt(2 * porosity) / onp.sqrt(onp.pi * (2 + c1**2 + c2**2))
        return r0 * (1 + c1 * fe.cos(4 * theta) + c2 * fe.cos(8 * theta))

    def build_base_pore():
        # Remark(Tianju): mshr may not work well if shift is not added.
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
        mesh = mshr.generate_mesh(material_domain, args.resolution * n_rows * n_cols)
        return mesh

    if args.shape_tag == 'beam':
        mesh = helper(lower_corner=fe.Point(0.5 * L0, 0.), 
                      upper_corner=fe.Point(1.5 * L0, L0), 
                      n_rows=1, 
                      n_cols=2)
    elif args.shape_tag == 'unit':
        mesh = helper(lower_corner=fe.Point(0., 0.), 
                      upper_corner=fe.Point(L0, L0), 
                      n_rows=1, 
                      n_cols=1)      
    elif args.shape_tag == 'dns':
        mesh = helper(lower_corner=fe.Point(0., 0.), 
                      upper_corner=fe.Point(args.dns_n_cols*L0, args.dns_n_rows*L0), 
                      n_rows=args.dns_n_rows, 
                      n_cols=args.dns_n_cols)         
    else:
        raise ValueError(f"Unknown shape: {args.shape_tag}")

    if save_mesh:
        mesh_file = fe.File(f'data/pvd/meshes/{args.shape_tag}/{args.case_id}_mesh.pvd')
        mesh.rename("mesh", "mesh")
        mesh_file << mesh
        fe.File(f'data/xml/meshes/{args.shape_tag}/{args.case_id}_mesh.xml') << mesh

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

