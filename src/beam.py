import fenics as fe
import jax.numpy as np
import numpy as onp
import shutil
import os
import glob
import mshr
from src.dr import DynamicRelaxSolve
from src.arguments import args
from src.utils import walltime
import datetime
import gin
import jax


def angle_to_rot_mat(theta):
    return onp.array([[onp.cos(theta), -onp.sin(theta)], [onp.sin(theta), onp.cos(theta)]])


def build_mesh(c1, c2, shape_tag, save_mesh):
    porosity = args.porosity
    L0 = args.L0

    resolution = 30
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
        mesh = mshr.generate_mesh(material_domain, resolution * n_rows * n_cols)
        return mesh

    if shape_tag == 'beam':
        mesh = helper(lower_corner=fe.Point(0.5 * L0, 0.), 
                      upper_corner=fe.Point(1.5 * L0, L0), 
                      n_rows=1, 
                      n_cols=2)
    elif shape_tag == 'unit':
        mesh = helper(lower_corner=fe.Point(0., 0.), 
                      upper_corner=fe.Point(L0, L0), 
                      n_rows=1, 
                      n_cols=1)        
    else:
        raise ValueError(f"Unknown shape: {shape_tag}")

    if save_mesh:
        mesh_file = fe.File(f'data/pvd/{shape_tag}_mesh.pvd')
        mesh.rename("mesh", "mesh")
        mesh_file << mesh

    return mesh


def compute_mass_inertia(c1, c2, case_id):
    L0 = args.L0
    mesh = build_mesh(c1, c2, 'unit', True)
    sc = fe.SpatialCoordinate(mesh)
    mass = fe.assemble(fe.Constant(1.) * fe.dx(domain=mesh))
    inertia = fe.assemble(((sc[0] - L0/2)**2 + (sc[1] - L0/2)**2) * fe.dx(domain=mesh))
    onp.save(f"data/numpy/unit/mass_{case_id}.npy", mass)
    onp.save(f"data/numpy/unit/inertia_{case_id}.npy", inertia)
    return mass, inertia


def DeformationGradient(u):
    I = fe.Identity(u.geometric_dimension())
    return I + fe.grad(u)


def NeoHookeanEnergy(u, shear_mod, bulk_mod):
    F = DeformationGradient(u)
    F = fe.variable(F)
    J = fe.det(F)
    Jinv = J**(-2 / 3)
    I1 = fe.tr(F.T * F)
    energy = ((shear_mod / 2) * (Jinv * (I1 + 1) - 3) +
              (bulk_mod / 2) * (J - 1)**2) 
    first_pk_stress = fe.diff(energy, F)
    return energy, first_pk_stress


def fem_solve(c1, c2, rot_angle_1, rot_angle_2, disp, save_data, save_sols, data_file_name=None):
    L0 = args.L0 
    porosity = args.porosity
    young_modulus = 100
    poisson_ratio = 0.3
    shear_mod = young_modulus / (2 * (1 + poisson_ratio))
    bulk_mod = young_modulus / (3 * (1 - 2*poisson_ratio))

    mesh = build_mesh(c1, c2, 'beam', True)

    V = fe.VectorFunctionSpace(mesh, 'P', 1)
    E = fe.FunctionSpace(mesh, 'DG', 0)
    u = fe.Function(V)
    du = fe.TrialFunction(V)
    v = fe.TestFunction(V)

    class LeftHole(fe.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and x[0] < 0.5 * L0 + 1e-8

    class RightHole(fe.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and x[0] > 1.5 * L0 - 1e-8

    left_hole = LeftHole()
    right_hole = RightHole()

    class Expression(fe.UserExpression):
        def __init__(self, *params):
            # Construction method of base class has to be called first
            super(Expression, self).__init__()
            self.params = params

        def eval(self, values, x):
            disp, rot_angle, center = self.params
            x_old = onp.array(x)
            # x_new = rotate_vector(rot_angle, x_old - center) + center
            rot_mat = angle_to_rot_mat(rot_angle)
            x_new = onp.dot(rot_mat, x_old - center) + center
            u = x_new - x_old
            values[0] = u[0] + disp
            values[1] = u[1]

        def value_shape(self):
            return (2,)

    boundaries = fe.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    boundaries.set_all(0)
    ds = fe.Measure('ds')(subdomain_data=boundaries)
    
    energy_density, PK_stress = NeoHookeanEnergy(u, shear_mod, bulk_mod)
    total_energy = energy_density * fe.dx 

    left_center = onp.array([0.5 * L0, 0.5 * L0])
    right_center = onp.array([1.5 * L0, 0.5 * L0])
    left_wall_condition = Expression(0., rot_angle_1, left_center)
    right_wall_condition = Expression(disp, rot_angle_2, right_center)

    bcs = [fe.DirichletBC(V, left_wall_condition, left_hole),
           fe.DirichletBC(V, right_wall_condition, right_hole)]
  
    dE = fe.derivative(total_energy, u, v)
    jacE = fe.derivative(dE, u, du)
    
    poisson_form = fe.inner(fe.sym(fe.grad(du)), fe.sym(fe.grad(v))) * fe.dx
    rhs = fe.dot(fe.Constant((0., 0.)), v) * fe.dx
    fe.solve(poisson_form == rhs, u, bcs, solver_parameters={'linear_solver': 'gmres'})

    # fe.solve(dE == 0, u, bcs, J=jacE)
    nIters, convergence = DynamicRelaxSolve(dE, u, bcs, jacE)

    if convergence:
        energy_val = float(fe.assemble(energy_density * fe.dx))
        print(f"energy_val = {energy_val}")
        if save_sols:
            e = fe.project(energy_density, E)
            u_vtk_file = fe.File(f'data/pvd/u.pvd')
            u_vtk_file << u
            u.rename("u", "u")
            e_vtk_file = fe.File(f'data/pvd/e.pvd')
            e.rename("e", "e")
            e_vtk_file << e

            xdmf_file = fe.XDMFFile('data/xdmf/u.xdmf')
            xdmf_file.parameters["functions_share_mesh"] = True
            xdmf_file.write(u, 0)
            xdmf_file.write(e, 0)

        if save_data:
            assert data_file_name is not None, f"Need to provide data_file_name, but is None."
            data_point = onp.array([c1, c2, rot_angle_1, rot_angle_2, disp, energy_val])
            now = datetime.datetime.now().strftime('%s')
            onp.save(f'data/numpy/{data_file_name}/{now}.npy', data_point)

    else:
        print(f"dr solver not converged")


def generate_data(cleanup, case_id, n_samples, shape_param, bounds):
    c1, c2 = shape_param
    rot_bound, disp_bound = bounds
    data_file_name = 'energy_' + str(n_samples) + '_' + case_id
    if cleanup:
        print(f"\nDelete all energy data...")
        # shutil.rmtree(f'data/numpy/energy_data', ignore_errors=True)
        numpy_files = glob.glob(f'data/numpy/{data_file_name}/*')
        for f in numpy_files:
            os.remove(f)

    key = jax.random.PRNGKey(0)
    features = jax.random.uniform(key, shape=(n_samples, 3), minval=onp.array([[-rot_bound, -rot_bound, -disp_bound]]), 
        maxval=onp.array([[rot_bound, rot_bound, disp_bound]]))

    for i, (rot_angle_1, rot_angle_2, disp) in enumerate(features):
        print(f"Generate data point {i + 1} out of {n_samples}")
        fem_solve(c1, c2, rot_angle_1=rot_angle_1, rot_angle_2=rot_angle_2, disp=disp, save_data=True, save_sols=False, data_file_name=data_file_name)


def main():
    case_ids = ['poreA', 'poreB']
    shape_params = [(0., 0.), (-0.2, 0.2)]
    bounds = [(1/5*onp.pi, 0.08), (1/5*onp.pi, 0.08)]
    n_samples = 1000
    # for i in range(len(case_ids)):
    generate_data(True, case_ids[1], n_samples, shape_params[1], bounds[1])
    # fem_solve(*shape_params[1], rot_angle_1=1*1/5*onp.pi, rot_angle_2=-1*1/5*onp.pi, disp=-0.08, save_data=False, save_sols=True)
    # for i in range(len(case_ids)):    
    #     compute_mass_inertia(*shape_params[i], case_ids[i])


@walltime
def exp():
    # fem_solve(c1=0., c2=0., rot_angle_1=1*1/5*onp.pi, rot_angle_2=-1*1/5*onp.pi, disp=-0.08, save_data=False, save_sols=True)
    fem_solve(c1=-0.2, c2=0.2, rot_angle_1=1*1/5*onp.pi, rot_angle_2=1*1/5*onp.pi, disp=-0.08, save_data=False, save_sols=True)


if __name__ == '__main__':
    exp()
    # main()
