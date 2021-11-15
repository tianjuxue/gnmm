import fenics as fe
import numpy as np
import shutil
import os
import glob
import mshr
from src.dr import DynamicRelaxSolve
from src.arguments import args
from src.utils import angle_to_rot_mat, rotate_vector, walltime
import datetime
import gin
import jax


def angle_to_rot_mat(theta):
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])


def build_mesh(c1, c2, shape_tag, save_mesh):
    porosity = args.porosity
    L0 = args.L0

    resolution = 30
    n_radial_points = 100

    def coords_fn(theta):
        r0 = L0 * np.sqrt(2 * porosity) / np.sqrt(np.pi * (2 + c1**2 + c2**2))
        return r0 * (1 + c1 * fe.cos(4 * theta) + c2 * fe.cos(8 * theta))

    def build_base_pore():
        # Remark(Tianju): mshr may not work well if shift is not added.
        shift = 0.1
        thetas = [float(i) * 2 * np.pi / n_radial_points + shift for i in range(n_radial_points)]
        radii = [coords_fn(theta) for theta in thetas]
        points = [(rtheta * np.cos(theta), rtheta * np.sin(theta)) for rtheta, theta in zip(radii, thetas)]
        return np.array(points), np.array(radii), np.array(thetas)

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
    else:
        raise ValueError(f"Unknown shape: {shape_tag}")

    if save_mesh:
        mesh_file = fe.File(f'data/pvd/{shape_tag}_mesh.pvd')
        mesh.rename("mesh", "mesh")
        mesh_file << mesh

    return mesh



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


def fem_solve(c1, c2, rot_angle_1, rot_angle_2, disp):
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
            x_old = np.array(x)
            # x_new = rotate_vector(rot_angle, x_old - center) + center
            rot_mat = angle_to_rot_mat(rot_angle)
            x_new = np.dot(rot_mat, x_old - center) + center
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

    left_center = np.array([0.5 * L0, 0.5 * L0])
    right_center = np.array([1.5 * L0, 0.5 * L0])

    left_wall_condition = Expression(0., rot_angle_1, left_center)
    right_wall_condition = Expression(disp, rot_angle_2, right_center)

    # left_wall_condition = fe.Constant((0., 0.))
    # right_wall_condition = fe.Constant((0., 0.))

    bcs = [fe.DirichletBC(V, left_wall_condition, left_hole),
           fe.DirichletBC(V, right_wall_condition, right_hole)]
  
    dE = fe.derivative(total_energy, u, v)
    jacE = fe.derivative(dE, u, du)
    
    poisson_form = fe.inner(fe.grad(du), fe.grad(v)) * fe.dx


    nodal_values = np.array(u.vector()[:])
    print(len(nodal_values))

    rhs = fe.dot(fe.Constant((0., 0.)), v) * fe.dx

    solve(poisson_form == rhs, u, bcs, solver_parameters={'linear_solver': 'gmres'})


solve = walltime(fe.solve)
 

def exp():
    fem_solve(c1=0., c2=0., rot_angle_1=0*1/5*np.pi, rot_angle_2=0*1/5*np.pi, disp=0.01)
 

if __name__ == '__main__':
    exp()
 
