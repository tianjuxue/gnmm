import fenics as fe
import numpy as onp
import shutil
import os
import glob
import mshr
import datetime
import gin
from src.dr import DynamicRelaxSolve
from src.arguments import args
from src.utils import walltime
from src.fem_commons import *


def angle_to_rot_mat(theta):
    return onp.array([[onp.cos(theta), -onp.sin(theta)], [onp.sin(theta), onp.cos(theta)]])


def compute_mass_inertia(c1, c2, case_id):
    L0 = args.L0
    args.shape_tag = 'unit'
    mesh = build_mesh(c1, c2, True)
    sc = fe.SpatialCoordinate(mesh)
    mass = fe.assemble(fe.Constant(1.) * fe.dx(domain=mesh))
    inertia = fe.assemble(((sc[0] - L0/2)**2 + (sc[1] - L0/2)**2) * fe.dx(domain=mesh))
    onp.save(f"data/numpy/{args.shape_tag}/mass_{case_id}.npy", mass)
    onp.save(f"data/numpy/{args.shape_tag}/inertia_{case_id}.npy", inertia)
    return mass, inertia


def fem_solve(c1, c2, rot_angle_1, rot_angle_2, disp, save_data, save_sols):
    L0 = args.L0 
    args.shape_tag = 'beam'
    mesh = build_mesh(c1, c2, True)

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
    
    energy_density, PK_stress = NeoHookeanEnergy(u)
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
    fe.solve(poisson_form == rhs, u, bcs)

    # fe.solve(dE == 0, u, bcs, J=jacE)
    nIters, convergence = DynamicRelaxSolve(dE, u, bcs, jacE)

    if convergence:
        energy_val = float(fe.assemble(energy_density * fe.dx))
        print(f"energy_val = {energy_val}")
        if save_sols:
            e = fe.project(energy_density, E)
            u_vtk_file = fe.File(f'data/pvd/sols/{args.shape_tag}/{args.case_id}/u.pvd')
            u_vtk_file << u
            u.rename("u", "u")
            e_vtk_file = fe.File(f'data/pvd/sols/{args.shape_tag}/{args.case_id}/e.pvd')
            e.rename("e", "e")
            e_vtk_file << e

            xdmf_file = fe.XDMFFile(f'data/xdmf/{args.shape_tag}/{args.case_id}/u.xdmf')
            xdmf_file.parameters["functions_share_mesh"] = True
            xdmf_file.write(u, 0)
            xdmf_file.write(e, 0)

        if save_data:
            data_point = onp.array([c1, c2, rot_angle_1, rot_angle_2, disp, energy_val])
            now = datetime.datetime.now().strftime('%s')
            onp.save(args.path_training_data + f'/{now}.npy', data_point)
    else:
        print(f"dr solver not converged")


def generate_data(shape_param, bounds):
    args.path_training_data = f'data/numpy/{args.shape_tag}/energy_sample_{args.num_samples}_resolution_{args.resolution}_{args.case_id}'
    c1, c2 = shape_param
    rot_bound, disp_bound = bounds
    print(f"\nDelete all existing energy data...")
    shutil.rmtree(args.path_training_data, ignore_errors=True)
    # numpy_files = glob.glob(args.path_training_data + f'/*')
    # for f in numpy_files:
    #     os.remove(f)
    os.mkdir(args.path_training_data)

    # key = jax.random.PRNGKey(0)
    # features = jax.random.uniform(key, shape=(args.num_samples, 3), minval=onp.array([[-rot_bound, -rot_bound, -disp_bound]]), 
    #     maxval=onp.array([[rot_bound, rot_bound, disp_bound]]))
    features = onp.random.uniform(low=(-rot_bound, -rot_bound, -disp_bound), high=(rot_bound, rot_bound, disp_bound), size=(args.num_samples, 3))

    for i, (rot_angle_1, rot_angle_2, disp) in enumerate(features):
        print(f"Generate data point {i + 1} out of {args.num_samples}")
        fem_solve(c1, c2, rot_angle_1=rot_angle_1, rot_angle_2=rot_angle_2, disp=disp, save_data=True, save_sols=False)


# def main():
#     args.shape_tag = 'beam' 
#     case_ids = ['poreA', 'poreB', 'poreC', 'poreD', 'poreE']
#     shape_params = [(0., 0.), (-0.05, 0.), (-0.1, 0.), (-0.15, 0.), (-0.2, 0.)]
#     bounds = onp.array([1/5*onp.pi, 0.1])
#     args.num_samples = 1000
#     for i in range(len(case_ids)):
#         args.case_id = case_ids[i]
#         generate_data(shape_params[i], bounds)

#     for i in range(len(case_ids)):    
#         compute_mass_inertia(*shape_params[i], case_ids[i])

#     onp.save(f'data/numpy/{args.shape_tag}/bounds.npy', bounds)



def main():
    args.shape_tag = 'beam' 
    args.resolution = 30
    args.num_samples = 1000
    case_ids = ['poreA', 'poreB', 'poreC', 'poreD', 'poreE', 'poreF']
    shape_params = [(0., 0.), (-0.05, 0.), (-0.1, 0.), (-0.15, 0.), (-0.2, 0.), (-0.2, 0.2)]
    bounds = onp.array([1/5*onp.pi, 0.1])

    for i in range(len(case_ids)):
        args.case_id = case_ids[i]
        generate_data(shape_params[i], bounds)

    for i in range(len(case_ids)):    
        compute_mass_inertia(*shape_params[i], case_ids[i])

    onp.save(f'data/numpy/{args.shape_tag}/bounds.npy', bounds)


@walltime
def exp():
    args.case_id = 'poreX'
    args.porosity = 0.6
    fem_solve(c1=-0., c2=0., rot_angle_1=1*1/5*onp.pi, rot_angle_2=1*1/5*onp.pi, disp=-0.1, save_data=False, save_sols=True)
    # fem_solve(c1=-0.2, c2=0.2, rot_angle_1=-1*1/5*onp.pi, rot_angle_2=1*1/5*onp.pi, disp=-0.1, save_data=False, save_sols=True)


if __name__ == '__main__':
    # exp()
    main()
