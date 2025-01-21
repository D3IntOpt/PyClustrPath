import numpy as np
import torch
import argparse
import scipy.io as sio
import sys
import os

from pyclustrpath import ConvexClusterSolver, DataProcessor
from pyclustrpath.utils import str2bool, visualize_clustering_results

from pyclustrpath import gv

# device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# torch.cuda.empty_cache()

# device = 'cpu'

def get_args():
    # define the argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--solver', default='ssnal', type=str, choices=['admm', 'ama', 'fast_ama', 'ssnal', 'pdhg'],
                        help='The solver to use: admm, fast_ama, ssnal, or pdhg')
    parser.add_argument('--k_neighbor', default=10, type=int, help='Number of neighbors to consider')
    parser.add_argument('--stop_tol', default=1e-6, type=float, help='Stopping tolerance for the solver')
    parser.add_argument('--max_iter', default=10000, type=int, help='Maximum number of iterations for the solver')

    parser.add_argument('--use_kkt', default=False, type=str2bool, help='Whether to use the KKT condition to stop the solver')
    # parser.add_argument('--admm_sub_method', default='cholesky', type=str, choices=['direct', 'cholesky', 'svd', 'cg'],
    #                     help='The method to solve the ADMM subproblem')
    parser.add_argument('--sigma', default=1.0, type=float, help='The parameter for the ADMM solver')

    # add a new argument to choose the data
    parser.add_argument('--data_type', default=0, type=int, help='The type of data to use: [0 : load sample data, 1 : generate data, 2 : load user data]')

    parser.add_argument('--data', default='lung', type=str,
                        help='The dataset to use: libras6, libras, COIL20, lung, MNIST_test')
    parser.add_argument('--device', default=device, type=str, help='The device to use: cpu or cuda')

    return parser.parse_args()


def main():
    # get the arguments
    args = get_args()
    gv._init(args)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'                             # Set the device

    data = sio.loadmat('data/realdata/lung.mat')                                        # Load the LUNG data
    tensor_data = torch.Tensor(data['X']).double()
    data_label = np.squeeze(data['col_assign'])

    gamma_upper = 0.93                                                                  # Set the list of gamma
    gamma_lower = 0.057
    gamma_num = 100
    gamma_step = (gamma_upper - gamma_lower) / (gamma_num - 1)
    gamma_list = np.arange(gamma_upper, gamma_lower - gamma_step, -gamma_step)

    from pyclustrpath import DataProcessor
    data_processor = DataProcessor(X=tensor_data, k=args.k_neighbor, device=device)     # Process the data

    from pyclustrpath import ConvexClusterSolver
    solver = ConvexClusterSolver(method=args.solver, gamma_list=gamma_list,             # Load the model
                                 stop_tol=args.stop_tol, max_iter=args.max_iter,
                                  use_kkt=args.use_kkt, device=device)

    solutions = solver.solve(data_processor)                                            # Solve the problems

    visualize_clustering_results(data=tensor_data, solution=solutions,                  # Visualize the clustering path
                                 gamma_list=gamma_list, label=data_label)



if __name__ == "__main__":
    main()
