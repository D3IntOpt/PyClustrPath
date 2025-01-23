import torch
from sklearn.datasets import make_blobs
import argparse

from cholespy import CholeskySolverD, MatrixType
# ref:https://github.com/rgl-epfl/cholespy

import cusparse_extension


def str2bool(v):
    """
    Convert string to boolean.
    :param v:
    :return:
    """
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def proj_l2(proj_input, weightVec):
    """
    Projects each column of the input matrix onto the L2 ball of a given radius.

    Parameters:
    - proj_input: Input matrix with dimensions d*n.
    - weightVec: A vector of radii, dimension 1*n, specifying the L2 norm target for each column.

    Returns:
    - output: The projected matrix.
    """
    # Check dimensions
    d, n = proj_input.size()
    output = proj_input.clone().detach()
    # output = proj_input
    if n != weightVec.numel():
        raise ValueError('Dimensions not agree.')

    weightVec = weightVec.squeeze(0)
    # Compute L2 norms of each column
    norm_input = torch.sqrt(torch.sum(proj_input ** 2, dim=0))
    # Identify columns that need scaling

    idx = norm_input > weightVec
    num = torch.sum(idx)
    if idx.any():
        scaling_factors = weightVec[idx] / norm_input[idx]
        output[:, idx] = proj_input[:, idx] * scaling_factors

    return output
def prox_l2(x, weightVec):
    """
    Proximal mapping operator for the L2 norm.

    Parameters:
    - x: Input tensor.
    - weightVec: A vector of radii, specifying the L2 norm target for each column.

    Returns:
    - output: The proximal operator,  the number of columns that were projected, and the L2 norm of each column.
    """
    # Check dimensions
    d, n = x.size()
    if n != weightVec.numel():
        raise ValueError('Dimensions not agree.')

    weightVec = weightVec.squeeze(0)
    # Compute L2 norms of each column
    norm_input = torch.sqrt(torch.sum(x ** 2, dim=0))
    # Identify columns that need scaling
    idx = norm_input > weightVec
    num_idx = torch.sum(idx)

    out = torch.zeros_like(x)
    if idx.any():
        scaling_factors = weightVec[idx] / norm_input[idx]
        out[:, idx] = x[:, idx] * (1.0 - scaling_factors)
    return out, idx, norm_input


def generate_unbalanced_gauss_data(d=10, n=10000, **params):
    centers = 4
    X, y = make_blobs(n_samples=n, n_features=d, centers=centers, cluster_std = 0.2, random_state=0)
    # data = torch.tensor(X)
    # print(data.shape)
    return X # dim:(n, d)

def sparse_cholesky_solve(solver, b, x=None):
    """
    Solve the linear system Ax = b using the Cholesky factorization of A.
    :param solver:
    :param b:
    :param x:
    :return:
    """
    pre_con = 1
    n, d = b.size()
    if x is None:
        x = torch.zeros_like(b)
    if pre_con:
        if b.size(1) <= 128 or b.device.type == 'cpu':
            b = b.contiguous()
            x = x.contiguous()
            solver.solve(b, x)
        else:
            # b = b.contiguous()
            block_size = 128
            X_block = torch.zeros((n, block_size), dtype=torch.float64, device=b.device)
            for i in range(0, d, block_size):
                b_block = b[:, i:i + block_size].contiguous()
                if b_block.shape[1] < 128:
                    X_block = X_block[:, :b_block.shape[1]].contiguous()
                solver.solve(b_block, X_block)
                # x[:, i:i + block_size] = X_block.clone()
                x[:, i:i + block_size] = X_block
    else:
        x = b
    return x

def sparse_complete_cholesky(A_sparse):

    """
    Complete Cholesky factorization for sparse matrix.
    :param A_sparse:
    :param b:
    :return:
    """
    # x = torch.zeros_like(b)
    A_num_rows = A_sparse.size(0)
    A_num_cols = A_sparse.size(1)
    nnz = A_sparse._nnz()
    # judge the storage format of A_sparse
    if A_sparse.layout == torch.sparse_csr:
        RowInd = A_sparse.crow_indices().int()
        ColInd = A_sparse.col_indices().int()
        Val = A_sparse.values()
        matrix_type = MatrixType.CSR

    elif A_sparse.layout == torch.sparse_coo:
        RowInd = A_sparse.indices()[0].int()
        ColInd = A_sparse.indices()[1].int()
        Val = A_sparse.values()
        matrix_type = MatrixType.COO
    else:
        raise ValueError('The input matrix should be in CSR or COO format.')
    solver = CholeskySolverD(A_num_rows, RowInd, ColInd, Val, matrix_type)

    return solver

def incomplete_cholesky_cusparse(A):
    """
    Incomplete Cholesky factorization using cuSPARSE.

    :param A: Input sparse matrix with CSR form.
    :return: The lower triangular sparse matrix L with CSR form.
    """

    csrRowInd = A.crow_indices().int()
    csrColInd = A.col_indices().int()
    csrVal = A.values()
    A_num_rows = A.size(0)
    A_num_cols = A.size(1)
    nnz = A._nnz()

    row_ptr_tensor, col_ind_tensor, values_tensor = cusparse_extension.csr_ichol(csrRowInd, csrColInd, csrVal, A_num_rows, A_num_cols, nnz)
    L = torch.sparse_csr_tensor(row_ptr_tensor, col_ind_tensor, values_tensor)
    # L_dense = L.to_dense().cpu().numpy()
    return L

def pre_conjugate_gradient_batched_for_ADMM(A, b, Ax, M_inv, x=None, tolerance=1e-6, max_iterations=1000):
    """
    Preconditioned conjugate gradient method for solving the linear system Ax = b.
    :param A: The matrix A.
    :param b: The vector b.
    :param Ax: The vector Ax.
    :param M_inv: The inverse of the preconditioner.
    :param x: The initial guess.
    :param tolerance: The tolerance for convergence.
    :param max_iterations: The maximum number of iterations.
    :return: The solution x, the vector Ax, and the number of iterations.
    """
    if x is None:
        x = torch.zeros_like(b)
    r = b - Ax
    z = torch.mm(r, M_inv)
    p = z.clone()
    res_old = torch.sum(r * z)
    for i in range(max_iterations):
        Ap = torch.sparse.mm(p, A)

        alpha = res_old / torch.sum(p * Ap)
        x = x + p * alpha
        r = r - Ap * alpha
        Ax = Ax + Ap * alpha

        z = torch.mm(r, M_inv)
        res_new = torch.sum(r * z)

        if res_new < tolerance:
            break  # successful convergence
        beta = (res_new / res_old)
        p = z + beta * p
        res_old = res_new
    return x, Ax, i+1


def conjugate_gradient_batched_for_ADMM(A, b, Ax, x=None,tolerance=1e-6, max_iterations=1000):
    """
    Conjugate gradient method for solving the linear system Ax = b.
    :param A: The matrix A.
    :param b: The vector b.
    :param Ax: The vector Ax.
    :param x: The initial guess.
    :param tolerance: The tolerance for convergence.
    :param max_iterations: The maximum number of iterations.
    :return: The solution x, the vector Ax, and the number of iterations.
    """
    if x is None:
        x = torch.zeros_like(b)
    r = b - Ax
    p = r.clone()
    res_old = torch.norm(r, p='fro')
    for i in range(max_iterations):
        Ap = torch.sparse.mm(p, A)
        alpha = res_old ** 2 / torch.sum(p * Ap)
        x = x + p * alpha
        r = r - Ap * alpha
        Ax = Ax + Ap * alpha
        res_new = torch.norm(r, p='fro')

        if res_new < tolerance:
            break  # successful convergence
        beta = (res_new / res_old) ** 2
        p = r + beta * p
        res_old = res_new

    return x, Ax, i+1