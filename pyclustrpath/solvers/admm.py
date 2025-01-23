import torch
import time
import pyclustrpath.global_variables as gv
from .base import SolverBase
from pyclustrpath.data_processor import DataProcessor
from pyclustrpath.utils import *

class ADMM(SolverBase):
    def __init__(self, gamma_vec_list, device, stop_tol=1e-5, max_iter=20000, gamma=1.618, sigma=1.0, use_kkt=False, admm_sub_method='cholesky', **params):
        super().__init__(gamma_vec_list=gamma_vec_list, stop_tol=stop_tol, max_iter=max_iter, device=device, use_kkt=use_kkt, **params)
        self.gamma = gamma
        self.sigma = sigma
        self.sigma_init = sigma
        self.sub_method = admm_sub_method
        self.params = params
        self.to_device(device)

    def solve(self, data: DataProcessor, **params):
        n, d, w = data.n, data.d, data.num_weights
        gv.logger.info('subproblem method: %s', self.sub_method)
        gv.logger.info(
            ' Gamma |   Iter   |   [ pinfeas      dinfeas    relgap ]  |   primobj   |   '
            'dualobj   |   time   |   sigma   |  [ cg_iter  row_prox ]')
        X_prev = data.X.clone().detach()
        Y_prev = torch.zeros((d, w), dtype=torch.float64, device=self.device)
        Z_prev = torch.zeros((d, w), dtype=torch.float64, device=self.device)
        admm_time = 0
        admm_iter = 0
        result_list = []
        solutions = torch.zeros((len(self.gamma_vec_list), d, n), dtype=torch.float64, device=self.device)
        for gamma_vec in self.gamma_vec_list:
            admm_iter += 1
            X_admm, Y_admm, Z_admm, info_admm, runhist_admm = self.solve_iter(data, X_prev, Y_prev, Z_prev, gamma_vec=gamma_vec)
            X_prev = X_admm
            Y_prev = Y_admm
            Z_prev = Z_admm
            flag_save_solution = info_admm['solved']

            if flag_save_solution:
                sol_X = X_admm
                sol_Z = Z_admm
                solutions[admm_iter - 1] = sol_X
                # break
            result_list.append([gamma_vec, info_admm['prim_obj'].cpu().item(), info_admm['dual_obj'].cpu().item(),
                                info_admm['rel_gap'].cpu().item(), info_admm['iter'], info_admm['time']])
            admm_time += info_admm['time']

        gv.logger.info('Total time taken by ADMM = %5.4f s', admm_time)

        solutions = solutions * data.max_dist
        return solutions



    def solve_iter(self, data: DataProcessor, X: torch.Tensor, Y: torch.Tensor, Z: torch.Tensor, gamma_vec, **params):
        '''
        solve the clustering problem using ADMM algorithm
        :param data:  DataProcessor object
        :param X: dim: (d, n)
        :param Y: dim: (d, num_weights)
        :param Z: dim: (d, num_weights)
        :param params:
        :return:  X, Y, Z, result, runhist
        '''

        # complete ADMM algorithm
        self.gamma_vec = gamma_vec
        self.sigma = self.sigma_init
        sig_fix = False
        use_infeasorg = 0

        if self.print_yes:
            gv.logger.info('--------------ADMM for solving Clustering with tau = %6.4f and gamma = %6.4f--------------', self.gamma, self.gamma_vec)
            gv.logger.info(' problem size: d = %3.0f, n = %3.0f, number of weights = %3.0f', data.d, data.n,
                           data.num_weights)
            gv.logger.info('subproblem method: %s', self.sub_method)

            gv.logger.info(
                '   Iter   |   [ pinfeas      dinfeas ] | [  pinforg      dinforg      relgap ]  |   primobj   |   '
                'dualobj   |   time   |   sigma   |   gamma   |  [ cg_iter  row_prox ]')

        weightVec = self.gamma_vec * data.weightVector
        time_s = time.perf_counter()

        # initialize the primal and dual variables
        # X->xi, Y->y, Z->x
        a_x = data.a_map(X)
        ata_x = data.at_map(a_x)

        at_z = data.at_map(Z)
        r_prim = a_x - Y
        proj_z = proj_l2(Z, weightVec)
        r_dual = Z - proj_z

        feas_prim = torch.norm(r_prim, p='fro') / (1 + torch.norm(Y, p='fro'))
        feas_dual = torch.norm(r_dual, p='fro') / (1 + torch.norm(Z, p='fro'))
        feas_prim_prev = feas_prim
        feas_dual_prev = feas_dual

        feas_max = max(feas_prim, feas_dual)

        break_yes = 0
        prim_win = 0
        dual_win = 0
        msg = ' '

        ###############
        # Use Incomplete Preconditional Conjugate Gradient Method(PCG: 1) or Conjugate Gradient Method(CG: 0)
        use_ichol = 1
        ###############

        # create a runhist object to record the primal and dual infeasibility for each iteration
        runhist = gv.runhist(self.max_iter)

        # create a sparse matrix C = I + sigma * ATA
        I_dense = torch.eye(data.n, dtype=torch.float64, device=data.device)
        if gv.use_coo:
            I = I_dense.to_sparse_coo()
        else:
            I = I_dense.to_sparse_csr()


        C = self.sigma * data.ata_mat_dense + I
        C_sparse = I + self.sigma * data.ata_mat
        if gv.use_coo:
            C_sparse = C_sparse.coalesce()

        if self.sub_method == 'cholesky':
            chol_solver = sparse_complete_cholesky(C_sparse)
        elif self.sub_method == 'cg':
            x_mult_c = X + self.sigma * ata_x
            if use_ichol == 1:  # incomplete cholesky
                L_ichol = incomplete_cholesky_cusparse(C_sparse).to_dense()
                y_ichol = torch.linalg.solve_triangular(L_ichol, I_dense, upper=False, left=True)
                C_inv = torch.linalg.solve_triangular(L_ichol.T, y_ichol, upper=True, left=True)


        cg_iter_total = 0
        for iter in range(self.max_iter):
            iter = iter + 1
            eta, eta_prev = 0, 0

            # ————————————————update X————————————————
            x_prev = X
            rhsxi = data.X - at_z + self.sigma * data.at_map(Y)
            if self.sub_method == 'direct':
                X = torch.linalg.solve(C, rhsxi, left=False)
            elif self.sub_method == 'cholesky':
                sparse_cholesky_solve(chol_solver, rhsxi.T, X.T)
            elif self.sub_method == 'svd':
                U, S, Vh = torch.linalg.svd(data.ata_mat_dense)
                SS = 1 + self.sigma * S
                X = rhsxi @ U @ torch.diag(1 / SS) @ U.T
            elif self.sub_method == 'cg':
                x_mult_c_prev = x_mult_c.clone().detach()
                cg_stoptol = max(0.9 * self.stop_tol, min(1.0 / iter ** 1.1, 0.9 * feas_max))
                cg_iter_max = min(500, int(np.sqrt(data.n * data.d)))
                if use_ichol == 1: # incomplete cholesky and pcg
                    X, x_mult_c, cg_iter = pre_conjugate_gradient_batched_for_ADMM(C_sparse, rhsxi, x_mult_c_prev, C_inv, x_prev, cg_stoptol, cg_iter_max, self.sigma)
                else: #cg
                    X, x_mult_c, cg_iter = conjugate_gradient_batched_for_ADMM(C_sparse, rhsxi, x_mult_c_prev,
                                                                               x_prev,
                                                                               cg_stoptol, cg_iter_max, self.sigma)
                cg_iter_total += cg_iter

            else:
                raise NotImplementedError("This method should be implemented by subclasses.")

            a_x = data.a_map(X)

            # ————————————————update Y————————————————
            y_input = a_x + (1 / self.sigma) * Z
            Y, rr, _ = prox_l2(y_input, weightVec / self.sigma)
            sum_rr = torch.sum(rr)

            # ————————————————update multiplier Z————————————————
            r_prim = a_x - Y
            Z = Z + self.gamma * self.sigma * r_prim
            at_z = data.at_map(Z)

            # ————————————————calculate the primal and dual infeasibility————————————————
            r_prim_norm = torch.norm(r_prim, p='fro')
            y_norm = torch.norm(Y, p='fro')
            feas_prim = r_prim_norm / (1 + y_norm)
            feas_prim_prev = feas_prim
            proj_z = proj_l2(Z, weightVec)
            r_dual = Z - proj_z

            feas_dual = torch.norm(r_dual, p='fro') / (1 + torch.norm(Z, p='fro'))
            feas_dual_prev = feas_dual
            feas_max = max(feas_prim, feas_dual)
            feas_max_prev = feas_max

            # record the primal and dual infeasibility for current iteration 'iter' using runhist
            runhist['feas_prim'].append(feas_prim)
            runhist['feas_dual'].append(feas_dual)
            runhist['feas_prim_prev'].append(feas_prim_prev)
            runhist['feas_dual_prev'].append(feas_dual_prev)
            runhist['feas_max'].append(feas_max)
            runhist['feas_max_prev'].append(feas_max_prev)
            runhist['sigma'].append(self.sigma)

            if self.sub_method == 'cg':
                runhist['psqmrxiiter'].append(cg_iter)

            # ————————————————check for termination————————————————
            if feas_max < self.stop_tol:
                if self.use_kkt:
                    # check the KKT condition
                    grad = at_z + X - data.X
                    eta = torch.norm(grad, p='fro') / (1 + torch.norm(X, p='fro'))
                    y_plus_z = Y + Z
                    y_plus_z_prox, _, _ = prox_l2(y_plus_z, weightVec)
                    res_vec = Y - y_plus_z_prox
                    eta = eta + torch.norm(res_vec, p='fro') / (1 + torch.norm(Y, p='fro'))
                    eta_prev = eta

                    obj_prim = 0.5 * torch.norm(X - data.X, p='fro') ** 2 + torch.matmul(weightVec, torch.sqrt(
                        torch.sum(a_x ** 2, dim=0)))
                    obj_dual = -0.5 * torch.norm(at_z, p='fro') ** 2 + torch.sum(data.X * at_z)
                    rel_gap = abs(obj_prim - obj_dual) / (1 + abs(obj_prim) + abs(obj_dual))
                else:
                    obj_prim = 0.5 * torch.norm(X - data.X, p='fro') ** 2 + torch.matmul(weightVec, torch.sqrt(
                        torch.sum(a_x ** 2, dim=0)))
                    obj_dual = -0.5 * torch.norm(at_z, p='fro') ** 2 + torch.sum(data.X * at_z)
                    rel_gap = abs(obj_prim - obj_dual) / (1 + abs(obj_prim) + abs(obj_dual))
                    eta = rel_gap
                    eta_prev = eta

                if eta < self.stop_tol:
                    break_yes = 1
                    msg = 'converged'

            time_1 = time.perf_counter()

            # ————————————————print result————————————————
            if time_1 - time_s > 7 * 3600:
                break_yes = 777
                msg = 'time out'
            if iter <= 200:
                print_iter = 20
            elif iter <= 2000:
                print_iter = 100
            else:
                print_iter = 200

            if (iter % print_iter == 1) or (iter == self.max_iter) or (break_yes > 0) or (iter < 20):
                if feas_max > self.stop_tol:
                    obj_prim = 0.5 * torch.norm(X - data.X, p='fro') ** 2 + torch.matmul(weightVec, torch.sqrt(
                        torch.sum(a_x ** 2, dim=0)))
                    obj_dual = -0.5 * torch.norm(at_z, p='fro') ** 2 + torch.sum(data.X * at_z)
                    rel_gap = (obj_prim - obj_dual) / (1 + abs(obj_prim) + abs(obj_dual))

                time_2 = time.perf_counter()
                time_iter = time_2 - time_s

                if self.print_yes and eta == 0:
                    gv.logger.info(
                        '  %5.0d   |  [ %-5.4e  %-5.4e] | [%- 5.4e  %- 5.4e  %-5.4e] | %- 5.4e | %- 5.4e |  %5.3f s | %- 3.2e | %- 5.4f |  [ %6.0f   %6.0f   ]',
                        iter, feas_prim, feas_dual, feas_prim_prev, feas_dual_prev, rel_gap, obj_prim, obj_dual,
                        time_iter, self.sigma, self.gamma, runhist['psqmrxiiter'][-1], sum_rr)
                elif self.print_yes and eta > 0:
                    gv.logger.info(
                        '  %5.0d   |  [ %-5.4e  %-5.4e] | [%- 5.4e  %- 5.4e  %-5.4e] | %- 5.4e | %- 5.4e |  %5.3f s | %- 5.4e | %- 5.4f |  [ %6.0f   %6.0f   ] |'
                        ' [eta = %3.2e,  etaorg = %3.2e]',
                        iter, feas_prim, feas_dual, feas_prim_prev, feas_dual_prev, rel_gap, obj_prim, obj_dual,
                        time_iter, self.sigma, self.gamma, runhist['psqmrxiiter'][-1], sum_rr, eta, eta_prev)

                if iter % (5 * print_iter) == 1:
                    z_norm = torch.norm(Z, p='fro')
                    a_x_norm = torch.norm(a_x, p='fro')
                    y_norm = torch.norm(Y, p='fro')
                    if self.print_yes:
                        gv.logger.info('[ Z_norm = %5.4e, ATx_norm = %5.4e, Y_norm = %5.4e ]', z_norm, a_x_norm, y_norm)

                runhist['obj_prim'].append(obj_prim)
                runhist['obj_dual'].append(obj_dual)
                runhist['rel_gap'].append(rel_gap)
                runhist['time'].append(time_iter)

            if break_yes > 0:
                if self.print_yes:
                    gv.logger.info('break_yes = %d, %s', break_yes, msg)
                break

            # ————————————————update sigma————————————————
            if feas_max < 5 * self.stop_tol:
                use_infeasorg = 1
            if use_infeasorg == 1:
                feasratio = feas_prim_prev / feas_max_prev
            else:
                feasratio = feas_prim / feas_dual

            if feasratio < 1:
                prim_win += 1
            else:
                dual_win += 1

            sigma_update_iter = self.sigma_fun(iter)
            sigma_scale = 2
            sigma_prev = self.sigma

            if (not sig_fix) and (iter % sigma_update_iter == 0):
                sigma_max = 1e3
                sigma_min = 1e-4
                if iter <= 2500:
                    if prim_win > max(1, 1.2 * dual_win):
                        prim_win = 0
                        self.sigma = max(sigma_min, self.sigma / sigma_scale)
                    elif dual_win > max(1, 1.2 * prim_win):
                        dual_win = 0
                        self.sigma = min(sigma_max, self.sigma * sigma_scale)

            # if sigma has changed
            if sigma_prev != self.sigma:
                if self.sub_method == 'cholesky':
                    if gv.use_coo:
                        C_sparse = (C_sparse - I) / sigma_prev * self.sigma + I
                        C_sparse = C_sparse.coalesce()
                        chol_solver = sparse_complete_cholesky(C_sparse)
                        # prof.step()
                    else:
                        C_sparse = (C_sparse + (-1) * I) * (1 / sigma_prev) * self.sigma + I
                        chol_solver = sparse_complete_cholesky(C_sparse)
                        # prof.step()
                elif self.sub_method == 'cg':
                    # C_sparse = C.to_sparse()
                    if use_ichol == 1:
                        if gv.use_coo:
                            C_sparse = (C_sparse - I) / sigma_prev * self.sigma + I
                            C_sparse = C_sparse.coalesce()
                            C_inv = torch.linalg.inv(C_sparse.to_dense()) # COO format incompelete chol to be realized
                        else:
                            C_sparse = (C_sparse + (-1) * I) * (1 / sigma_prev) * self.sigma + I
                            L_ichol = incomplete_cholesky_cusparse(C_sparse).to_dense()
                            y_ichol = torch.linalg.solve_triangular(L_ichol, I_dense, upper=False, left=True)
                            C_inv = torch.linalg.solve_triangular(L_ichol.T, y_ichol, upper=True, left=True)

                    x_mult_c = (x_mult_c - X) / sigma_prev * self.sigma + X

        # record original variables
        if iter == self.max_iter:
            msg = 'maximum iteration reached!'
            obj_prim = 0.5 * torch.norm(X - data.X, p='fro') ** 2 + torch.matmul(weightVec, torch.sqrt(
                torch.sum(a_x ** 2, dim=0)))
            obj_dual = -0.5 * torch.norm(at_z, p='fro') ** 2 + torch.sum(data.X * at_z)
            rel_gap = abs(obj_prim - obj_dual) / (1 + abs(obj_prim) + abs(obj_dual))

            if self.use_kkt:
                grad = at_z + X - data.X
                eta = torch.norm(grad, p='fro') / (1 + torch.norm(X, p='fro'))
                y_plus_z = Y + Z
                y_plus_z_prox, _, _ = prox_l2(y_plus_z, weightVec)
                res_vec = Y - y_plus_z_prox
                eta = eta + torch.norm(res_vec, p='fro') / (1 + torch.norm(Y, p='fro'))
                eta_prev = eta
            else:
                eta = rel_gap
                eta_prev = eta
        if self.print_yes:
            gv.logger.info('msg = %s', msg)
            gv.logger.info('--------------------------------------------------------------')
            gv.logger.info('number iter = %2.0d', iter)
            gv.logger.info('time = %3.2f', time_iter)
            gv.logger.info('time per iter = %5.4f', time_iter / iter)
            # gv.logger.info('cputime = %3.2f', time_iter)
            gv.logger.info('primobj = %9.8e, dualobj = %9.8e, relgap = %3.2e', obj_prim, obj_dual, rel_gap)
            gv.logger.info('primfeasorg = %3.2e, dualfeasorg = %3.2e', feas_prim_prev, feas_dual_prev)
            gv.logger.info('Total CG number = %3.0d, CG per iter = %3.1f', cg_iter_total, cg_iter_total / iter)
            gv.logger.info('eta = %3.2e, etaorg = %3.2e', eta, eta_prev)
            gv.logger.info('min(Z) = %3.2e, max(Z) = %3.2e', torch.min(Z), torch.max(Z))
            gv.logger.info('--------------------------------------------------------------')

        if gv.solver_method == 'admm':
            gv.logger.info(
                '%5.4f  %5.0d   |  [ %-5.4e  %-5.4e  %-5.4e] | %- 7.6e | %- 7.6e |  %5.4f s | %- 3.2e |  [ %6.2f   %6.0f   ]',
                self.gamma_vec, iter, feas_prim, feas_dual, rel_gap, obj_prim, obj_dual,
                time_iter, self.sigma, cg_iter_total/iter, sum_rr)

        if iter < self.max_iter:
            solved = True
        else:
            solved = False
        abs_gap = abs(obj_prim - obj_dual)
        result = {'iter': iter, 'dual_obj': obj_dual, 'prim_obj': obj_prim, 'rel_gap': rel_gap, 'time': time_iter,
                  'eta': eta, 'cg_iter_total': cg_iter_total, 'feas_prim': feas_prim, 'feas_dual': feas_dual,
                  'solved': solved, 'sigma': self.sigma, 'sum_rr': sum_rr, 'abs_gap': abs_gap}

        return X, Y, Z, result, runhist

    def sigma_fun(self, iter):
        if iter < 30:
            sigma_update_iter = 3
        elif iter < 60:
            sigma_update_iter = 6
        elif iter < 120:
            sigma_update_iter = 12
        elif iter < 250:
            sigma_update_iter = 25
        elif iter < 500:
            sigma_update_iter = 50
        else:
            sigma_update_iter = 100
        return sigma_update_iter