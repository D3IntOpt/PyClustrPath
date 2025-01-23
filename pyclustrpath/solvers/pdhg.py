import torch
import time
import pyclustrpath.global_variables as gv
from .base import SolverBase
from pyclustrpath import DataProcessor
from pyclustrpath.utils import *



class PDHG(SolverBase):
    '''
    Primal-Dual Hybrid Gradient algorithm
    '''

    def solve(self, data):
        pdhg_iter = 0
        result_list = []
        pdhg_time = 0
        X_prev = data.X.clone().detach()
        Z_prev = data.a_map(data.X)
        Z_prev = proj_l2(Z_prev, self.gamma_vec_list[0] * data.weightVector)
        solutions = torch.zeros((len(self.gamma_vec_list), data.d, data.n), dtype=torch.float64, device=self.device)
        gv.logger.info('\t Gamma \t|\t Iter \t|\t   Primobj   \t|\t   Dualobj   \t|\t Relative gap \t|\t Time(s) \t|')

        for gamma_vec in self.gamma_vec_list:
            pdhg_iter += 1
            X_pdhg, Z_pdhg, info_pdhg = self.solve_iter(data, X_prev, Z_prev, gamma_vec=gamma_vec)
            X_prev = X_pdhg.clone().detach()
            Z_prev = Z_pdhg.clone().detach()

            flag_save_solution = info_pdhg['solved']
            if flag_save_solution:
                sol_X = X_pdhg
                sol_Z = Z_pdhg
                solutions[pdhg_iter - 1] = sol_X

                # break
            result_list.append([gamma_vec, info_pdhg['primobj'].cpu().item(),  info_pdhg['dualobj'].cpu().item(), info_pdhg['dualgap'].cpu().item(), info_pdhg['iter'], info_pdhg['time']])
            pdhg_time += info_pdhg['time']

        gv.logger.info('Total time taken by PDHG = %5.4f s', pdhg_time)
        solutions = solutions*data.max_dist
        return solutions


    def solve_iter(self, data: DataProcessor, X: torch.Tensor, Z: torch.Tensor, gamma_vec, **params):
        '''
        solve the clustering problem using PDHG algorithm
        :param data:
        :param X:
        :param Z:
        :param params:
        :return:
        '''
        # complete PDHG algorithm
        self.gamma_vec = gamma_vec
        _, S, _ = torch.svd_lowrank(data.nodeArcMatrix, q=1, niter=400)
        kappa = S[0]
        if self.print_yes:
            gv.logger.info('--------------PDHG for solving Clustering with kappa = %6.4f and gamma = %6.4f--------------',
                           kappa, self.gamma_vec)
            gv.logger.info('\t Iter \t|\t Primobj \t|\t Dualobj \t|\t Relative gap \t|\t Time(s) \t|')
        time_s = time.perf_counter()

        tau_0 = 1.618
        sigma = tau_0 / kappa
        tau = 1.0 / (0.751 * tau_0 * kappa)
        weightVec = self.gamma_vec * data.weightVector
        iter = 0

        # compute the primal objective
        norms = torch.sqrt(torch.sum(data.a_map(X) ** 2, dim=0))
        obj_prim_1 = 0.5 * torch.norm(X - data.X, p='fro') ** 2
        obj_prim_2 = torch.matmul(weightVec, norms)
        obj_prim = obj_prim_1 + obj_prim_2

        ATZ = data.at_map(Z)
        ATZ_norm = torch.norm(ATZ, p='fro')
        obj_dual = -0.5 * ATZ_norm ** 2 + torch.sum(ATZ * data.X)
        gap_dual = abs(obj_prim - obj_dual) / (1.0 + abs(obj_prim) + abs(obj_dual))

        for iter in range(self.max_iter):
        # for iter in tqdm(range(self.max_iter), desc="PDHG", ncols=100, unit="epoch"):
            iter = iter + 1

            # update primal variable X
            X_old = X.clone().detach()
            X = sigma / (1 + sigma) * (data.X - ATZ) + (1.0 / (1.0 + sigma)) * X_old

            # update dual variable Z
            proj_input = Z + tau * data.a_map(2 * X - X_old)
            Z = proj_l2(proj_input, weightVec)

            # compute duality gap
            ATZ = data.at_map(Z)
            ATZ_norm = torch.norm(ATZ, p='fro')
            obj_dual = -0.5 * ATZ_norm ** 2 + torch.sum(ATZ * data.X)


            obj_prim_1 = 0.5 * torch.norm(X - data.X, p='fro') ** 2
            obj_prim_2 = torch.matmul(weightVec, torch.sqrt(torch.sum(data.a_map(X) ** 2, dim=0)))
            obj_prim = obj_prim_1 + obj_prim_2

            gap_dual = abs(obj_prim - obj_dual) / (1.0 + abs(obj_prim) + abs(obj_dual))

            if self.device == 'cuda':
                torch.cuda.synchronize()
            time_iter = time.perf_counter() - time_s
            if iter % 200 == 0 and self.print_yes:
                gv.logger.info('\t %5.0d \t|\t %- 5.4e \t|\t %- 5.4e \t|\t %- 5.4e \t|\t %- 5.4e \t|', iter, obj_prim,
                               obj_dual,
                               gap_dual, time_iter)
            if gap_dual < self.stop_tol:
                break
        if self.device == 'cuda':
            torch.cuda.synchronize()
        time_e = time.perf_counter()
        time_used = (time_e - time_s)

        gv.logger.info('\t %5.4f \t %5.0d \t|\t %- 7.6e \t|\t %- 7.6e \t|\t %- 7.6e \t|\t %- 5.4f s\t|',
                       self.gamma_vec, iter, obj_prim,
                       obj_dual, gap_dual, time_iter)

        if iter > self.max_iter:
            solved = False
        else:
            solved = True

        result = {'iter': iter, 'dualobj': obj_dual, 'primobj': obj_prim, 'dualgap': gap_dual, 'time': time_used,
                  'solved': solved}
        return X, Z, result