import torch
import time
from .base import SolverBase
import pyclustrpath.global_variables as gv
from pyclustrpath.data_processor import DataProcessor
from pyclustrpath.utils import *

class AMA(SolverBase):
    def solve(self, data: DataProcessor, **params):
        result_list = []
        n, d, w = data.n, data.d, data.num_weights
        Z_prev = torch.zeros((d, w), dtype=torch.float64, device=self.device)
        gv.logger.info('\t Gamma \t|\t Iter \t|\t   Primobj   \t|\t   Dualobj   \t|\t Relative gap \t|\t Time(s) \t|')

        ama_time = 0
        ama_iter = 0
        solutions = torch.zeros((len(self.gamma_vec_list), d, n), dtype=torch.float64)
        for gamma_vec in self.gamma_vec_list:
            ama_iter += 1
            X_ama, Z_ama, info_ama = self.solve_iter(data, Z_prev, gamma_vec=gamma_vec)
            Z_prev = Z_ama.clone().detach()

            flag_save_solution = info_ama['solved']
            if flag_save_solution:
                sol_X = X_ama
                sol_Z = Z_ama
                solutions[ama_iter - 1] = sol_X
                # break
            result_list.append([gamma_vec, info_ama['primobj'].cpu().item(),  info_ama['dualobj'].cpu().item(), info_ama['dualgap'].cpu().item(), info_ama['iter'], info_ama['time']])
            ama_time += info_ama['time']

        gv.logger.info('Total time taken by AMA = %5.4f s', ama_time)
        solutions = solutions*data.max_dist
        return solutions


    def solve_iter(self, data: DataProcessor, Z: torch.Tensor, gamma_vec, **params):
        '''
        solve the clustering problem using AMA algorithm
        :param data:
        :param Z: dim: (d, num_weights)
        :param params:
        :return: X_new, Z_new, result
        '''
        # complete AMA algorithm
        time_s = time.perf_counter()
        mu = 2 / (data.n + 1)
        self.gamma_vec = gamma_vec

        # print_yes = 0
        if self.print_yes:
            gv.logger.info('--------------AMA for solving Clustering with mu = %6.4f and gamma = %6.4f--------------',
                           mu, self.gamma_vec)
            gv.logger.info('\t Iter \t|\t Primobj \t|\t Dualobj \t|\t Relative gap \t|\t Time(s) \t|')

        weightVec = self.gamma_vec * data.weightVector
        amapb_vec = data.a_map(data.X)

        Z_new = Z.clone().detach()

        X_new = data.X + data.at_map(Z_new)
        iter = 0
        obj_dual = 0

        # compute the primal objective
        norms = torch.sqrt(torch.sum(data.a_map(X_new) ** 2, dim=0))
        primalobj_1 = 0.5 * torch.norm(X_new - data.X, p='fro') ** 2
        primalobj_2 = torch.matmul(weightVec, norms)
        obj_prim = primalobj_1 + primalobj_2

        gap_dual = abs(obj_prim - obj_dual) / (1.0 + abs(obj_prim) + abs(obj_dual))

        if self.print_yes:
            gv.logger.info('\t %5.0d \t|\t %- 5.4e \t|\t %- 5.4e \t|\t %- 5.4e \t|\t %- 5.4e \t|', iter, obj_prim, obj_dual,
                        gap_dual, time.perf_counter() - time_s)

        for iter in range(self.max_iter):
            iter = iter + 1
            Delta = data.at_map(Z_new)
            gvec = amapb_vec + data.a_map(Delta)
            proj_input = - (mu * gvec - Z_new)

            # projection onto l2 ball
            Z_new = proj_l2(proj_input, weightVec)

            # compute duality gap
            ATZ = data.at_map(Z_new)
            ATZ_norm = torch.norm(ATZ, p='fro')
            obj_dual = -0.5 * ATZ_norm ** 2 - torch.sum(ATZ * data.X)

            # Recovery primal variable
            X_new = data.X + ATZ
            primalobj_1 = 0.5 * ATZ_norm ** 2
            primalobj_2 = torch.matmul(weightVec, torch.sqrt(torch.sum(data.a_map(X_new) ** 2, dim=0)))
            obj_prim = primalobj_1 + primalobj_2
            gap_dual = abs(obj_prim - obj_dual) / (1.0 + abs(obj_prim) + abs(obj_dual))
            if self.device == 'cuda':
                torch.cuda.synchronize()
            time_iter = time.perf_counter() - time_s

            if self.print_yes and (iter % 100 == 0):
                gv.logger.info('\t %5.0d \t|\t %- 5.4e \t|\t %- 5.4e \t|\t %- 5.4e \t|\t %- 5.4e \t|', iter, obj_prim,
                               obj_dual, gap_dual, time_iter)
            if gap_dual < self.stop_tol:
                break
        if self.device == 'cuda':
            torch.cuda.synchronize()
        time_e = time.perf_counter()
        time_used = (time_e - time_s)

        gv.logger.info('\t %5.4f \t %5.0d \t|\t %- 7.6e \t|\t %- 7.6e \t|\t %- 7.6e \t|\t %- 5.4f s\t|', self.gamma_vec, iter, obj_prim,
                           obj_dual, gap_dual, time_iter)

        if iter < self.max_iter:
            solved = True
        else:
            solved = False

        result = {'iter': iter, 'primobj': obj_prim, 'dualobj': obj_dual, 'dualgap': gap_dual, 'time': time_used,
                  'solved': solved}
        return X_new, Z_new, result

class FAST_AMA(SolverBase):

    def solve(self, data: DataProcessor, **params):
        n, d, w = data.n, data.d, data.num_weights
        Z_prev = torch.zeros((d, w), dtype=torch.float64, device=self.device)
        gv.logger.info('\t Gamma \t|\t Iter \t|\t   Primobj   \t|\t   Dualobj   \t|\t Relative gap \t|\t Time(s) \t|')
        fast_ama_time = 0
        fast_ama_iter = 0
        solutions = torch.zeros((len(self.gamma_vec_list), d, n), dtype=torch.float64, device=self.device)
        for gamma_vec in self.gamma_vec_list:
            fast_ama_iter += 1
            X_fast_ama, Z_fast_ama, info_fast_ama = self.solve_iter(data, Z_prev, gamma_vec=gamma_vec)
            Z_prev = Z_fast_ama.clone().detach()
            fast_ama_time += info_fast_ama['time']
            flag_save_solution = info_fast_ama['solved']
            if flag_save_solution:
                sol_X = X_fast_ama
                sol_Z = Z_fast_ama
                solutions[fast_ama_iter - 1] = sol_X
                # break

        gv.logger.info('Total time taken by FAST_AMA = %5.4f s', fast_ama_time)
        solutions = solutions * data.max_dist
        return solutions

    def solve_iter(self, data: DataProcessor, Z: torch.Tensor, gamma_vec, **params):
        '''
        solve the clustering problem using FAST_AMA algorithm
        :param data: DataProcessor object
        :param Z: dim: (d, num_weights)
        :param params:
        :return: X_new, Z_new, result
        '''
        # complete FAST_AMA algorithm
        time_s = time.perf_counter()
        self.gamma_vec = gamma_vec
        largest_eigenvalues, _ = torch.lobpcg(data.ata_mat, k=1, B=None, largest=True)
        rho = largest_eigenvalues[0]
        mu = 1 / (rho + 1)
        if self.print_yes:
            gv.logger.info('--------------FAST_AMA for solving Clustering with mu = %6.4f and gamma = %6.4f--------------',
                           mu, self.gamma_vec)
            gv.logger.info('\t Iter \t|\t   Primobj   \t|\t   Dualobj   \t|\t Relative gap \t|\t Time(s) \t|')

        weightVec = self.gamma_vec * data.weightVector
        amapb_vec = data.a_map(data.X)
        Z_tilde = Z.clone().detach()
        Z_old = Z.clone().detach()

        X_new = data.X + data.at_map(Z_old)
        iter = 0
        obj_dual = 0

        # compute the primal objective
        norms = torch.sqrt(torch.sum(data.a_map(X_new) ** 2, dim=0))
        primalobj_1 = 0.5 * torch.norm(X_new - data.X, p='fro') ** 2
        primalobj_2 = torch.matmul(weightVec, norms)
        obj_prim = primalobj_1 + primalobj_2

        gap_dual = abs(obj_prim - obj_dual) / (1.0 + abs(obj_prim) + abs(obj_dual))
        alpha_old = 1

        for iter in range(self.max_iter):
            iter = iter + 1
            Delta = data.at_map(Z_tilde)
            gvec = amapb_vec + data.a_map(Delta)
            proj_input = - (mu * gvec - Z_tilde)

            # projection onto l2 ball
            Z_new = proj_l2(proj_input, weightVec)
            alpha = (1 + np.sqrt(1 + 4 * alpha_old ** 2)) / 2.0
            Z_tilde = Z_new + (alpha_old / alpha) * (Z_new - Z_old)
            Z_old = Z_new
            alpha_old = alpha

            # compute duality gap
            ATZ = data.at_map(Z_new)
            ATZ_norm = torch.norm(ATZ, p='fro')
            obj_dual = -0.5 * ATZ_norm ** 2 - torch.sum(ATZ * data.X)

            # Recovery primal variable
            X_new = data.X + ATZ
            primalobj_1 = 0.5 * ATZ_norm ** 2
            primalobj_2 = torch.matmul(weightVec, torch.sqrt(torch.sum(data.a_map(X_new) ** 2, dim=0)))
            obj_prim = primalobj_1 + primalobj_2
            gap_dual = abs(obj_prim - obj_dual) / (1.0 + abs(obj_prim) + abs(obj_dual))

            if self.device == 'cuda':
                torch.cuda.synchronize()
            time_iter = time.perf_counter() - time_s

            if self.print_yes and (iter % 100 == 0):
                gv.logger.info('\t %5.0d \t|\t %- 5.4e \t|\t %- 5.4e \t|\t %- 5.4e \t|\t %- 5.4e \t|', iter, obj_prim,
                               obj_dual, gap_dual, time_iter)
            if gap_dual < self.stop_tol:
                break
        if self.device == 'cuda':
            torch.cuda.synchronize()
        time_e = time.perf_counter()
        time_used = (time_e - time_s)

        if iter == 0:
            Z_new = Z_old
        if iter < self.max_iter:
            solved = True
        else:
            solved = False

        gv.logger.info('\t %5.4f \t %5.0d \t|\t %- 7.6e \t|\t %- 7.6e \t|\t %- 7.6e \t|\t %- 5.4f s\t|',
                       self.gamma_vec, iter, obj_prim, obj_dual, gap_dual, time_iter)
        result = {'iter': iter,'primobj': obj_prim, 'dualobj': obj_dual, 'dualgap': gap_dual, 'time': time_used,
                  'solved': solved}
        return X_new, Z_new, result