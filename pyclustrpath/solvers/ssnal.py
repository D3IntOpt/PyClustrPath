import torch
import time
import os

import pyclustrpath.global_variables as gv
from .admm import ADMM
from .base import SolverBase
from pyclustrpath import DataProcessor
from pyclustrpath.utils import *

import sys
import importlib.util
#load ssncg module
current_path = os.path.dirname(os.path.abspath(__file__))
file_path = current_path + '/ssncg.cpython-310.pyc'
module_name = 'ssncg'
spec = importlib.util.spec_from_file_location(module_name, file_path)
ssncg = importlib.util.module_from_spec(spec)
sys.modules[module_name] = ssncg
spec.loader.exec_module(ssncg)

class SSNAL(SolverBase):

    def __init__(self, gamma_vec_list, device, gamma=1.618, stop_tol=1e-5, max_iter=100, use_kkt=False, admm_sub_method='cholesky', **params):
        super().__init__(gamma_vec_list=gamma_vec_list, stop_tol=stop_tol, max_iter=max_iter, use_kkt=use_kkt, device=device, **params)
        self.sigma = 0
        self.admm_sub_method = admm_sub_method
        self.params = params
        self.to_device(device)

    def solve(self, data: DataProcessor, **params):
        n, d, w = data.n, data.d, data.num_weights
        bench_num = 3

        gv.logger.info(
            'Gamma |   Iter   |   [ pinfeas      dinfeas    relgap ]  |   primobj   |   '
            'dualobj   |   time   |   sigma   |  [ SSNCG_iter_per_iter  row_prox ]')

        X_prev = data.X.clone().detach()
        Y_prev = data.a_map(X_prev)
        Z_prev = torch.zeros((d, w), dtype=torch.float64, device=self.device)
        ssnal_iter = 0
        ssnal_time = 0

        solutions = torch.zeros((len(self.gamma_vec_list), d, n), dtype=torch.float64, device=self.device)
        for gamma_vec in self.gamma_vec_list:
            ssnal_iter += 1
            admm_iter = 200
            X_SSNAL, Y_SSNAL, Z_SSNAL, info_SSNAL, runhist_SSNAL = self.solve_iter(data, X_prev, Y_prev, Z_prev,
                                                                                   admm_iter=admm_iter, gamma_vec=gamma_vec,
                                                                                   **params)
            X_prev = X_SSNAL
            Y_prev = Y_SSNAL
            Z_prev = Z_SSNAL

            flag_save_solution = info_SSNAL['solved']
            ssnal_time += info_SSNAL['time']

            if flag_save_solution:
                # sol_X = X_SSNAL
                # sol_Z = Z_SSNAL
                # sol_Y = Y_SSNAL
                solutions[ssnal_iter - 1] = X_SSNAL
                # break

        gv.logger.info('Total time taken by SSNAL = %5.4f s', ssnal_time)

        solutions = solutions * data.max_dist
        return solutions

    def solve_iter(self, data: DataProcessor, X: torch.Tensor, Y: torch.Tensor, Z: torch.Tensor, admm_iter: int, gamma_vec,
                   **params):
        """
        solve the clustering problem using SSNAL algorithm
        :param data:  DataProcessor object
        :param X: dim: (d, n)
        :param Y: dim: (d, w)
        :param Z: dim: (d, w)
        :param admm_iter: the maximum number of iterations for pre-training ADMM
        :param gamma_vec: gamma value
        :param params:
        :return: X_new, Y_new, Z_new, result, runhist
        """
        self.gamma_vec = gamma_vec
        msg = ' '
        runhist = gv.runhist(self.max_iter)
        time_s = time.perf_counter()
        weightVec = self.gamma_vec * data.weightVector

        # ————————————————phase I————————————————
        if admm_iter > 0:

            solver = ADMM(gamma_vec_list=[self.gamma_vec], admm_sub_method=self.admm_sub_method, stop_tol=self.stop_tol, max_iter=admm_iter, use_kkt=self.use_kkt,device=self.device)
            X, Y, Z, info_admm, runhist_admm = solver.solve_iter(data, X, Y, Z, gamma_vec=gamma_vec)

            self.sigma = min(info_admm['sigma'], 10)

            if info_admm['eta'] < self.stop_tol:

                gv.logger.info(
                    '%5.4f  admm%3.0d |  [ %-5.4e  %-5.4e  %-5.4e] | %- 7.6e | %- 7.6e |  %5.4f s | %- 3.2e |  [ %6.2f   %6.0f   ]',
                    self.gamma_vec, info_admm['iter'], info_admm['feas_prim'], info_admm['feas_dual'],
                    info_admm['rel_gap'],
                    info_admm['prim_obj'], info_admm['dual_obj'], info_admm['time'], info_admm['sigma'],
                    info_admm['cg_iter_total'], info_admm['sum_rr'])

                result = info_admm
                runhist = runhist_admm

                return X, Y, Z, result, runhist

        # ————————————————Initialization————————————————
        # print_yes = False
        printminoryes = False

        breakyes = False

        norm_y = torch.norm(Y, p='fro')
        a_x = data.a_map(X)
        at_z = data.at_map(Z)
        r_prim = a_x - Y
        norm_r_prim = torch.norm(r_prim, p='fro')

        proj_z = proj_l2(Z, weightVec)
        r_dual = Z - proj_z

        feas_prim = norm_r_prim / (1 + norm_y)
        feas_dual = torch.norm(r_dual, p='fro') / (1 + torch.norm(Z, p='fro'))
        feas_max = max(feas_prim, feas_dual)

        feas_prim_prev = feas_prim
        feas_dual_prev = feas_dual

        obj_prim = 0.5 * torch.norm(X - data.X, p='fro') ** 2 + torch.matmul(weightVec, torch.sqrt(
            torch.sum(a_x ** 2, dim=0)))
        obj_dual = -0.5 * torch.norm(at_z, p='fro') ** 2 + torch.sum(data.X * at_z)
        rel_gap = abs(obj_prim - obj_dual) / (1 + abs(obj_prim) + abs(obj_dual))

        if self.print_yes:
            gv.logger.info('****************************************************************************************')
            gv.logger.info(' --------------Phase II: NAL_Clustering with  gamma = %6.4f--------------', self.gamma_vec)
            gv.logger.info('****************************************************************************************')
            if printminoryes:
                gv.logger.info('d = %d, n = %d, w = %d', data.d, data.n, data.num_weights)
            gv.logger.info(
                '   Iter   |   [ pinfeas      dinfeas ] | [  pinforg      dinforg      relgap ]  |   primobj   |   '
                'dualobj   |   time   |   sigma   |   rankS   |  [ cg_iter  row_prox ]')

            gv.logger.info(
                '  %5.0d   |  [ %-5.4e  %-5.4e] | [%- 5.4e  %- 5.4e  %-5.4e] | %- 5.4e | %- 5.4e |  %5.3f s | %- 3.2e |',
                0, feas_prim, feas_dual, feas_prim_prev, feas_dual_prev, rel_gap, obj_prim, obj_dual,
                info_admm['time'], self.sigma)

        if feas_max < max(1e-6, self.stop_tol):
            if self.use_kkt:
                grad = at_z + X - data.X
                eta = torch.norm(grad, p='fro') / (1 + torch.norm(X, p='fro'))
                ypz = Y + Z
                ypz_prox, _, _ = prox_l2(ypz, weightVec)
                res_vec = Y - ypz_prox
                eta = eta + torch.norm(res_vec, p='fro') / (1 + torch.norm(Y, p='fro'))
                eta_org = eta
            else:
                obj_prim = 0.5 * torch.norm(X - data.X, p='fro') ** 2 + torch.matmul(weightVec, torch.sqrt(
                    torch.sum(a_x ** 2, dim=0)))
                obj_dual = -0.5 * torch.norm(at_z, p='fro') ** 2 + torch.sum(data.X * at_z)
                rel_gap = abs(obj_prim - obj_dual) / (1 + abs(obj_prim) + abs(obj_dual))
                eta = rel_gap
                eta_org = eta
            if eta < self.stop_tol:
                breakyes = 1
                msg = 'converged'

        if breakyes:
            time_2 = time.perf_counter()
            time_total = time_2 - time_s
            if printminoryes:
                gv.logger.info('msg = %s', msg)
                gv.logger.info('--------------------------------------------------------------')
                gv.logger.info('  admm iter = %3.0d, admm time = %3.1f s', info_admm['iter'], info_admm['time'])
                gv.logger.info('  number iter = %2.0d', 0)
                gv.logger.info('  time = %3.2f', time_total)
                gv.logger.info('  time per iter = %5.4f', time_total / iter)
                gv.logger.info('  primobj = %9.8e, dualobj = %9.8e, relgap = %3.2e', obj_prim, obj_dual, rel_gap)
                gv.logger.info('  primfeasorg = %3.2e, dualfeasorg = %3.2e', feas_prim, feas_dual)
                gv.logger.info('  eta = %3.2e, etaorg = %3.2e', eta, eta_org)
                gv.logger.info('min(Z) = %3.2e, max(Z) = %3.2e', torch.min(Z), torch.max(Z))
                gv.logger.info('--------------------------------------------------------------')

        # ————————————————SSNCG————————————————
        use_SSNCG = 1
        if use_SSNCG:
            parNCG = {}
            parNCG['matvecfname'] = 'matvecIpxi'
            parNCG['sigma'] = self.sigma
            parNCG['tolconst'] = 0.5
            parNCG['dim'] = data.d
            parNCG['maxiter'] = 100

            maxitersub = 10
            prim_win = 0
            dual_win = 0

            ssncgopt = {}
            ssncgopt['tol'] = self.stop_tol
            ssncgopt['precond'] = 0
            ssncgopt['bscale'] = 1
            ssncgopt['cscale'] = 1
            # for iter = 1:maxiter
            ssncg_iter_total = 0

        for iter in range(1, self.max_iter + 1):
            # for iter in tqdm(range(1, self.max_iter + 1), desc="SSNAL", ncols=100, unit="epoch"):
            Z_old = Z
            parNCG['sigma'] = self.sigma
            eta, eta_org = 0, 0

            if feas_prim < 1e-5:
                maxitersub = max(maxitersub, 30)
            elif feas_prim < 1e-3:
                maxitersub = max(maxitersub, 30)
            elif feas_prim < 1e-1:
                maxitersub = max(maxitersub, 20)

            ssncgopt['maxitersub'] = maxitersub
            time_ssncg_s = time.perf_counter()
            Y, a_x, X, parNCG, info_NCG = ssncg.SSNCG(data, Z, a_x, X, weightVec, parNCG, ssncgopt, self.device)
            ssncg_iter_total += info_NCG['iter']
            if self.device == 'cuda':
                torch.cuda.synchronize()
            time_ssncg_e = time.perf_counter() - time_ssncg_s
            # print('Time SSNCG = %5.3f s' % time_ssncg_e)

            if info_NCG['breakyes'] < 0:
                parNCG['tolconst'] = max(parNCG['tolconst'] / 1.06, 1e-3)

            r_prim = a_x - Y
            Z = Z_old + self.sigma * r_prim
            at_z = data.at_map(Z)
            norm_r_prim = torch.norm(r_prim, p='fro')
            norm_y = torch.norm(Y, p='fro')

            feas_prim = norm_r_prim / (1 + norm_y)
            feas_prim_prev = feas_prim
            proj_z = proj_l2(Z, weightVec)
            r_dual = Z - proj_z
            feas_dual = torch.norm(r_dual, p='fro') / (1 + torch.norm(Z, p='fro'))
            feas_dual_prev = feas_dual
            feas_max = max(feas_prim, feas_dual)
            feas_max_prev = feas_max

            runhist['feas_prim'][iter] = feas_prim
            runhist['feas_dual'][iter] = feas_dual
            runhist['feas_max'][iter] = feas_max
            runhist['feas_prim_prev'][iter] = feas_prim_prev
            runhist['feas_dual_prev'][iter] = feas_dual_prev
            runhist['sigma'][iter] = self.sigma
            runhist['rankS'][iter] = parNCG['rankS']

            ## ————————————————check for termination————————————————
            if feas_max < max(1e-6, self.stop_tol):
                if self.use_kkt:
                    grad = at_z + X - data.X
                    eta = torch.norm(grad, p='fro') / (1 + torch.norm(X, p='fro'))
                    ypz = Y + Z
                    ypz_prox, _, _ = prox_l2(ypz, weightVec)
                    res_vec = Y - ypz_prox
                    eta = eta + torch.norm(res_vec, p='fro') / (1 + torch.norm(Y, p='fro'))
                    eta_org = eta
                else:
                    obj_prim = 0.5 * torch.norm(X - data.X, p='fro') ** 2 + torch.matmul(weightVec, torch.sqrt(
                        torch.sum(a_x ** 2, dim=0)))
                    obj_dual = -0.5 * torch.norm(at_z, p='fro') ** 2 + torch.sum(data.X * at_z)
                    rel_gap = abs(obj_prim - obj_dual) / (1 + abs(obj_prim) + abs(obj_dual))
                    eta = rel_gap
                    eta_org = eta

                if eta < self.stop_tol:
                    breakyes = 1
                    msg = 'converged'

            # ————————————————print results————————————————
            if feas_max > self.stop_tol:
                obj_prim = 0.5 * torch.norm(X - data.X, p='fro') ** 2 + torch.matmul(weightVec, torch.sqrt(
                    torch.sum(a_x ** 2, dim=0)))
                obj_dual = -0.5 * torch.norm(at_z, p='fro') ** 2 + torch.sum(data.X * at_z)
                rel_gap = abs(obj_prim - obj_dual) / (1 + abs(obj_prim) + abs(obj_dual))
            time_e = time.perf_counter()
            time_total = time_e - time_s

            if printminoryes and eta == 0:
                gv.logger.info(
                    '  %5.0d   |  [ %-5.4e  %-5.4e] | [%- 5.4e  %- 5.4e  %-5.4e] | %- 5.4e | %- 5.4e |  %5.3f s | %- 3.2e | %5.0d |',
                    iter, feas_prim, feas_dual, feas_prim_prev, feas_dual_prev, rel_gap, obj_prim, obj_dual, time_total,
                    self.sigma, parNCG['rankS'])
            elif printminoryes and eta != 0:
                gv.logger.info(
                    '  %5.0d   |  [ %-5.4e  %-5.4e] | [%- 5.4e  %- 5.4e  %-5.4e] | %- 5.4e | %- 5.4e |  %5.3f s | %- 3.2e | %5.0d | '
                    '[eta = %3.2e,  etaorg = %3.2e]',
                    iter, feas_prim, feas_dual, feas_prim_prev, feas_dual_prev, rel_gap, obj_prim, obj_dual, time_total,
                    self.sigma, parNCG['rankS'], eta, eta_org)

                runhist['obj_prim'][iter] = obj_prim
                runhist['obj_dual'][iter] = obj_dual
                runhist['rel_gap'][iter] = rel_gap
                runhist['time'][iter] = time_total
                runhist['psqmrxiiter'][iter] = info_NCG['iter']

            if breakyes > 0:
                if self.print_yes:
                    gv.logger.info('breakyes = %3.1f, %s', breakyes, msg)
                break

            if feas_prim_prev < feas_dual_prev:
                prim_win += 1
            else:
                dual_win += 1

            if iter < 10:
                sigma_update_iter = 2
            elif iter < 20:
                sigma_update_iter = 3
            elif iter < 200:
                sigma_update_iter = 3
            elif iter < 500:
                sigma_update_iter = 10

            sigma_scale = 5
            sigma_max = 1e5
            if iter % sigma_update_iter == 0:
                sigmamin = 1e-4
                if prim_win > max(1, 1.2 * dual_win):
                    prim_win = 0
                    self.sigma = max(sigmamin, self.sigma / sigma_scale)
                elif dual_win > max(1, 1.2 * prim_win):
                    dual_win = 0
                    self.sigma = min(sigma_max, self.sigma * sigma_scale)

        # ———————————————— recover orignal variables————————————————
        if iter == self.max_iter:
            msg = ' maximum iteration reached'
            obj_prim = 0.5 * torch.norm(X - data.X, p='fro') ** 2 + \
                       torch.matmul(weightVec, torch.sqrt(torch.sum(a_x ** 2, dim=0)))
            obj_dual = -0.5 * torch.norm(at_z, p='fro') ** 2 + torch.sum(data.X * at_z)
            rel_gap = abs(obj_prim - obj_dual) / (1 + abs(obj_prim) + abs(obj_dual))
            if self.use_kkt:
                grad = at_z + X - data.X
                eta = torch.norm(grad, p='fro') / (1 + torch.norm(X, p='fro'))
                ypz = Y + Z
                ypz_prox, _, _ = prox_l2(ypz, weightVec)
                res_vec = Y - ypz_prox
                eta = eta + torch.norm(res_vec, p='fro') / (1 + torch.norm(Y, p='fro'))
                eta_org = eta
            else:
                eta = rel_gap
                eta_org = eta

        if iter < self.max_iter:
            solved = True
        else:
            solved = False

        time_2 = time.perf_counter()
        time_total = time_2 - time_s
        if printminoryes:
            gv.logger.info('msg = %s', msg)
            gv.logger.info('--------------------------------------------------------------')
            gv.logger.info('  admm iter = %3.0d, admm time = %3.1f', info_admm['iter'], info_admm['time'])
            gv.logger.info('  number iter = %2.0d', iter)
            gv.logger.info('  time = %3.2f', time_total)
            gv.logger.info('  time per iter = %5.4f', time_total / iter)
            gv.logger.info('  primobj = %9.8e, dualobj = %9.8e, relgap = %3.2e', obj_prim, obj_dual, rel_gap)
            gv.logger.info('  primfeasorg = %3.2e, dualfeasorg = %3.2e', feas_prim, feas_dual)
            gv.logger.info('  eta = %3.2e, etaorg = %3.2e', eta, eta_org)
            gv.logger.info('min(Z) = %3.2e, max(Z) = %3.2e', torch.min(Z), torch.max(Z))
            gv.logger.info('--------------------------------------------------------------')

        gv.logger.info(
            '%5.4f  %5.0d   |  [ %-5.4e  %-5.4e  %-5.4e] | %- 7.6e | %- 7.6e |  %5.4f s | %- 3.2e |  [ %6.2f   %6.0f   ]',
            self.gamma_vec, iter, feas_prim, feas_dual, rel_gap, obj_prim, obj_dual,
            time_total, self.sigma, ssncg_iter_total / iter, parNCG['rankS'])

        abs_gap = abs(obj_prim - obj_dual)
        result = {'iter': iter, 'dual_obj': obj_dual, 'prim_obj': obj_prim, 'rel_gap': rel_gap, 'time': time_total,
                  'eta': eta, 'eta_org': eta_org, 'feas_prim': feas_prim, 'feas_dual': feas_dual,
                  'solved': solved, 'sigma': self.sigma, 'admm_time': info_admm['time'], 'abs_gap': abs_gap}

        return X, Y, Z, result, runhist

