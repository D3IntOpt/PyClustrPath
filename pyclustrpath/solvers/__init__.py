from .ama import AMA, FAST_AMA
from .admm import ADMM
from .ssnal import SSNAL
from .pdhg import PDHG

def ConvexClusterSolver(method, gamma_list, device, **params):
    method = method.lower()
    if method == 'ama':
        return AMA(gamma_vec_list=gamma_list, device=device, **params)
    elif method == 'fast_ama':
        return FAST_AMA(gamma_vec_list=gamma_list, device=device, **params)

    elif method == 'admm':
        return ADMM(gamma_vec_list=gamma_list, device=device, **params)

    elif method == 'ssnal':
        return SSNAL(gamma_vec_list=gamma_list, device=device, **params)

    elif method == 'pdhg':
        return PDHG(gamma_vec_list=gamma_list, device=device, **params)

    else:
        raise ValueError(f"Unknown method: {method}")


__all__ = ['AMA', 'FAST_AMA', 'ADMM', 'SSNAL', 'PDHG']