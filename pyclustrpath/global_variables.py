import logging
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_HOME"] = "/usr/local/cuda-12.1"

logger = logging.getLogger('mylogger')

def _init(args):
    global logger
    log_init(logger, args)

    global solver_method, data_name, use_coo
    solver_method = args.solver if args.solver is not None else 'ssnal'
    data_name = args.data
    # use_coo = 0 if args.device == 'cpu' else 0
    use_coo = 0

def log_init(logger, agrs):
    '''
    :param logger: global logger
    :param agrs: input arguments
    :return:
    '''

    logger_path = f'log_file/output_{agrs.solver}_{agrs.data}_{agrs.stop_tol}.log'

    logger = logging.getLogger('mylogger')
    logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)  # suppress matplotlib font manager warning

    if not os.path.exists('log_file'):
        os.makedirs('log_file')
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.DEBUG,
        handlers=[
            logging.FileHandler(os.path.join(logger_path)),
            logging.StreamHandler()
        ])
    logger.info(agrs)

    return logger


def runhist(maxiter=10000):
    '''
    :param maxiter: maximum iteration
    :return: runhist
    '''
    runhist = {'feas_prim': [], 'feas_dual': [], 'feas_prim_prev': [], 'feas_dual_prev': [], 'feas_max': [],
               'feas_max_prev': [], 'sigma': [], 'psqmrxiiter': [], 'obj_prim': [], 'obj_dual': [], 'rel_gap': [],
               'time': [], 'rankS': []}
    # create a dictionary to store the history of the optimization process

    for key in runhist.keys():
        runhist[key] = [0] * (maxiter + 1)

    return runhist

