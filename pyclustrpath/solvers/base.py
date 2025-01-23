import torch
class SolverBase:
    def __init__(self, gamma_vec_list, device, gamma_vec=0, stop_tol=1e-5, max_iter=100000, use_kkt=False, **params):
        """
        Initialize the solver
        :param gamma_vec_list: list of gamma values
        :param device: device to run the solver
        :param gamma_vec: gamma value
        :param stop_tol: stopping tolerance
        :param max_iter: maximum number of iterations
        :param use_kkt: whether to use the KKT condition to stop the solver
        """
        self.gamma_vec_list = gamma_vec_list
        self.gamma_vec = gamma_vec
        self.stop_tol = stop_tol
        self.max_iter = max_iter
        self.use_kkt = use_kkt
        self.params = params
        self.device = device
        self.to_device(device)

        self.csv_write_flag = False
        self.bench_test = False
        self.print_yes = 0


    def solve(self, data):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def to_device(self, device):
        """
        Move the tensors to the specified device
        """
        for attr_name, attr_value in self.__dict__.items():
            if isinstance(attr_value, torch.Tensor):
                setattr(self, attr_name, attr_value.to(device))