import time
import torch
import math
import numpy as np

from sklearn.neighbors import NearestNeighbors
import pyclustrpath.global_variables as gv
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    from knn_cuda import KNN  # Ref: https://github.com/unlimblue/KNN_CUDA

class DataProcessor:
    def __init__(self, X, k=10, device=torch.device('cpu'), **params):
        self.device = device
        # self.X = X.t() # dim: d*n for matrix multiplication in Math
        self.X = X
        self.X = self.X.to(device)
        self.label = params.get('label', None)
        self.relabel = params.get('relabel', None)

        self.k = k
        self.d, self.n = self.X.shape
        self.max_gamma = 0
        self.num_weights = params.get('num_weights', 0)
        self.unique_pairs = None # dim: num_weights * 2
        self.weightVector = None    # dim: 1 * weight_num
        self.nodeArcMatrix = None   # dim: n * weight_num
        self.ata_mat = None        # dim: n * n
        self.ata_mat_dense = None
        self.L = None # Laplacian matrix
        self.dis_matrix = None
        self.max_dist = 0

        self.weight_norm_flag = 0
        self.preprocess()

        self.compute_weight()
        self.to_device(device)


    def preprocess(self):
        """
        Preprocess the data by normalizing the data using the maximum distance
        :return:
        """
        time_start = time.perf_counter()
        X_T = self.X.T
        self.dis_matrix = torch.cdist(X_T, X_T)
        self.max_dist = self.dis_matrix.max()
        self.X = self.X / self.max_dist

        if self.device == 'cuda':
            torch.cuda.synchronize()
        time_end = time.perf_counter()
        print('Time taken to preprocess data = %5.4f s' % (time_end - time_start))
        return

    def compute_weight(self, phi=0.5, gamma=1, **params):
        """
        Compute the weight matrix for the data processor
        :param phi:  the parameter for the exponential function
        :param gamma: the parameter for the weight vector
        :param params:  other parameters
        """
        time_start = time.perf_counter()
        gv.logger.info('Start to calculate weights!')

        # Step 1: Calculate the distances and indices of KNN
        if self.device == 'cpu':  # KNN in sklearn
            X_np = self.X.cpu().numpy().T
            knn = NearestNeighbors(n_neighbors=self.k + 1)
            knn.fit(X_np)
            distances, indices = knn.kneighbors(X_np)
        else:  # KNN in knn_cuda
            knn = KNN(k=self.k + 1, transpose_mode=True)
            X_T = self.X.T
            # if X is 2D, then unsqueeze(0) will add a dimension at the 0-th position for the batch size
            X_tend = X_T.unsqueeze(0)
            distances, indices = knn(X_tend, X_tend)
            # Lower the 3D tensor distances to 2D tensor
            distances = distances.squeeze(0).cpu().numpy()
            indices = indices.squeeze(0).cpu().numpy()

        self.max_gamma = 0.5 * np.max(distances)

        # Step 2: Construct adjacency matrix and weight matrix
        i_indices = np.repeat(np.arange(indices.shape[0]), self.k + 1)
        j_indices = indices.flatten()
        exp_distances = np.exp(-phi * distances.flatten() ** 2)


        # Construct sparse adjacency matrix and distance matrix
        adj_matrix = torch.zeros((self.n, self.n), dtype=torch.float64, device=self.device)
        adj_matrix[i_indices, j_indices] = 1
        adj_matrix[j_indices, i_indices] = 1

        dis_matrix = np.zeros((self.n, self.n))
        dis_matrix[i_indices, j_indices] = exp_distances
        dis_matrix[j_indices, i_indices] = exp_distances

        # Transform to PyTorch tensors
        dis_matrix = torch.tensor(dis_matrix, dtype=torch.float64, device=self.device)

        # Step 3: Extract non-zero weights and corresponding index pairs in the upper triangle
        row_indices, col_indices = torch.triu_indices(self.n, self.n, 1, device=self.device)
        upper_triangle = dis_matrix[row_indices, col_indices]
        non_zero_mask = upper_triangle > 0
        unique_pairs = torch.stack((row_indices[non_zero_mask], col_indices[non_zero_mask]), dim=1)
        weights = gamma * upper_triangle[non_zero_mask]

        # update class attributes
        self.unique_pairs = unique_pairs
        self.weightVector = weights.unsqueeze(0)  # shape: (1, num_weights)

        # Step 4: Normalize the weight vector (if enabled)
        if self.weight_norm_flag:
            scale_weight = math.sqrt(self.d) / torch.sum(self.weightVector)
            self.weightVector *= scale_weight

        self.num_weights = len(weights)

        # Step 5: Construct the node-arc matrix (NodeArcMatrix)
        NodeArc2 = torch.zeros((self.n, self.num_weights), dtype=torch.float64)
        NodeArc2[unique_pairs[:, 0], torch.arange(self.num_weights)] = 1
        NodeArc2[unique_pairs[:, 1], torch.arange(self.num_weights)] = -1
        self.nodeArcMatrix = NodeArc2.to(self.device).to_sparse_coo() if gv.use_coo else NodeArc2.to(
            self.device).to_sparse_csr()

        # Step 6: Compute the Laplacian matrix (ata_mat)
        degree_matrix = torch.diag(torch.sum(adj_matrix, axis=1))
        laplacian_matrix = degree_matrix - adj_matrix
        # self.L = torch.tensor(laplacian_matrix, dtype=torch.float64, device=self.device).to_sparse()
        self.ata_mat_dense = laplacian_matrix.clone().detach()
        if gv.use_coo == 1:
            self.ata_mat = self.ata_mat_dense.to_sparse_coo()
        else:
            self.ata_mat = self.ata_mat_dense.to_sparse_csr()
        # self.L = laplacian_matrix.clone().detach().to_sparse()

        time_end = time.perf_counter()
        gv.logger.info(f'Time taken to generate weight matrix = {time_end - time_start:.4f} s')

        # Construct the affinity matrix（ATA）
        # self.construct_amap()
        return

    def a_map(self, x):
        '''
        compute the affinity matrix Amap: @(x) x*A0
        :param x: dim: (d, n)
        :return: Amap :dim: (d, num_weights)
        '''
        a_map = torch.sparse.mm(x, self.nodeArcMatrix)
        return a_map

    def at_map(self, x):
        '''
        compute the affinity matrix ATmap: @(x) x*A0'
        :param x: dim: (d, num_weights)
        :return: ATmap: dim: (d, n)
        '''
        at_map = torch.sparse.mm(x, self.nodeArcMatrix.t())
        return at_map

    def ata_map(self, x):
        '''
        compute the affinity matrix ATAmap: @(x) x*ATAmat
        :param x: dim: (??? ,n)
        :return: ATAmap: dim: (???, n)
        '''
        at_map = torch.sparse.mm(x, self.ata_mat)
        return at_map

    def construct_amap(self):
        '''
        construct the affinity matrix
        :return:
        '''
        if gv.use_coo == 1:
            self.ata_mat = torch.sparse.mm(self.nodeArcMatrix, self.nodeArcMatrix.t())
        else:
            self.ata_mat = torch.sparse.mm(self.nodeArcMatrix, self.nodeArcMatrix.t().to_sparse_csr())
        self.ata_mat_dense = self.ata_mat.to_dense()
        return

    def to_device(self, device):
        """
       Transfers all tensor attributes to the specified device
       """
        for attr_name, attr_value in self.__dict__.items():
            # check if the attribute is a tensor
            if isinstance(attr_value, torch.Tensor):
                # transfer the tensor to the specified device
                setattr(self, attr_name, attr_value.to(device))
        return self

    def get_nodeArcMatrix(self):
        nodeArcMatrix1 = self.nodeArcMatrix.to('cpu').to_dense().numpy()
        # data_dict = {'X': nodeArcMatrix1}
        data_path = 'nodeArcMatrix1000.npy'
        np.save(data_path, nodeArcMatrix1)
        return self.nodeArcMatrix