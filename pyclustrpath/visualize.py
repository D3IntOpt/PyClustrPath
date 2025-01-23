import torch
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def visualize_clustering_results(data: torch.Tensor, solution: torch.Tensor, gamma_list, label,**params):
    """
    Visualize the clustering path.
    :param data: The data tensor.
    :param solution: The solution tensor.
    :param gamma_list: The list of gamma.
    :param label: The label of the data.
    :param params:
    :return:
    """

    dim, n = data.shape
    path_length = len(gamma_list)
    XX = torch.zeros((dim, path_length+1, n))
    for i in range(path_length):
        XX[:, i, :] = solution[i]
    XX[:, path_length, :] = data

    # PCA
    XX_flat = XX.reshape(dim, -1).T # Reshape
    pca = PCA(n_components=3)
    XX3 = pca.fit_transform(XX_flat)
    XX3 = XX3.T.reshape(3, path_length + 1, n)

    # plot
    Labels = np.unique(label)
    label_num = len(Labels)
    # colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'w']
    # set label_num colors
    colors = plt.cm.rainbow(np.linspace(0, 1, label_num))
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    for i in range(n):
        ax.plot(XX3[0, :, i], XX3[1, :, i], XX3[2, :, i], color='black', linewidth=0.3)
    for k in range(label_num):
        k_id = np.where(label == Labels[k])[0]
        ax.scatter(XX3[0, -1, k_id], XX3[1, -1, k_id], XX3[2, -1, k_id], s=20, c=colors[k],
                   label=f'Object {Labels[k]}')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.savefig('cluster_path.png', dpi=300)
    plt.show()

    return