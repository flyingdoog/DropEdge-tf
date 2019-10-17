# coding=utf-8
import numpy as np
import scipy.sparse as sp
from normalization import fetch_normalization

class Sampler:
    """Sampling the input graph data."""
    def __init__(self, adj, normalization, task_type="full"):
        self.adj = adj
        self.train_adj = adj

    def randomedge_sampler(self, percent):
        """
        Randomly drop edge and preserve percent% edges.
        """
        "Opt here"
        if percent >= 1.0:
            return self.adj
        
        nnz = len(self.train_adj[1])
        perm = np.random.permutation(nnz)
        preserve_nnz = int(nnz*percent)
        perm = perm[:preserve_nnz]
        test= self.train_adj[0][perm,0]
        r_adj = sp.coo_matrix((self.train_adj[1][perm],
                               (self.train_adj[0][perm,0],
                                self.train_adj[0][perm,1])),
                              shape=self.train_adj[2])
        return r_adj

    def vertex_sampler(self, percent, normalization):
        """
        Randomly drop vertexes.
        """
        print('not implemented yet')
        exit();

    def degree_sampler(self, percent, normalization):
        """
        Randomly drop edge wrt degree (high degree, low probility).
        """
        print('not implemented yet')
        exit();

