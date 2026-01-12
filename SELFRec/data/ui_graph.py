import numpy as np
from collections import defaultdict
import sys

sys.path.append('/home/fengxs/explainLLMRec/SELFRec')
from data.data import Data
from data.graph import Graph
import scipy.sparse as sp
import pickle


class Interaction(Data, Graph):
    def __init__(self, conf, training, test):
        Graph.__init__(self)
        Data.__init__(self, conf, training, test)

        self.user = {}
        self.item = {}
        self.id2user = {}
        self.id2item = {}
        self.training_set_u = defaultdict(dict)
        self.training_set_i = defaultdict(dict)
        self.test_set = defaultdict(dict)
        self.test_set_item = set()
        self.__generate_set()
        self.user_num = len(self.training_set_u)
        self.item_num = len(self.training_set_i)
        self.ui_adj = self.__create_sparse_bipartite_adjacency()
        self.u_adj, self.i_adj = self.__create_sparse_bipartite_adjacency_u_n_i()
        self.norm_adj, _ = self.normalize_graph_mat(self.ui_adj)
        self.interaction_mat = self.__create_sparse_interaction_matrix()
        self.u_adj_test, self.i_adj_test = self.__test_mat_generation()
        # popularity_user = {}
        # for u in self.user:
        #     popularity_user[self.user[u]] = len(self.training_set_u[u])
        # popularity_item = {}
        # for u in self.item:
        #     popularity_item[self.item[u]] = len(self.training_set_i[u])

    def __generate_set(self):
        for entry in self.training_data:
            user, item, timestamp, rating = entry
            if user not in self.user:
                self.user[user] = len(self.user)
                self.id2user[self.user[user]] = user
            if item not in self.item:
                self.item[item] = len(self.item)
                self.id2item[self.item[item]] = item
                # userList.append
            self.training_set_u[user][item] = (rating, timestamp)
            self.training_set_i[item][user] = (rating, timestamp)
        for entry in self.test_data:
            user, item, timestamp, rating = entry
            if user not in self.user or item not in self.item:
                continue
            self.test_set[user][item] = (rating, timestamp)
            self.test_set_item.add(item)

    def __create_sparse_bipartite_adjacency(self, self_connection=False):
        '''
        return a sparse adjacency matrix with the shape (user number + item number, user number + item number)
        '''
        n_nodes = self.user_num + self.item_num
        row_idx = [self.user[pair[0]] for pair in self.training_data]
        col_idx = [self.item[pair[1]] for pair in self.training_data]
        user_np = np.array(row_idx)
        item_np = np.array(col_idx)
        ratings = np.ones_like(user_np, dtype=np.float32)
        tmp_adj = sp.csr_matrix((ratings, (user_np, item_np + self.user_num)), shape=(n_nodes, n_nodes),
                                dtype=np.float32)
        adj_mat = tmp_adj + tmp_adj.T
        if self_connection:
            adj_mat += sp.eye(n_nodes)
        return adj_mat

    def __create_sparse_bipartite_adjacency_u_n_i(self, self_connection=False):
        '''
        return a sparse adjacency matrix with the shape (user number + item number, user number + item number)
        '''
        # n_nodes = self.user_num + self.item_num
        # row_idx = [self.user[pair[0]] for pair in self.training_data]
        # col_idx = [self.item[pair[1]] for pair in self.training_data]
        # user_np = np.array(row_idx)
        # item_np = np.array(col_idx)
        # ratings = np.ones_like(user_np, dtype=np.float32)
        # tmp_adj = sp.csr_matrix((ratings, (user_np, item_np + self.user_num)), shape=(n_nodes, n_nodes),dtype=np.float32)
        # adj_mat = tmp_adj + tmp_adj.T
        # if self_connection:
        #     adj_mat += sp.eye(n_nodes)
        # return tmp_adj, tmp_adj.T

        n_nodes = self.user_num + self.item_num
        row_idx = [self.user[pair[0]] for pair in self.training_data]
        col_idx = [self.item[pair[1]] for pair in self.training_data]
        user_np = np.array(row_idx)
        item_np = np.array(col_idx)
        ratings = np.ones_like(user_np, dtype=np.float32)

        # 创建用户和物品之间的邻接矩阵
        tmp_adj = sp.csr_matrix((ratings, (user_np, item_np + self.user_num)), shape=(n_nodes, n_nodes),
                                dtype=np.float32)

        if self_connection:
            tmp_adj += sp.eye(n_nodes)

        # 用户到物品的邻接矩阵
        user_to_item_adj = tmp_adj[:self.user_num, self.user_num:]

        # 物品到用户的邻接矩阵
        item_to_user_adj = tmp_adj.T[self.user_num:, :self.user_num]

        return user_to_item_adj, item_to_user_adj

    def __test_mat_generation(self, self_connection=False):
        '''
        return a sparse adjacency matrix with the shape (user number + item number, user number + item number)
        '''
        # n_nodes = self.user_num + self.item_num
        # row_idx = [self.user[pair[0]] for pair in self.training_data]
        # col_idx = [self.item[pair[1]] for pair in self.training_data]
        # user_np = np.array(row_idx)
        # item_np = np.array(col_idx)
        # ratings = np.ones_like(user_np, dtype=np.float32)
        # tmp_adj = sp.csr_matrix((ratings, (user_np, item_np + self.user_num)), shape=(n_nodes, n_nodes),dtype=np.float32)
        # adj_mat = tmp_adj + tmp_adj.T
        # if self_connection:
        #     adj_mat += sp.eye(n_nodes)
        # return tmp_adj, tmp_adj.T

        n_nodes = self.user_num + self.item_num
        # keys = [int(a) for a in self.item.keys()]
        # print(self.item)
        # 如果 pair[0] 在 self.user 中且 pair[1] 在 self.item 中才进行操作
        row_idx = [self.user[pair[0]] for pair in self.test_data if pair[0] in self.user and pair[1] in self.item]
        col_idx = [self.item[pair[1]] for pair in self.test_data if pair[0] in self.user and pair[1] in self.item]

        user_np = np.array(row_idx)
        item_np = np.array(col_idx)
        ratings = np.ones_like(user_np, dtype=np.float32)
        # print(len(ratings), len(user_np), len(item_np))

        # 创建用户和物品之间的邻接矩阵
        tmp_adj = sp.csr_matrix((ratings, (user_np, item_np + self.user_num)), shape=(n_nodes, n_nodes),
                                dtype=np.float32)

        if self_connection:
            tmp_adj += sp.eye(n_nodes)

        # 用户到物品的邻接矩阵
        user_to_item_adj = tmp_adj[:self.user_num, self.user_num:]

        # 物品到用户的邻接矩阵
        item_to_user_adj = tmp_adj.T[self.user_num:, :self.user_num]

        return user_to_item_adj, item_to_user_adj

    def convert_to_laplacian_mat(self, adj_mat):
        adj_shape = adj_mat.get_shape()
        n_nodes = adj_shape[0] + adj_shape[1]
        (user_np_keep, item_np_keep) = adj_mat.nonzero()
        ratings_keep = adj_mat.data
        tmp_adj = sp.csr_matrix((ratings_keep, (user_np_keep, item_np_keep + adj_shape[0])), shape=(n_nodes, n_nodes),
                                dtype=np.float32)
        tmp_adj = tmp_adj + tmp_adj.T
        return self.normalize_graph_mat(tmp_adj)

    def __create_sparse_interaction_matrix(self):
        """
        return a sparse adjacency matrix with the shape (user number, item number)
        """
        row, col, entries = [], [], []
        for pair in self.training_data:
            row += [self.user[pair[0]]]
            col += [self.item[pair[1]]]
            entries += [1.0]
        interaction_mat = sp.csr_matrix((entries, (row, col)), shape=(self.user_num, self.item_num), dtype=np.float32)
        return interaction_mat

    def get_user_id(self, u):
        if u in self.user:
            return self.user[u]

    def get_item_id(self, i):
        if i in self.item:
            return self.item[i]

    def training_size(self):
        return len(self.user), len(self.item), len(self.training_data)

    def test_size(self):
        return len(self.test_set), len(self.test_set_item), len(self.test_data)

    def contain(self, u, i):
        'whether user u rated item i'
        if u in self.user and i in self.training_set_u[u]:
            return True
        else:
            return False

    def contain_user(self, u):
        'whether user is in training set'
        if u in self.user:
            return True
        else:
            return False

    def contain_item(self, i):
        """whether item is in training set"""
        if i in self.item:
            return True
        else:
            return False

    def user_rated(self, u):
        return list(self.training_set_u[u].keys()), list(self.training_set_u[u].values())

    def item_rated(self, i):
        return list(self.training_set_i[i].keys()), list(self.training_set_i[i].values())

    def row(self, u):
        u = self.id2user[u]
        k, v = self.user_rated(u)
        vec = np.zeros(len(self.item))
        # print vec
        for pair in zip(k, v):
            iid = self.item[pair[0]]
            vec[iid] = pair[1]
        return vec

    def col(self, i):
        i = self.id2item[i]
        k, v = self.item_rated(i)
        vec = np.zeros(len(self.user))
        # print vec
        for pair in zip(k, v):
            uid = self.user[pair[0]]
            vec[uid] = pair[1]
        return vec

    def matrix(self):
        m = np.zeros((len(self.user), len(self.item)))
        for u in self.user:
            k, v = self.user_rated(u)
            vec = np.zeros(len(self.item))
            # print vec
            for pair in zip(k, v):
                iid = self.item[pair[0]]
                vec[iid] = pair[1]
            m[self.user[u]] = vec
        return m

    def create_adj_mat(self):
        adj_mat = sp.dok_matrix((self.user_num + self.item_num, self.user_num + self.item_num), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        self.R = sp.dok_matrix((self.user_num, self.item_num), dtype=np.float32)
        R = self.R.tolil()

        adj_mat[:self.user_num, self.user_num:] = R
        adj_mat[self.user_num:, :self.user_num] = R.T
        adj_mat = adj_mat.todok()

        def normalized_adj_single(adj):
            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj)
            # norm_adj = adj.dot(d_mat_inv)
            print('generate single-normalized adjacency matrix.')
            return norm_adj.tocoo()

        def get_D_inv(adj):
            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)
            return d_mat_inv

        def check_adj_if_equal(adj):
            dense_A = np.array(adj.todense())
            degree = np.sum(dense_A, axis=1, keepdims=False)

            temp = np.dot(np.diag(np.power(degree, -1)), dense_A)
            print('check normalized adjacency matrix whether equal to this laplacian matrix.')
            return temp

        norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
        mean_adj_mat = normalized_adj_single(adj_mat)

        return adj_mat.tocsr(), norm_adj_mat.tocsr(), mean_adj_mat.tocsr()


class InteractionPlus(Data, Graph):
    def __init__(self, conf, training, test, filter_low=False, filter_high=False):
        Graph.__init__(self)
        Data.__init__(self, conf, training, test)

        self.user = {}
        self.item = {}
        self.id2user = {}
        self.id2item = {}
        self.training_set_u = defaultdict(dict)
        self.training_set_i = defaultdict(dict)
        self.test_set = defaultdict(dict)
        self.test_set_item = set()
        self.__generate_set()
        self.user_num = len(self.training_set_u)
        self.item_num = len(self.training_set_i)
        self.ui_adj = self.__create_sparse_bipartite_adjacency(filter_low=filter_low, filter_high=filter_high)
        self.u_adj, self.i_adj = self.__create_sparse_bipartite_adjacency_u_n_i()
        self.norm_adj, self.deg = self.normalize_graph_mat(self.ui_adj)
        self.interaction_mat = self.__create_sparse_interaction_matrix()
        self.u_adj_test, self.i_adj_test = self.__test_mat_generation()
        # a,b = self.__generate_ratings_timestamps()

    def __generate_set(self):
        for entry in self.training_data:
            user, item, timestamp, rating = entry
            if user not in self.user:
                self.user[user] = len(self.user)
                self.id2user[self.user[user]] = user
            if item not in self.item:
                self.item[item] = len(self.item)
                self.id2item[self.item[item]] = item
                # userList.append
            self.training_set_u[user][item] = (rating, timestamp)
            self.training_set_i[item][user] = (rating, timestamp)
        for entry in self.test_data:
            user, item, timestamp, rating = entry
            if user not in self.user or item not in self.item:
                continue
            self.test_set[user][item] = (rating, timestamp)
            self.test_set_item.add(item)

    def __create_sparse_bipartite_adjacency(self, self_connection=False, filter_high=False, filter_low=False):
        '''
        return a sparse adjacency matrix with the shape (user number + item number, user number + item number)
        '''
        # if filter_low:
        #     filtered_training_data = [pair for pair in self.training_data if pair[3] >= 3]
        #     row_idx = [self.user[pair[0]] for pair in filtered_training_data]
        #     col_idx = [self.item[pair[1]] for pair in filtered_training_data]
        # elif filter_high:
        #     filtered_training_data = [pair for pair in self.training_data if pair[3] < 3]
        #     row_idx = [self.user[pair[0]] for pair in filtered_training_data]
        #     col_idx = [self.item[pair[1]] for pair in filtered_training_data]
        # else:
        row_idx = [self.user[pair[0]] for pair in self.training_data]
        col_idx = [self.item[pair[1]] for pair in self.training_data]
        n_nodes = self.user_num + self.item_num
        user_np = np.array(row_idx)
        item_np = np.array(col_idx)
        ratings = np.ones_like(user_np, dtype=np.float32)
        tmp_adj = sp.csr_matrix((ratings, (user_np, item_np + self.user_num)), shape=(n_nodes, n_nodes),
                                dtype=np.float32)
        adj_mat = tmp_adj + tmp_adj.T
        if self_connection:
            adj_mat += sp.eye(n_nodes)
        return adj_mat


    def __create_sparse_bipartite_adjacency_u_n_i(self, self_connection=False):
        '''
        return a sparse adjacency matrix with the shape (user number + item number, user number + item number)
        '''
        # n_nodes = self.user_num + self.item_num
        # row_idx = [self.user[pair[0]] for pair in self.training_data]
        # col_idx = [self.item[pair[1]] for pair in self.training_data]
        # user_np = np.array(row_idx)
        # item_np = np.array(col_idx)
        # ratings = np.ones_like(user_np, dtype=np.float32)
        # tmp_adj = sp.csr_matrix((ratings, (user_np, item_np + self.user_num)), shape=(n_nodes, n_nodes),dtype=np.float32)
        # adj_mat = tmp_adj + tmp_adj.T
        # if self_connection:
        #     adj_mat += sp.eye(n_nodes)
        # return tmp_adj, tmp_adj.T

        n_nodes = self.user_num + self.item_num
        row_idx = [self.user[pair[0]] for pair in self.training_data]
        col_idx = [self.item[pair[1]] for pair in self.training_data]
        user_np = np.array(row_idx)
        item_np = np.array(col_idx)
        ratings = np.ones_like(user_np, dtype=np.float32)

        # 创建用户和物品之间的邻接矩阵
        tmp_adj = sp.csr_matrix((ratings, (user_np, item_np + self.user_num)), shape=(n_nodes, n_nodes),
                                dtype=np.float32)

        if self_connection:
            tmp_adj += sp.eye(n_nodes)

        # 用户到物品的邻接矩阵
        user_to_item_adj = tmp_adj[:self.user_num, self.user_num:]

        # 物品到用户的邻接矩阵
        item_to_user_adj = tmp_adj.T[self.user_num:, :self.user_num]

        return user_to_item_adj, item_to_user_adj


    def __test_mat_generation(self, self_connection=False):
        '''
        return a sparse adjacency matrix with the shape (user number + item number, user number + item number)
        '''
        # n_nodes = self.user_num + self.item_num
        # row_idx = [self.user[pair[0]] for pair in self.training_data]
        # col_idx = [self.item[pair[1]] for pair in self.training_data]
        # user_np = np.array(row_idx)
        # item_np = np.array(col_idx)
        # ratings = np.ones_like(user_np, dtype=np.float32)
        # tmp_adj = sp.csr_matrix((ratings, (user_np, item_np + self.user_num)), shape=(n_nodes, n_nodes),dtype=np.float32)
        # adj_mat = tmp_adj + tmp_adj.T
        # if self_connection:
        #     adj_mat += sp.eye(n_nodes)
        # return tmp_adj, tmp_adj.T

        n_nodes = self.user_num + self.item_num
        # keys = [int(a) for a in self.item.keys()]
        # print(self.item)
        # 如果 pair[0] 在 self.user 中且 pair[1] 在 self.item 中才进行操作
        row_idx = [self.user[pair[0]] for pair in self.test_data if pair[0] in self.user and pair[1] in self.item]
        col_idx = [self.item[pair[1]] for pair in self.test_data if pair[0] in self.user and pair[1] in self.item]

        user_np = np.array(row_idx)
        item_np = np.array(col_idx)
        ratings = np.ones_like(user_np, dtype=np.float32)
        # print(len(ratings), len(user_np), len(item_np))

        # 创建用户和物品之间的邻接矩阵
        tmp_adj = sp.csr_matrix((ratings, (user_np, item_np + self.user_num)), shape=(n_nodes, n_nodes),
                                dtype=np.float32)

        if self_connection:
            tmp_adj += sp.eye(n_nodes)

        # 用户到物品的邻接矩阵
        user_to_item_adj = tmp_adj[:self.user_num, self.user_num:]

        # 物品到用户的邻接矩阵
        item_to_user_adj = tmp_adj.T[self.user_num:, :self.user_num]

        return user_to_item_adj, item_to_user_adj


    def convert_to_laplacian_mat(self, adj_mat):
        adj_shape = adj_mat.get_shape()
        n_nodes = adj_shape[0] + adj_shape[1]
        (user_np_keep, item_np_keep) = adj_mat.nonzero()
        ratings_keep = adj_mat.data
        tmp_adj = sp.csr_matrix((ratings_keep, (user_np_keep, item_np_keep + adj_shape[0])), shape=(n_nodes, n_nodes),
                                dtype=np.float32)
        tmp_adj = tmp_adj + tmp_adj.T
        return self.normalize_graph_mat(tmp_adj)


    def __create_sparse_interaction_matrix(self):
        """
        return a sparse adjacency matrix with the shape (user number, item number)
        """
        row, col, entries = [], [], []
        for pair in self.training_data:
            row += [self.user[pair[0]]]
            col += [self.item[pair[1]]]
            entries += [1.0]
        interaction_mat = sp.csr_matrix((entries, (row, col)), shape=(self.user_num, self.item_num), dtype=np.float32)
        return interaction_mat


    def get_user_id(self, u):
        if u in self.user:
            return self.user[u]


    def get_item_id(self, i):
        if i in self.item:
            return self.item[i]


    def training_size(self):
        return len(self.user), len(self.item), len(self.training_data)


    def test_size(self):
        return len(self.test_set), len(self.test_set_item), len(self.test_data)


    def contain(self, u, i):
        'whether user u rated item i'
        if u in self.user and i in self.training_set_u[u]:
            return True
        else:
            return False


    def contain_user(self, u):
        'whether user is in training set'
        if u in self.user:
            return True
        else:
            return False


    def contain_item(self, i):
        """whether item is in training set"""
        if i in self.item:
            return True
        else:
            return False


    def user_rated(self, u):
        return list(self.training_set_u[u].keys()), list(self.training_set_u[u].values())


    def item_rated(self, i):
        return list(self.training_set_i[i].keys()), list(self.training_set_i[i].values())


    def row(self, u):
        u = self.id2user[u]
        k, v = self.user_rated(u)
        vec = np.zeros(len(self.item))
        # print vec
        for pair in zip(k, v):
            iid = self.item[pair[0]]
            vec[iid] = pair[1]
        return vec


    def col(self, i):
        i = self.id2item[i]
        k, v = self.item_rated(i)
        vec = np.zeros(len(self.user))
        # print vec
        for pair in zip(k, v):
            uid = self.user[pair[0]]
            vec[uid] = pair[1]
        return vec


    def matrix(self):
        m = np.zeros((len(self.user), len(self.item)))
        for u in self.user:
            k, v = self.user_rated(u)
            vec = np.zeros(len(self.item))
            # print vec
            for pair in zip(k, v):
                iid = self.item[pair[0]]
                vec[iid] = pair[1]
            m[self.user[u]] = vec
        return m


    def create_adj_mat(self):
        adj_mat = sp.dok_matrix((self.user_num + self.item_num, self.user_num + self.item_num), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        self.R = sp.dok_matrix((self.user_num, self.item_num), dtype=np.float32)
        R = self.R.tolil()

        adj_mat[:self.user_num, self.user_num:] = R
        adj_mat[self.user_num:, :self.user_num] = R.T
        adj_mat = adj_mat.todok()

        def normalized_adj_single(adj):
            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj)
            # norm_adj = adj.dot(d_mat_inv)
            print('generate single-normalized adjacency matrix.')
            return norm_adj.tocoo()

        def get_D_inv(adj):
            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)
            return d_mat_inv

        def check_adj_if_equal(adj):
            dense_A = np.array(adj.todense())
            degree = np.sum(dense_A, axis=1, keepdims=False)

            temp = np.dot(np.diag(np.power(degree, -1)), dense_A)
            print('check normalized adjacency matrix whether equal to this laplacian matrix.')
            return temp

        norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
        mean_adj_mat = normalized_adj_single(adj_mat)

        return adj_mat.tocsr(), norm_adj_mat.tocsr(), mean_adj_mat.tocsr()


if __name__ == "__main__":
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    sys.path.append('/home/fengxs/explainLLMRec/SELFRec')
    from util.sampler import next_batch_pairwise, next_user_samples, next_item_samples
    from base.torch_interface import TorchGraphInterface
    from util.loss_torch import bpr_loss, l2_reg_loss
    from base.graph_recommender import GraphRecommender
    import json
    from data.loader import FileIO
    from util.conf import ModelConf

    model = 'LightGCN'

    conf = ModelConf('/home/fengxs/explainLLMRec/SELFRec/conf/' + model + '.conf')
    training_data = FileIO.load_data_set(conf['training.set'], conf['model.type'])
    test_data = FileIO.load_data_set(conf['test.set'], conf['model.type'])
    # DEVICE = f'cuda:{args.device}'
    # linear_list = np.linspace(0, 0.5, 26).tolist()

    # for pair_rate in linear_list:
    a = InteractionPlus(conf, training_data, test_data)
