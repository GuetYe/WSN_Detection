import math

import torch
import numpy as np
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class SPConvlayer(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907

    构建空间卷积，类似GCN
    目的输入[batch,N,M,W]，输出[batch,N,M,W]

    """

    def __init__(self, node_num, modal_num, in_features, out_features1, out_features2,  bias=False, gpu=False):
        super(SPConvlayer, self).__init__()
        self.node_num = node_num  # 节点数量
        self.modal_num = modal_num  # 模态数量
        self.in_features = in_features  # 窗口大小
        self.out_features1 = out_features1
        self.out_features2 = out_features2
        self.relu = torch.nn.ReLU()
        self.gpu = gpu
        self.weight = Parameter(
            torch.rand(self.modal_num, self.out_features1, in_features, self.out_features2), requires_grad=True)  # input_features, out_features
        if bias:
            self.bias = Parameter(torch.rand(1, 1, self.out_features1, self.out_features2), requires_grad=True)
        else:
            self.register_parameter('bias', None)
        # self.reset_parameters()
        self.reset_param()
    #
    def reset_param(self):
            # for m in self.modules():
            #     if isinstance(m, (nn.Conv2d, nn.Linear)):
        torch.nn.init.kaiming_uniform_(self.weight)

    # def reset_parameters(self):
    #     """
    #     bias
    #     :return:
    #     """
    #     stdv = 1. / math.sqrt(self.weight.size(2))
    #     self.weight.data.uniform_(-stdv, stdv)  # 随机化参数
    #     if self.bias is not None:
    #         self.bias.data.uniform_(-stdv, stdv)

    # def add_I(self, a_adj):
    #     """
    #     添加单位矩阵
    #     :param a_adj:邻接矩阵
    #     :return:
    #     """
    #     x_label = a_adj.shape[0]
    #     y_label = a_adj.shape[1]
    #     for i in range(x_label):
    #         for j in range(y_label):
    #             if i == j and a_adj[i][j] == 0:
    #                 a_adj[i][j] = a_adj[i][j] + 1
    #     return a_adj

    # def get_d(self, ai_adj):
    #     """
    #     获得度矩阵,按行求和
    #     :param ai_adj:A+I 添加了自身信息的邻接矩阵
    #     :return:
    #     """
    #     x_label = ai_adj.shape[0]
    #     y_label = ai_adj.shape[1]
    #     for i in range(x_label):
    #         sum = 0
    #         for j in range(y_label):
    #             sum += ai_adj[i][j]
    #         ai_adj[i][i] = sum
    #     for i in range(x_label):
    #         for j in range(y_label):
    #             if i != j:
    #                 ai_adj[i][j] = 0
    #     return ai_adj

    # def normalize_adj(self, adj, d_matrix):
    #     """
    #     归一化
    #     :param adj:
    #     :param d_matrix:度矩阵
    #     :return:
    #     """
    #     try:
    #         d_inv_sqrt = np.power(d_matrix, 0.5)
    #         d_nor = np.linalg.inv(d_inv_sqrt)
    #     except:
    #         print("矩阵不存在逆矩阵")
    #     else:
    #         return np.array(adj).dot(d_nor).transpose().dot(d_nor)
    #     # try:
    #     #     d_inv_sqrt = np.power(d_matrix, -0.5)
    #     #     d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    #     # except:
    #     #     print("矩阵不存在逆矩阵")
    #     #     d_inv_sqrt = np.ones_like(d_matrix) * 1e-6
    #     # return np.matmul(np.matmul(d_inv_sqrt, adj), d_inv_sqrt)
    def normalize(self, mx):
        """Row-normalize sparse matrix"""
        if self.gpu ==False:
            mx=mx + torch.eye(mx.shape[1]).unsqueeze(0).repeat(mx.shape[0],1,1)
        else:
            mx=mx + torch.eye(mx.shape[1]).unsqueeze(0).repeat(mx.shape[0],1,1).cuda()
        # print(mx)
        rowsum = torch.sum(mx, dim=(2))  #  矩阵行求和
        # print("矩阵行求和", rowsum)
        r_inv = torch.pow(rowsum, -1)  # 求和的-1次方
        # print("求和的-1次方", r_inv)
        r_inv[torch.isinf(r_inv)] = 0.   # 如果是inf，转换成0
        # print("如果是inf，转换成0", r_inv)
        r_mat_inv = torch.diag_embed(r_inv)  # 构造对角戏矩阵
        # print("构造对角戏矩阵", r_mat_inv)
        mx = torch.einsum("abc,acd->abd", r_mat_inv, mx)  # 构造D-1*A，非对称方式，简化方式
        return mx

    # def add_self_link(self, adj):
    #     """
    #     获得度矩阵归一化后的邻接矩阵
    #     :param adj:
    #     :return:
    #     """
    #     adj_copy = adj.clone() 
    #     for i in range(adj_copy.shape[0]):
    #         # print(adj_copy[i])
    #         AI_adj = self.add_I(adj_copy[i])  # A+I
    #         # print(AI_adj.shape)
    #         D_matrix = self.get_d(AI_adj)+ 1e-6 
    #         adj_dad = self.normalize_adj(AI_adj, D_matrix)
    #         adj[i] = torch.tensor(adj_dad)
    #     # print("ai",adj.shape)
    #     return adj
    """
    ——————————————————————————————————————————————————————————————————————
    这个错误通常是由于在进行反向传播时，出现了所谓的原位操作（inplace operation）。这些操作会直接修改张量，而不会创建新的张量，
    并且可能导致梯度计算失效。为了解决这个问题，您可以使用PyTorch提供的类似clone()或detach()这样的函数，
    将需要更新的张量复制到一个新张量中，并在复制后的张量上执行原位操作。
    ——————————————————————————————————————————————————————————————————————
    """

    def forward(self, x, adj):
        # [batch,node,modal,window_time] [batch,modal,window_time,window_time]-->[batch,node,modal,window_time]
        h = torch.einsum('bnma,mpaw->bnpw', x,
                         self.weight)
        # [batch,node,node] [batch,node,modal,window_time]-->[batch,node,modal,window_time]
        adj_copy = adj # 复制y到一个新的张量中
        # print(adj_copy[0])
        adj_copy=self.normalize(adj_copy)
        # print(adj_copy[0],"bianbianbian")
        output = self.relu(torch.einsum('bnc,bcmw->bnmw', adj_copy,
                              h))
    
        if self.bias is not None:
            return output + self.bias
        else:
            return output
    
    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

    """
    ——————————————————————————————————————————————————————————————————————
    修改后： 将对adj处理的函数写在forward里
    ——————————————————————————————————————————————————————————————————————
    """

    # def forward(self, x, adj):
    #     # [batch,node,modal,window_time] [batch,modal,window_time,window_time]-->[batch,node,modal,window_time]
    #     h = torch.einsum('bnma,maw->bnmw', x,
    #                      self.weight)
    #     # [batch,node,node] [batch,node,modal,window_time]-->[batch,node,modal,window_time]
    #     for b in range(adj.shape[0]):

    #         x_label = adj[b].shape[0]
    #         print(x_label)
    #         y_label = adj[b].shape[1]
    #         print(y_label)

    #         # 加单位矩阵
    #         for i in range(x_label):
    #             for j in range(y_label):
    #                 if i == j and adj[b][i][j] == 0:
    #                     adj[b][i][j] = adj[b][i][j] + 1

    #         AI_adj=adj[b]
    #         print(AI_adj.shape)
    #         # 获得度矩阵
    #         for i in range(x_label):
    #             sum = 0
    #             for j in range(y_label):
    #                 sum += adj[b][i][j]
    #             adj[b][i][i] = sum
    #         for i in range(x_label):
    #             for j in range(y_label):
    #                 if i != j:
    #                     adj[b][i][j] = 0
    #         D_matrix=adj[b]
    #         print(D_matrix.shape)

    #         # try:
    #         #     d_inv_sqrt = np.power(D_matrix, 0.5)
    #         #     d_nor = np.linalg.pinv(d_inv_sqrt)
    #         # except:
    #         #     print("矩阵不存在逆矩阵")
    #         # else:
    #         #     adj[b]= torch.tensor(np.array(AI_adj).dot(d_nor).transpose().dot(d_nor))# D（A+I）D
    #         try:
    #             d_inv_sqrt = np.power(D_matrix, -0.5)
    #             d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    #         except:
    #             print("矩阵不存在逆矩阵")
    #             d_inv_sqrt = np.ones_like(D_matrix) * 1e-6
    #         return np.matmul(np.matmul(d_inv_sqrt, adj), d_inv_sqrt)

    #     print(adj.shape)
    #     output = torch.einsum('bnc,bcmw->bnmw', adj,
    #                           h)

    #     if self.bias is not None:
    #         return output + self.bias
    #     else:
    #         return output


    # def __repr__(self):
    #     return self.__class__.__name__ + ' (' \
    #         + str(self.in_features) + ' -> ' \
    #         + str(self.out_features) + ')'


if __name__ == '__main__':
    torch.set_printoptions(profile="full")
    SPConv = SPConvlayer(51, 12, 10, 24, 5, False,False)

    x = torch.rand(5, 51, 12, 10)
    adj = torch.rand(5, 51, 51).fill_(1.0)
    # x = torch.load("data/batch5window20/data_pt/positive.pt").permute(0, 2, 1, 3).float().cuda() # [batch,N,M,W]
    # adj = torch.load("data/batch5window20/data_pt/adj.pt")  # [N,N]
    # # print(type(adj))
    # adj = torch.tensor(adj).expand(x.shape[0], 51,51).float().cuda()  # [batch,N,N]

    out = SPConv(x, adj)
    print(out.shape)

    """
    ————————————————————————————
    以下为测试用例
    ————————————————————————————
    """
    # seed = 3
    # torch.manual_seed(seed)  # 固定随机种子（CPU）
    # if torch.cuda.is_available():  # 固定随机种子（GPU)
    #     torch.cuda.manual_seed(seed)  # 为当前GPU设置
    #     torch.cuda.manual_seed_all(seed)  # 为所有GPU设置
    # np.random.seed(seed)  # 保证后续使用random函数时，产生固定的随机数
    #
    # # ——test sample1 ——#
    # a = torch.randint(2, (2, 2, 2))  # batch个adj,即 [batch, nodenum, nodenum]
    # print("a", a)
    # b = torch.randint(6, (2, 2, 3, 4))  # batch个属性张量,即 [batch, nodenum, modalnum, timestamp]
    # print("b", b)
    # print(torch.einsum('bnc,bcwa->bnwa', a, b).shape)
    # print(torch.einsum('bnc,bcwa->bnwa', a, b))
    #
    # print("#####\n\n###########")
    # # ——test sample2 ——#
    # h = torch.einsum('bnc,bcwa->bnwa', a, b)  # batch个属性张量,即 [batch, nodenum, modalnum, timestamp]
    # print("h", h)
    # w = torch.randint(2, (3, 4, 4))  # batch个聚合权重,即 [ modalnum, timestamp, timestamp]
    # print("W", w)
    # print(torch.einsum('bnwa,wak->bnwk', h, w).shape)
    # print(torch.einsum('bnwa,wak->bnwk', h, w))

    # ####乘积先后可变 #######
    # a = torch.randint(2, (2, 2, 2))  # batch个adj,即 [batch, nodenum, nodenum]
    # print("a", a)
    # b = torch.randint(6, (2, 2, 3, 4))  # batch个属性张量,即 [batch, nodenum, modalnum, timestamp]
    # print("b", b)
    # w = torch.randint(2, (3, 4, 4))  # batch个聚合权重,即 [batch, modalnum, timestamp, timestamp]
    #
    # h = torch.einsum('bcwa,wak->bcwk', b, w)
    # print(torch.einsum('bnc,bcwa->bnwa', a, h))
    # h = torch.einsum('bnc,bcwa->bnwa', a, b)
    # print(torch.einsum('bcwa,wak->bcwk', h, w))
