from rawData.IBRL.IBRL_cut import IBRL_cut_run
from rawData.IBRL.IBRL_load import get_IBRL_data, get_Node_MSG
from util.gen_adj import adj_TopK, adj_CoverDistance
import torch
from rawData.IBRL.IBRL_config import data_param
from modal.gen_net.GAN_net import Generator, Discriminator, get_gan_param
import numpy as np
import random
from GAN_cpu import gan
from util.utils import normalize

"""
——————————————————————————————————————————————————————————————————————
处理原生数据集，生成处理后的数据集文件（即用于给神经网络输入的数据集）
包括各种正例负例数据文件
——————————————————————————————————————————————————————————————————————
"""
args = data_param()  # 数据集处理参数
opt = get_gan_param()  # GAN参数


def IBRL_dataset():
    """
    处理原生数据集，生成处理后的数据集文件（即用于给神经网络输入的数据集）
    :return: 正例负例数据文件
    """
    IBRL_cut_run()
    orindata = get_IBRL_data()
    # print(orindata.shape)
    NodeMSG = get_Node_MSG()
    # print(NodeMSG.shape)
    adj = adj_TopK(NodeMSG)  # top_k
    # adj = adj_CoverDistance(NodeMSG)  # 覆盖距离
    # print(adj.shape)

    batch_num=5

    train_data = orindata[:, :, 0:int(orindata.shape[2] * args.train_rate)].copy()
    train_data = normalize(train_data, train_data)
    print(train_data.shape)
    inp_group = []
    # for i in range(batch_num):  # 组数，即batch数
    for i in range(train_data.shape[2] - opt.window_size + 1):
        z = torch.tensor(np.random.normal(0, 1, (opt.batch_size, opt.latent_dim))).float().cuda()
        inpdata = torch.tensor(train_data[:, :, i:int(i + opt.window_size)]).unsqueeze(0).float().cuda()
        # print(inpdata.shape)
        inp_group.append((inpdata, z))
    gan(inp_group, opt)  # 生成全图正负例

    ############## 对节点生成负例 ################

    # 生成任意需操作的节点序号
    neg_index = random.randint(0, 50)
    torch.save(neg_index, "data/batch2048window20/data_pt/neg_index.pt")
    # with open('data_n_pt/neg_index.txt', 'w') as fl:
    #     for line in neg_index:
    #         fl.write(str(line))
    #         fl.write('\n')

    # print(neg_index)
    neg = torch.load("data/batch2048window20/data_pt/negative.pt")
    posi = torch.load("data/batch2048window20/data_pt/positive.pt")
    for i in range(batch_num):
        num = random.randint(0, 2)
        # print(num)
        """
        num = 0时，取任意一种模态的信息进行负例操作
        num = 1时，取两种模态的信息进行负例操作
        num = 2时，所有模态的信息都进行负例操作
        """
        # 优化
        if num == 0:
            modelnum = random.randint(0, 2)
            neg_series = neg[i][modelnum][neg_index].detach().cpu().numpy()
            # print(neg_series)
            posi[i][modelnum][neg_index] = torch.from_numpy(neg_series)
        elif num == 1:
            neg_series = neg[i][0][neg_index].detach().cpu().numpy()
            posi[i][0][neg_index] = torch.from_numpy(neg_series)
            neg_series = neg[i][1][neg_index].detach().cpu().numpy()
            posi[i][1][neg_index] = torch.from_numpy(neg_series)
        # if num == 2:
        else:
            neg_series = neg[i][0][neg_index].detach().cpu().numpy()
            posi[i][0][neg_index] = torch.from_numpy(neg_series)
            neg_series = neg[i][1][neg_index].detach().cpu().numpy()
            posi[i][1][neg_index] = torch.from_numpy(neg_series)
            neg_series = neg[i][2][neg_index].detach().cpu().numpy()
            posi[i][2][neg_index] = torch.from_numpy(neg_series)
    torch.save(posi, "data/batch2048window20/data_pt/negative_n.pt")  # 存放负例

    torch.save(adj, "data/batch2048window20/data_pt/adj.pt")
    # print(adj.shape)
    # num=0
    # for i in range(51):
    #     for j in range(51):
    #        if adj[i][j]==1:
    #            num=num+1
    # print(num)

    ###############生成adj负例#################
    # a=0
    # target_node = random.randint(0, 50)
    change_node_list = random.sample(range(0, 50), args.change_node_num)
    # target_node=1
    # change_node_list[0]=1
    # print(adj[target_node][change_node_list[a]])
    # print(target_node,change_node_list)
    """
    改变所选取节点对的连接情况
    """
    for i in range(len(change_node_list)):
        if change_node_list[i] != neg_index:
            if adj[neg_index][change_node_list[i]] == 1:
                adj[neg_index][change_node_list[i]] = 0
                adj[change_node_list[i]][neg_index] = 0
            else:
                adj[neg_index][change_node_list[i]] = 1
                adj[change_node_list[i]][neg_index] = 1
        else:
            continue
    # print(target_node)
    # print(change_node_list[a])
    # print(adj[target_node][change_node_list[a]])
    print(args.change_node_num)
    torch.save(adj, "data/batch2048window20/data_pt/adj_negative.pt")



if __name__ == "__main__":
    IBRL_dataset()
