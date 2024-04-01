import  numpy as np
import torch,os,random
from model.new import GNN
# from torch_geometric.loader import DataLoader
from torch.utils.data.dataloader import DataLoader
import torch.nn as nn
from tqdm import tqdm
from util.utils import combine_graph, normalize, get_adj_wei

from Config import parse_args,get_data_inf


config = parse_args()
epoch = config.epoch
node_num,modal_num = get_data_inf(config.dataset)

use_cuda = config.gpu
device = "cpu"
if use_cuda:
    device = "cuda"


model = GNN(modal_num,node_num,config.window_size).to(device)
# model.load_state_dict(torch.load("save/para_CIMIS.pth"))
train_normal = torch.load("processed/"+config.dataset+"/train_normal.pt")
loader = DataLoader(train_normal,shuffle=False,batch_size=1)

file_list = os.listdir("processed/"+config.dataset+"/")
train_anomaly_groups = []
for filename in file_list:
    if filename[0:14] == "train_abnormal":
        print(filename,"filename")
        train_anomaly_groups.append(torch.load("processed/"+config.dataset+"/"+filename))
# print(len(train_anomaly_groups))
anomaly_loaders = []
for group in train_anomaly_groups:
    anomaly_loaders.append(DataLoader(group,shuffle=False,batch_size=1))
# print(len(anomaly_loaders))

optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

cf_criteon = nn.BCELoss().to(device)
rec_criteon = nn.MSELoss().to(device)

normal_loss_list = np.array([0 for i in range(len(train_normal))])
abnorm_loss_list = np.array([0 for i in range(len(train_normal))])
# print(loss_list.shape)
normal_label = torch.tensor([[1.0,0.0]]).to(device)

loss_save_file = open("save/loss_a%s.txt" %config.dataset,"w",encoding="utf8")

adj = torch.ones(4, 51, 51).float().cuda()
for epochnum in range(epoch):
    ####----train the normal data----#####
    H_ = (torch.ones(2, config.batch_size, 16).to(device),
          torch.ones(2, config.batch_size, 16).to(device))
    print("epoch_num ",epochnum)
    ct=0
    for i in tqdm(loader):
        inp = i[0].squeeze(0).float().to(device)
        # print(inp.shape)
        cf_label = i[1].squeeze(0).float().to(device)
        # print(label.shape)
        # cf_label = i[3].squeeze(0).to(device)
        if i[2][0] != H_[0].shape[1]:
            H_[0] = H_[0][:, H_[0].shape[1] - i[2][0]:H_[0].shape[1], :].contiguous()
            H_[1] = H_[1][:, H_[1].shape[1] - i[2][0]:H_[1].shape[1], :].contiguous()
        # print(inp.shape, adj.shape, H_[0].shape)
        cf_out, new_H = model(inp,adj, H_)
        # print(cf_out.shape, new_H[0].shape)
        H_ = (new_H[0].detach(),new_H[1].detach())
        # print(cf_out.shape, cf_label.shape)
        cf_loss = cf_criteon(cf_out,cf_label)
        lossvalue = cf_loss #+rec_loss
    #     # print(type(cf_loss),type(rec_loss))

        optimizer.zero_grad()
        lossvalue.backward()
        optimizer.step()
        normal_loss_list[ct] = lossvalue.item()
        ct = ct+1
    loss_save_file.write("epoch%s: normal_patch_loss %s" %(epochnum,np.sum(normal_loss_list)/normal_loss_list.shape[0]))
    print(np.sum(normal_loss_list)/normal_loss_list.shape[0])
    torch.save(H_,"save/save_a_H_%s.pth" %config.dataset)

    random.shuffle(anomaly_loaders)
    ab_loss_res = []
    ####----train the abnormal data----#####
    for anomaly_loader in anomaly_loaders[0:2]:
        H_ = (torch.ones(2, config.batch_size, 16).to(device),
              torch.ones(2, config.batch_size, 16).to(device))
        ct = 0
        for i in tqdm(anomaly_loader):
            inp = i[0].squeeze(0).float().to(device)
            # print(inp.shape)
            cf_label = i[1].squeeze(0).float().to(device)
            # print(label.shape)
            # cf_label = i[3].squeeze(0).to(device)
            if i[2][0] != H_[0].shape[1]:
                H_[0] = H_[0][:, H_[0].shape[1] - i[2][0]:H_[0].shape[1], :].contiguous()
                H_[1] = H_[1][:, H_[1].shape[1] - i[2][0]:H_[1].shape[1], :].contiguous()
            cf_out, new_H = model(inp,adj, H_)
            # print(cf_out.shape, new_H[0].shape)
            H_ = (new_H[0].detach(), new_H[1].detach())
            #     # print(cf_out.shape, i[2].squeeze(0).shape)
            cf_loss = cf_criteon(cf_out, cf_label)
            lossvalue = cf_loss  # +rec_loss
            # print(type(cf_loss),type(rec_loss))
            abnorm_loss_list[ct] = lossvalue
            optimizer.zero_grad()
            lossvalue.backward()
            optimizer.step()
            ct = ct + 1
        print(np.sum(abnorm_loss_list) / abnorm_loss_list.shape[0])
        ab_loss_res.append(np.sum(abnorm_loss_list) / abnorm_loss_list.shape[0])
    loss_save_file.write(
        "epoch%s: abnormal_patch_loss %s \n" % (epochnum, np.sum(ab_loss_res) / len(ab_loss_res)))
    torch.save(model.state_dict(), "save/para_a_%s.pth" % config.dataset)
loss_save_file.close()
