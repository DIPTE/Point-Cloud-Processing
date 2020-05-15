from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F


class STN3d(nn.Module):
    def __init__(self):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)


    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32))).view(1,9).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x#输出3*3旋转矩阵


class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1,self.k*self.k).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x#输出64*64旋转矩阵


class PointNetfeat(nn.Module):
    def __init__(self, global_feat = True, feature_transform = False):
        super(PointNetfeat, self).__init__()
        # self.stn = STN3d()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):
        n_pts = x.size()[2]
        # trans = self.stn(x)
        # #[32, 3, 2500]
        # x = x.transpose(2, 1)#[32, 2500, 3]
        # x = torch.bmm(x, trans)#[32, 2500, 3]
        # x = x.transpose(2, 1)#[32, 3, 2500]
        x = F.relu(self.bn1(self.conv1(x)))#[32, 64, 2500]
        # pointfeat = x
        # if self.feature_transform:
        #     trans_feat = self.fstn(x)
        #     x = x.transpose(2,1)
        #     x = torch.bmm(x, trans_feat)
        #     x = x.transpose(2,1)
        # else:
        #     trans_feat = None
        pointfeat = x#[32, 64, 2500]
        #.view( -1,64)
        x = F.relu(self.bn2(self.conv2(x)))#[32, 128, 2500]
        x = self.bn3(self.conv3(x))#[32, 1024, 2500]
        x = torch.max(x, 2, keepdim=True)[0]#[32, 1024, 1]
        x = x.view(-1, 1024)#[32, 1024]
        if self.global_feat:
            return pointfeat,x#, trans, trans_feat#新增输出pointfeat  [32, 64, 2500]) torch.Size([32, 1024])
        else:
            #[32, 1024]
            x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)#[32, 1024, 2500]
            return pointfeat, torch.cat([x, pointfeat], 1)#, trans, trans_feat#新增输出pointfeat#[32, 64, 2500]) torch.Size([32, 1088, 2500])

class PointNetCls(nn.Module):
    def __init__(self, k=2, feature_transform=False):
        super(PointNetCls, self).__init__()
        self.feature_transform = feature_transform
        self.feat = PointNetfeat(global_feat=True, feature_transform=feature_transform)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        _,x, trans, trans_feat = self.feat(x)
        # print(x.size())[32, 1024]
        x = F.relu(self.bn1(self.fc1(x)))
        # print(x.size())torch.Size([32, 512])
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1), trans, trans_feat

class VoxelFeatureEncoding(nn.Module):
    def __init__(self, k = 2, feature_transform=False):
        super(VoxelFeatureEncoding, self).__init__()
        self.feature_transform = feature_transform
        self.feat = PointNetfeat(global_feat=True, feature_transform=feature_transform)
        self.k = k
        # self.stn = STN3d()
        self.conv1 = torch.nn.Conv1d(1088, 1024, 1)
        # self.conv2 = torch.nn.Conv1d(512, 256, 1)
        # self.conv3 = torch.nn.Conv1d(256, 128, 1)
        # self.conv4 = torch.nn.Conv1d(128, self.k, 1)
        self.bn1 = nn.BatchNorm1d(1024)
        # self.bn2 = nn.BatchNorm1d(256)
        # self.bn3 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU()
        # if self.feature_transform:
        #     self.fstn = STNkd(k=1024)

        self.fc1 = nn.Linear(1024, 512)#[32, 2500, 5][32, 1024]
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.3)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(256)



    def forward(self, x):
        pointfeat,x = self.feat(x)#[32, 64, 2500]  [32, 1024]
        # pointfeat, x, trans, trans_feat = self.feat(x)  # [32, 64, 2500]  [32, 1024]
        # print(pointfeat.size(),x.size(), trans.size(), trans_feat.size())
        # batchsize = x.size()[0]
        n_pts = pointfeat.size()[2]
        x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
        x= torch.cat([x, pointfeat], 1)
        # print(x.size())#torch.Size([32, 1088, 2500])

        # self.fstn1 = STNkd(k=1088)
        # trans = self.fstn1(x)
        # x = x.transpose(2, 1)
        # x = torch.bmm(x, trans)
        # x = x.transpose(2, 1)
        # print(x.size())[32, 1088, 2500]
        x = F.relu(self.bn1(self.conv1(x)))
        # print(x.size())#[32, 1024, 2500]

        # if self.feature_transform:
        #     # print('feature_transform start!!!')
        #     trans_feat = self.fstn(x)
        #     # print(x.size())
        #     x = x.transpose(2,1)
        #     # print(x.size())
        #     x = torch.bmm(x, trans_feat)
        #     # print(x.size())
        #     x = x.transpose(2,1)
        #     # print(x.size())
        #     # print( 'feature_transform done!!!')
        # else:
        #     trans_feat = None
        # print(x.size())[32, 1024, 2500]
        x = torch.max(x, 2, keepdim=True)[0]  # [32, 1024, 1]
        # print(x.size())[32, 1024, 1]
        x = x.view(-1, 1024)  # [32, 1024]
        # print(x.size())#[32, 1024]
        # x = F.relu(self.bn2(self.conv2(x)))
        # print(x.size())[32, 256, 2500]
        # x = F.relu(self.bn3(self.conv3(x)))
        # print(x.size())[32, 128, 2500]
        # x = self.conv4(x)
        # print(x.size())[32, 5, 2500]
        # x = x.transpose(2, 1).contiguous()
        # print(x.size())[32, 2500, 5]
        # x = F.log_softmax(x.view(-1, self.k), dim=-1)
        # print(x.size())[80000, 5]
        # x = x.view(batchsize, n_pts, self.k)
        # print(x.size())[32, 2500, 5]
        # return x, trans, trans_feat
        # print(x.size())[32, 1024]
        x = F.relu(self.bn2(self.fc1(x)))
        # print(x.size())[32, 512]
        x = F.relu(self.bn3(self.dropout(self.fc2(x))))
        # print(x.size())[32, 256]
        x = self.fc3(x)
        # print(x.size())[32, 5]
        return F.log_softmax(x, dim=1)#, trans, trans_feat


class PointNetDenseCls(nn.Module):
    def __init__(self, k = 2, feature_transform=False):
        super(PointNetDenseCls, self).__init__()
        self.k = k
        self.feature_transform=feature_transform
        self.feat = PointNetfeat(global_feat=False, feature_transform=feature_transform)
        self.conv1 = torch.nn.Conv1d(1088, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, self.k, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, x):
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = x.transpose(2,1).contiguous()
        x = F.log_softmax(x.view(-1,self.k), dim=-1)
        x = x.view(batchsize, n_pts, self.k)
        return x, trans, trans_feat

def feature_transform_regularizer(trans):
    d = trans.size()[1]
    batchsize = trans.size()[0]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2,1)) - I, dim=(1,2)))
    return loss
if __name__ == '__main__':
    sim_data = Variable(torch.rand(32,3,2500))
    # trans = STN3d()
    # out = trans(sim_data)
    # print('stn', out.size())
    # print('loss', feature_transform_regularizer(out))
    #
    # sim_data_64d = Variable(torch.rand(32, 64, 2500))
    # trans = STNkd(k=64)
    # out = trans(sim_data_64d)
    # print('stn64d', out.size())
    # print('loss', feature_transform_regularizer(out))

    # pointfeat = PointNetfeat(global_feat=True)
    # a,out, _, _ = pointfeat(sim_data)
    # print('global feat', a.size(),out.size())#a.detach().numpy()
    # #
    # pointfeat = PointNetfeat(global_feat=False)
    # a,out, _, _ = pointfeat(sim_data)
    # print('point feat', a.size(),out.size())#a.detach().numpy()

    # cls = VoxelFeatureEncoding(k=5)
    # out, _, _ = cls(sim_data)
    # print('class', out.size())
    #
    cls = VoxelFeatureEncoding(feature_transform=True,k=5)
    out = cls(sim_data)#, _, _
    print('class', out.size())

    # cls = PointNetCls(k = 5)
    # out, _, _ = cls(sim_data)
    # print('class', out.size())

    # seg = PointNetDenseCls(k = 3)
    # out, _, _ = seg(sim_data)
    # print('seg', out.size())