from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from pointnet.dataset_TXT import ShapeNetDataset, ModelNetDataset
# from pointnet.model import PointNetCls, feature_transform_regularizer
from pointnet.model_VFE_TXT import PointNetCls, feature_transform_regularizer,VoxelFeatureEncoding
import torch.nn.functional as F
from tqdm import tqdm
from tensorboardX import SummaryWriter
writer = SummaryWriter('logs')

parser = argparse.ArgumentParser()
parser.add_argument(
    '--batchSize', type=int, default=16, help='input batch size 32')
parser.add_argument(
    '--num_points', type=int, default=1024, help='input batch size 2500')
# parser.add_argument(
#     '--workers', type=int, help='number of data loading workers', default=1)
parser.add_argument(
    '--nepoch', type=int, default=3, help='number of epochs to train for 250')
parser.add_argument('--outf', type=str, default='cls_vfe_txt', help='output folder')
# parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--dataset', type=str, default='D:\wangke\shenlan\Point Cloud Processing\Charter5\pointnet.pytorch-master\modelnet40_normal_resampled_txt', help="dataset path")#required=True
parser.add_argument('--dataset_type', type=str, default='modelnet40', help="dataset type shapenet|modelnet40")
parser.add_argument('--feature_transform', action='store_true', help="use feature transform")

opt = parser.parse_args()
print(opt)

blue = lambda x: '\033[94m' + x + '\033[0m'

opt.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

if opt.dataset_type == 'shapenet':
    dataset = ShapeNetDataset(
        root=opt.dataset,
        classification=True,
        npoints=opt.num_points)

    test_dataset = ShapeNetDataset(
        root=opt.dataset,
        classification=True,
        split='test',
        npoints=opt.num_points,
        data_augmentation=False)
elif opt.dataset_type == 'modelnet40':
    dataset = ModelNetDataset(
        root=opt.dataset,
        npoints=opt.num_points,
        split='modelnet40_train')# split='trainval')

    test_dataset = ModelNetDataset(
        root=opt.dataset,
        split='modelnet40_test',
        npoints=opt.num_points,
        data_augmentation=False)
else:
    exit('wrong dataset type')


dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=opt.batchSize,
    shuffle=True)#,
    # num_workers=int(opt.workers))

testdataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=opt.batchSize,
        shuffle=True)#,
        # num_workers=int(opt.workers))

print(len(dataset), len(test_dataset))
num_classes = len(dataset.classes)
print('classes', num_classes)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

classifier = VoxelFeatureEncoding(k=num_classes)#, feature_transform=opt.feature_transform)#实例化模型
# classifier = PointNetCls(k=num_classes, feature_transform=opt.feature_transform)#实例化模型

# if opt.model != '':
#     classifier.load_state_dict(torch.load(opt.model))


optimizer = optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9, 0.999))
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
classifier.cuda()

num_batch = len(dataset) / opt.batchSize
'''断点续训'''
path = "./cls_vfe_txt/cls_model_199.pth"#断点路径
checkpoint = torch.load(path)#加载断点
classifier.load_state_dict(checkpoint['model'])
optimizer.load_state_dict(checkpoint['optimizer'])
start_epoch = checkpoint['epoch']+1
# start_epoch=0

for epoch in range(opt.nepoch):
    scheduler.step()
    for i, data in enumerate(dataloader, 0):
        points, target = data
        target = target[:, 0]
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()
        optimizer.zero_grad()
        classifier = classifier.train()
        pred = classifier(points)
        # pred, trans, trans_feat = classifier(points)
        loss = F.nll_loss(pred, target)
        # if opt.feature_transform:
        #     loss += feature_transform_regularizer(trans_feat) * 0.001
        loss.backward()
        optimizer.step()
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.data).cpu().sum()
        print('[%d: %d/%d] train loss: %f accuracy: %f' % (epoch+start_epoch, i, num_batch, loss.item(), correct.item() / float(opt.batchSize)))
        writer.add_scalar('Train/Loss', loss.item(), epoch + start_epoch)
        writer.add_scalar('Train/Acc', correct.item() / float(opt.batchSize), epoch + start_epoch)
        if i % 100 == 0:
            j, data = next(enumerate(testdataloader, 0))
            points, target = data
            target = target[:, 0]
            points = points.transpose(2, 1)
            points, target = points.cuda(), target.cuda()
            classifier = classifier.eval()
            pred = classifier(points)
            # pred, _, _ = classifier(points)
            loss = F.nll_loss(pred, target)
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.data).cpu().sum()
            print('[%d: %d/%d] %s loss: %f accuracy: %f' % (epoch+start_epoch, i, num_batch, blue('test'), loss.item(), correct.item()/float(opt.batchSize)))
            writer.add_scalar('Test/Loss', loss.item(), epoch + start_epoch)
            writer.add_scalar('Test/Acc', correct.item() / float(opt.batchSize), epoch + start_epoch)
    # torch.save(classifier.state_dict(), '%s/cls_model_%d.pth' % (opt.outf, epoch))
    state = { 'model': classifier.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch': epoch+start_epoch }
    torch.save(state, '%s/cls_model_%d.pth' % (opt.outf, epoch+start_epoch))





total_correct = 0
total_testset = 0
for i,data in tqdm(enumerate(testdataloader, 0)):
    points, target = data

    target = target[:, 0]
    points = points.transpose(2, 1)
    points, target = points.cuda(), target.cuda()
    classifier = classifier.eval()
    pred = classifier(points)
    # pred, _, _ = classifier(points)
    pred_choice = pred.data.max(1)[1]
    correct = pred_choice.eq(target.data).cpu().sum()
    total_correct += correct.item()
    total_testset += points.size()[0]

print("final accuracy {}".format(total_correct / float(total_testset)))