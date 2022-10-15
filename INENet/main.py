import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from torch.nn import DataParallel
from tensorboardX import SummaryWriter
import transforms3d
from tqdm import tqdm
import sys, os
sys.path.append("..")
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from data_utils import ModelNet40_Reg
from model import OverlapNet
# from ours_model import OurDCP
from scipy.spatial.transform import Rotation
from evaluate_funcs import evaluate_mask


use_cuda = torch.cuda.is_available()
torch.cuda.set_device(0)
if not os.path.isdir("./logs"):
    os.mkdir("./logs")
writer = SummaryWriter('./logs')
batchsize = 32
epochs = 300
lr = 1e-3
num_subsampled_rate = 0.75
unseen = False
noise = False
file_type = 'modelnet40'
# file_type = 'S3DIS'


# set seed
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# 设置随机数种子
setup_seed(1234)


def test_one_epoch(net, test_loader, return_t=False):
    net.eval()
    total_loss = 0
    # loss_fn = nn.BCEWithLogitsLoss()
    # loss_fn = nn.L1Loss()
    loss_fn = nn.HuberLoss(delta=0.4)
    # loss_fn = nn.MSELoss()

    with torch.no_grad():
        count = 0
        accs = []
        preciss = []
        recalls = []
        f1s = []
        for src, target, rotation, translation, euler, mask_gt in tqdm(test_loader):

            mask_gt_cp = torch.where(mask_gt == 0.0, -1.0, 1.0)
            if use_cuda:
                src = src.cuda()
                target = target.cuda()

            mask, mask_idx, threshold = net(src, target)
            loss = loss_fn(mask - threshold, mask_gt_cp.cuda())

            total_loss += loss.item()
            # 计算准确率
            count += 1
            # 评估
            acc, precis, recall, f1 = evaluate_mask(mask_idx, mask_gt)
            accs.append(acc)
            preciss.append(precis)
            recalls.append(recall)
            f1s.append(f1)
        acc = np.mean(accs)
        precis = np.mean(preciss)
        recall = np.mean(recalls)
        f1 = np.mean(f1s)
    if return_t:
        return total_loss, acc, precis, recall, f1, torch.mean(threshold).item()

    return total_loss, acc, precis, recall, f1


def train_one_epoch(net, opt, train_loader, return_t=False):
    net.train()
    total_loss = 0
    count = 0
    accs = []
    preciss = []
    recalls = []
    f1s = []
    # loss_fn = nn.BCEWithLogitsLoss()
    # loss_fn = nn.L1Loss()
    loss_fn = nn.HuberLoss(delta=0.4)
    # loss_fn = nn.MSELoss()

    for src, target, rotation, translation, euler, mask_gt in tqdm(train_loader):
        # print(src.shape, target.shape)
        # 用于计算损失的阈值，mask_gt用于计算准确率
        mask_gt_cp = torch.where(mask_gt==0.0, -1.0, 1.0)
        if use_cuda:
            src = src.cuda()
            target = target.cuda()

        mask, mask_idx, threshold = net(src, target)

        opt.zero_grad()
        loss = loss_fn(mask - threshold, mask_gt_cp.cuda())

        total_loss += loss.item()

        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 5, norm_type=2)
        opt.step()

        # 计算准确率
        count += 1
        # 评估
        acc, precis, recall, f1 = evaluate_mask(mask_idx, mask_gt)
        accs.append(acc)
        preciss.append(precis)
        recalls.append(recall)
        f1s.append(f1)
    acc = np.mean(accs)
    precis = np.mean(preciss)
    recall = np.mean(recalls)
    f1 = np.mean(f1s)

    if return_t:
        return total_loss, acc, precis, recall, f1, torch.mean(threshold).item()

    return total_loss, acc, precis, recall, f1


if __name__ == '__main__':

    best_loss = np.inf
    best_acc = 0

    train_loader = DataLoader(
        dataset=ModelNet40_Reg(partition='train', max_angle=45, max_t=0.5, unseen=unseen, file_type=file_type,
                               num_subsampled_rate=num_subsampled_rate, partial_source=True, noise=noise),
        batch_size=batchsize,
        shuffle=True,
        num_workers=4
    )
    test_loader = DataLoader(
        dataset=ModelNet40_Reg(partition='test', max_angle=45, max_t=0.5,  unseen=unseen, file_type=file_type,
                               num_subsampled_rate=num_subsampled_rate, partial_source=True, noise=noise),
        batch_size=batchsize,
        shuffle=False,
        num_workers=4
    )

    if file_type == 'modelnet40':
        num_subsampled_points = int(num_subsampled_rate * 1024)
    elif file_type == 'S3DIS':
        num_subsampled_points = int(num_subsampled_rate * 2048)
    elif file_type in ['3DMatch', 'Apollo', 'bunny']:
        num_subsampled_points = int(num_subsampled_rate * 4096)

    net = OverlapNet(num_subsampled_points=num_subsampled_points)
    opt = optim.AdamW(params=net.parameters(), lr=lr)
    # 动态调整学习率
    # scheduler = MultiStepLR(opt, milestones=[50, 100, 150], gamma=0.1)

    if use_cuda:
        net = net.cuda()
        # net = DataParallel(net, device_ids=[0, 1])

    start_epoch = -1
    RESUME = False  # 是否加载模型继续上次训练
    if RESUME:
        path_checkpoint = "./checkpoint/ckpt%s.pth"%(str(file_type)+str(num_subsampled_points))  # 断点路径

        checkpoint = torch.load(path_checkpoint)  # 加载断点
        net.load_state_dict(checkpoint['net'])  # 加载模型可学习参数
        # scheduler.load_state_dict(checkpoint["lr_step"])
        opt.load_state_dict(checkpoint['optimizer'])  # 加载优化器参数
        start_epoch = checkpoint['epoch']  # 设置开始的epoch
        # 加载上次best结果
        best_loss = checkpoint['best_loss']
        best_acc = checkpoint['best_acc']
        best_recall = checkpoint['best_recall']
        best_precis = checkpoint['best_precis']
        best_f1 = checkpoint['best_f1']

    for epoch in range(start_epoch + 1, epochs):

        train_loss, train_acc, train_precis, train_recall, train_f1, train_threshold = train_one_epoch(net, opt, train_loader, return_t=True)

        test_loss, test_acc, test_precis, test_recall, test_f1, test_threshold = test_one_epoch(net, test_loader, return_t=True)

        # scheduler.step()

        if test_acc >= best_acc:
            best_loss = test_loss
            best_acc = test_acc
            best_precis = test_precis
            best_recall = test_recall
            best_f1 = test_f1
            # 保存最好的checkpoint
            checkpoint_best = {
                "net": net.state_dict(),
            }
            if not os.path.isdir("./checkpoint"):
                os.mkdir("./checkpoint")
            torch.save(checkpoint_best, './checkpoint/ckpt_best%s.pth'%(str(file_type)+str(num_subsampled_points)))

        print('---------Epoch: %d---------' % (epoch+1))
        print('Train: Loss: %f, Acc: %f, Precis: %f, Recall: %f, F1: %f'
                      % (train_loss, train_acc, train_precis, train_recall, train_f1))

        print('Test: Loss: %f, Acc: %f, Precis: %f, Recall: %f, F1: %f'
                      % (test_loss, test_acc, test_precis, test_recall, test_f1))

        print('Best: Loss: %f, Acc: %f, Precis: %f, Recall: %f, F1: %f'
                      % (best_loss, best_acc, best_precis, best_recall, best_f1))
        writer.add_scalar('Train/train loss', train_loss, global_step=epoch)
        writer.add_scalar('Train/train Acc', train_acc, global_step=epoch)
        writer.add_scalar('Train/train Precis', train_precis, global_step=epoch)
        writer.add_scalar('Train/train Recall', train_recall, global_step=epoch)
        writer.add_scalar('Train/train F1', train_f1, global_step=epoch)
        writer.add_scalar('Train/train threshold', train_threshold, global_step=epoch)

        writer.add_scalar('Test/test loss', test_loss, global_step=epoch)
        writer.add_scalar('Test/test Acc', test_acc, global_step=epoch)
        writer.add_scalar('Test/test Precis', test_precis, global_step=epoch)
        writer.add_scalar('Test/test Recall', test_recall, global_step=epoch)
        writer.add_scalar('Test/test F1', test_f1, global_step=epoch)
        writer.add_scalar('Test/test threshold', test_threshold, global_step=epoch)

        writer.add_scalar('Best/best loss', best_loss, global_step=epoch)
        writer.add_scalar('Best/best Acc', best_acc, global_step=epoch)
        writer.add_scalar('Best/best Precis', best_precis, global_step=epoch)
        writer.add_scalar('Best/best Recall', best_recall, global_step=epoch)
        writer.add_scalar('Best/best F1', best_f1, global_step=epoch)

        # 保存checkpoint
        checkpoint = {
            "net": net.state_dict(),
            'optimizer': opt.state_dict(),
            "epoch": epoch,
            # "lr_step": scheduler.state_dict(),
            "best_loss":best_loss,
            'best_acc': best_acc,
            'best_recall':best_recall,
            'best_precis':best_precis,
            'best_f1':best_f1,
        }
        if not os.path.isdir("./checkpoint"):
            os.mkdir("./checkpoint")
        torch.save(checkpoint, './checkpoint/ckpt%s.pth'%(str(file_type)+str(num_subsampled_points)))
    writer.close()



#
