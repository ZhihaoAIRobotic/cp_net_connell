import torch
from cpnet2 import CPNET
from Cornell import Cornell
import torch.optim as optim
import datetime
import os
import argparse
import logging
import tensorboardX
import cv2
#空的数据如何处理？为什么tiff中会存在空数据？因为crop的时候没有处理好数据
#参数读取函数[[argparse]]
#找区域平均评分最高的点  及其对应的antipoddal point
#batch size:每次训练的数据样本数
#len(dataloader) : 并不是每次训练的数据样本数，而是迭代器的个数。每次训练的数据的长度=总数据量/迭代器的个数。

def parse_args():
    parser = argparse.ArgumentParser(description='Train GG-CNN')
    # Network
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--batches_per_epoch', type=int, default=50)
    parser.add_argument('--epochs', type=int, default=15000)
    parser.add_argument('--val_batches', type=int, default=150)


    args = parser.parse_args()
    return args



# def iou(pred, target, n_classes = 37):
#     # n_classes ：the number of classes in your dataset
#     ious = []
#     pred = pred.view(-1)
#     target = target.view(-1)
#     pred_inds = pred == 1
#     target_inds = target == 1
#     intersection = (pred_inds[target_inds]).long().sum().data.cpu()[0]  # Cast to long to prevent overflows
#     union = pred_inds.long().sum().data.cpu()[0] + target_inds.long().sum().data.cpu()[0] - intersection
#     if union == 0:
#         ious.append(float('nan'))  # If there is no ground truth, do not include in evaluation
#     else:
#         ious.append(float(intersection) / float(max(union, 1)))
#     return np.array(ious)



def validate(net, device, val_data, batches_per_epoch):
    """
    Run validation. btach size:12  batches_per_epoch:250
    :param net: Network
    :param device: Torch device
    :param val_data: Validation Dataset
    :param batches_per_epoch: Number of batches to run
    :return: Successes, Failures and Losses
    """
    net.eval()
    results = {
        'loss': 0,
    }

    ld = len(val_data) #batch size明明为1，为什么会出现len为89的情况

    #print('112',ld)

    with torch.no_grad():
        batch_idx = 0
        while batch_idx < batches_per_epoch:
            data_num=0
            for x, y, didx, rot, zoom_factor in val_data:
                batch_idx += 1
                data_num=data_num+1
                #print(data_num)
                if batches_per_epoch is not None and batch_idx >= batches_per_epoch:
                    break
                xc = x.to(device)
                yc = y.to(device)
                lossd = net.compute_loss(xc, yc)
                loss = lossd['loss']
                results['loss'] += loss.item()/batches_per_epoch

    return results




def train(epoch, net, device, train_data, optimizer, batches_per_epoch, vis=False):
    """
    Run one training epoch batchsize:32 batches_per_epoch:
    :param epoch: Current epoch
    :param net: Network
    :param device: Torch device
    :param train_data: Training Dataset
    :param optimizer: Optimizer
    :param batches_per_epoch:  Data batches to train on
    :param vis:  Visualise training progress
    :return:  Average Losses for Epoch
    """
    results = {
        'loss': 0,
    }

    net.train()

    batch_idx = 0
    ld = len(train_data)  # batch size明明为1，为什么会出现len为89的情况

    #print('111', ld)
    # Use batches per epoch to make training on different sized datasets (cornell/jacquard) more equivalent.
    while batch_idx < batches_per_epoch:
        for x, y, a_, b_, c_ in train_data:
            batch_idx += 1
            if batch_idx >= batches_per_epoch:
                break

            xc = x.to(device)
            yc = y.to(device)
            # #test
            # print( a_, b_, c_)
            # for i in range(100000000000):
            #
            #     lossd = net.compute_loss(xc, yc)
            #     loss = lossd['loss']
            #     print('Epoch: {}, Batch: {}, Loss: {:0.4f}'.format(epoch, batch_idx, loss.item()))
            #     optimizer.zero_grad()
            #     loss.backward()
            #     optimizer.step()
            #     if i % 50 == 0:
            #         torch.save(net.state_dict(), 'model_test/model2.pkl')

            lossd = net.compute_loss(xc, yc)

            loss = lossd['loss']


            #if batch_idx % 10 == 0:
               # logging.info('Epoch: {}, Batch: {}, Loss: {:0.4f}'.format(epoch, batch_idx, loss.item()))
                #print('Epoch: {}, Batch: {}, Loss: {:0.4f}'.format(epoch, batch_idx, loss.item()))
                #torch.save(net.state_dict(), 'model_test/model3_lr00001_bs16_d1.pkl')
            results['loss'] += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    results['loss'] /= 16
    results['loss'] /= batch_idx #求batch的平均，但是每一个batch中又包含32个数据

    return results



def run():
    args = parse_args()

    Path_dataset = '../cornell_data_clean2'

    # 首先初始化一个tensorboardX
    writer = tensorboardX.SummaryWriter('log_lr0001_bs16_d1')  # 括号里为数据存放的地址

    # train dataset
    train_dataset = Cornell(Path_dataset, start=0.0, end=0.9)
    # val dataset
    val_dataset = Cornell(Path_dataset, start=0.9, end=1)

    train_data = torch.utils.data.DataLoader(
        train_dataset,
        batch_size= 16 ,#args.batch_size,
        shuffle=True,
        #num_workers=args.num_workers
    )
    val_data = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        #num_workers=args.num_workers
    )

    net = CPNET()

    #net.load_state_dict(torch.load('model_test/model3_lr0001_bs16_d1.pkl'))
    device = torch.device("cuda:0")
    net = net.to(device)

    optimizer = optim.Adam(net.parameters(),lr=0.001)
    # logging.info('Done')
    print('Done')

    for epoch in range(args.epochs):
        #epoch=epoch
        print(epoch)
        train_results = train(epoch, net, device, train_data, optimizer,args.batches_per_epoch )#
        test_results = validate(net, device, val_data, args.val_batches)#
        torch.save(net.state_dict(), 'model_test/model3_lr0001_bs16_d1.pkl')
        writer.add_scalars('Accu',{'v':test_results['loss'],'t':train_results['loss']},epoch)
        print(train_results)
        print(test_results)


# # test
# Path_dataset = '../cornell_data'
# # train dataset
# train_dataset = Cornell(Path_dataset, start=0.0, end=0.9)
# #for i in range(855):
# a,b,c,d,e=train_dataset.__getitem__(352)
# print(a)


if __name__ == '__main__':
    run()

