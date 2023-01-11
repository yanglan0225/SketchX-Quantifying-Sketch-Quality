import config as config_vector
import dataloader
import models
import argparse
import torch.optim as optim
import torch
from torch.utils.tensorboard import SummaryWriter
import os
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
#torch.backends.cudnn.enabled = False
def set_optimizer(config, params):
    """
    setting optimizer
    :param config:
    :return:
    """

    optimizer = optim.Adam(params, lr=config.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.max_epoch * config.max_train_batch, eta_min=1e-8)

    return optimizer, scheduler

def train(model, loader, val_loader, optimizer, scheduler, config, loss_func, metric, lamada_g):
    model.train()
    metric.train()
    max_batch = config.max_train_batch

    savedir = os.path.join(config.save_dir, config.file_name)
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    logdir = os.path.join(config.log_dir, config.file_name)
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    log_writer = SummaryWriter(log_dir=logdir)

    best_acc = 0.0 # to find out the best performance model

    for epoch in range(config.max_epoch):
        model.train()
        metric.train()

        train_loss = 0.0
        train_correct = 0.0
        total_num = 0

        since = time.time()

        for i, data in enumerate(loader):
            labels = data[0].long().to(config.device)
            src = data[1].to(config.device).float()
            lens = data[2].to(config.device)

            ori_feat = model(src, lens)
            edited_feat, loss_g, x_norm = metric(ori_feat, labels)

            batch_loss = loss_func(edited_feat, labels)

            total_loss = batch_loss + lamada_g * loss_g

            batch_preds = torch.max(edited_feat, 1)[1]

            batch_correct = (batch_preds==labels).sum()

            train_loss += total_loss.item() * len(edited_feat)

            train_correct += batch_correct.item()

            total_num += len(edited_feat)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            scheduler.step()

            log_writer.add_scalar('CLS_Loss/Train', batch_loss.item(), i + epoch * max_batch)
            log_writer.add_scalar('Acc/Train', train_correct/total_num, i + epoch * max_batch)
            log_writer.add_scalar('LR/lstm', scheduler.get_lr()[0], i + epoch * max_batch)
            log_writer.add_scalar('LR/metric', scheduler.get_lr()[1], i + epoch * max_batch)

            if i > 0 and i % 10 == 0:
                print("Epoch {:.0f}/{:.0f} || batch: {:.0f}/{:.0f}, average_loss: {:.6f}, loss_g: {:.6f} || train_acc: {:.4f}"
                .format(epoch, config.max_epoch, i+1, max_batch, total_loss.item(), loss_g.item(), train_correct/total_num))

            if i > 0 and i % int(0.01 * max_batch) == 0:
                state = {'model': model.state_dict(), 'metric': metric.state_dict(), 'optimizer': optimizer.state_dict()}
                torch.save(state, os.path.join(savedir, 'temp.pkl'))

        print('Epoch {:.0f} training cost time: {:.0f} m, {:.0f} s'.format(epoch, (time.time() - since) // 60,
                                                                           (time.time() - since) % 60))
        torch.save(state, os.path.join(savedir, str(epoch) + '.pkl'))


        since = time.time()
        val_loss, val_acc = valid(model, val_loader, epoch, config, loss_func, metric, lamada_g)
        log_writer.add_scalar('Acc/Valid', val_acc, epoch)
        print('Epoch {:.0f} validing cost time: {:.0f} m, {:.0f} s'.format(epoch, (time.time() - since) // 60,
                                                                           (time.time() - since) % 60))

        if val_acc > best_acc:
            print('best acc updated, Epoch {:.0f}'.format(epoch))
            best_acc = val_acc
            best_model_params = model.state_dict()
            best_metric_params = metric.state_dict()
            state = {'model': best_model_params, 'metric': best_metric_params, 'optimizer':optimizer.state_dict()}
            torch.save(state, os.path.join(savedir, 'best_model.pkl'))
    # save model

def valid(model, loader, epoch, config, loss_func, metric, lambda_g):
    model.eval()
    metric.eval()

    val_loss = 0.0
    val_correct = 0.0
    val_total = 0

    with torch.no_grad():

        for i, data in enumerate(loader):
            labels = data[0].long().to(config.device)
            src = data[1].to(config.device).float()
            lens = data[2].to(config.device)

            ori_feat = model(src, lens)

            edited_feat, loss_g, x_norm, _ = metric(ori_feat, labels)

            batch_loss = loss_func(edited_feat, labels)

            total_loss = batch_loss

            batch_preds = torch.max(edited_feat, 1)[1]

            batch_correct = (batch_preds == labels).sum()

            val_loss += total_loss.item() * len(edited_feat)

            val_correct += batch_correct.item()

            val_total += len(edited_feat)

        print("{:.0f} Epoch || valid loss: {:.6f} || valid acc: {:.4f}".format(
            epoch, val_loss/val_total, val_correct/val_total
        ))

        return val_loss, val_correct/val_total

def config_print(config):
    print('data_dir: ', config.data_dir)
    print('max length: ', config.max_length)
    print('batch size: ', config.batch_size)
    print('max epoch: ', config.max_epoch)
    print('device: ', config.device)
    print('lr: ', config.lr)
    print('l_a', config.l_a)
    print('u_a', config.u_a)
    print('l_m', config.l_m)
    print('u_m', config.u_m)
    print('scale', config.scale)

def get_parameter_number(nets):
    total_num = 0
    trainable_num = 0
    for net in nets:
        total_num += sum(p.numel() for p in net.parameters())
        trainable_num += sum(p.numel() for p in net.parameters() if p.requires_grad)
    print('Total:', total_num, 'Trainable:', trainable_num)

def main():

    config = config_vector.Config()

    train_loader, max_train, val_loader, max_valid = dataloader.get_loader(config)

    config.max_train_batch = max_train
    config.max_valid_batch = max_valid

    parser = argparse.ArgumentParser()
    parser.add_argument('--file_name', type=str,
                        help='the different training setting')
    args = parser.parse_args()

    config_print(config)

    config.file_name = args.file_name

    ### init model
    metric = models.GACL(config.d_feed, config.num_classes, config)

    model = models.LSTM(config)

    get_parameter_number([model, metric])

    optimizer, scheduler = set_optimizer(config, [{'params': model.parameters(), 'lr': config.lr},
                                                  {'params': metric.parameters(), 'lr': config.lr}])

    if config.pretrain_file is not None:
        pretrain_dict = torch.load(config.pretrain_file, map_location='cpu')
        model.load_state_dict(pretrain_dict['model'])
        metric.load_state_dict(pretrain_dict['metric'])

    model.to(config.device)
    loss_func = torch.nn.CrossEntropyLoss()
    metric.to(config.device)

    lamada_g = 150

    train(model, train_loader, val_loader, optimizer, scheduler, config, loss_func, metric, lamada_g)

if __name__=="__main__":
    main()
