import os
import sys
import time
import argparse
import logging
import torch
import numpy as np
from matplotlib import pyplot as plt
import CTNetModules
import CTNet_util as util
import Function
import CustomDataset as CD
import LoadDataset as LD
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
logger = logging.getLogger(__name__)
torch.autograd.set_detect_anomaly(True)

def train_CTNet(args, tfa_module, tfa_optimizer, tfa_scheduler, train_dataloader, val_dataloader, epoch, tb_writer, tfr_loss):
    epoch_start_time = time.time()
    tfa_module.train()
    train_tfr_loss = 0
    for batch in train_dataloader:
        inputs = batch["GLCT data"]  # Obtain input the GLCT data
        # print('GLCT data', inputs.shape)
        # train = inputs[:, 1:, 1:]
        # print('label data', train.shape)
        output_tfr = tfa_module(inputs)
        # print('---------output_fr',output_fr.shape)
        labels = batch["label data"]  # Obtain input label data
        # print('label', labels.shape)
        # targets_pre = labels[:, 1:, 1:]
        # print('targets_pre', targets_pre.shape)
        targets = labels.cuda()
        # print('targets',targets.shape)

        loss_train = tfr_loss(output_tfr, targets)
        # loss_train = torch.tensor(loss_train)
        tfa_optimizer.zero_grad()
        with torch.autograd.detect_anomaly():
            loss_train.backward()
        tfa_optimizer.step()
        train_tfr_loss += loss_train.data.item()

    tfa_module.eval()
    val_tfr_loss, val_tfr_nl1d= 0, 0
    for batch in val_dataloader:
        inputs = batch["GLCT data"]
        # print('data', inputs.shape)
        # train = inputs[:, 1:, 1:]
        # print('train', train.shape)
        with torch.no_grad():
            output_tfr = tfa_module(inputs)
        # print('output_fr',output_fr.shape)
        labels = batch["label data"]
        # print('label', labels.shape)
        # targets_pre = labels[:, 1:, 1:]
        # print('targets', targets_pre.shape)
        targets = labels.cuda()
        # print('output_tfr.shape',output_tfr.shape)
        # print('targets.shape',targets.shape)
        # if epoch == 180:
        #     if batch == 1:
        #         # output_tfr = output_tfr.to(torch.float32)
        #         # x_abs_cpu = x_abs[i, :, :].detach().cpu().numpy()
        #         x_abs = torch.abs(output_fr)  # .squeeze(-3)
        #         for i in range(x_abs.size(0)):
        #             plt.figure()
        #             plt.imshow(x_abs[i, :, :].detach().cpu().numpy(), cmap='jet', aspect='auto', origin='lower',
        #                        extent=[0, 4, 0,50])
        #             plt.colorbar()
        #             plt.title(f'Sub-matrix {i + 1}')
        #             plt.xlabel('Width')
        #             plt.ylabel('Height')
        #             plt.show()
        # nl1d_batch = tfr_nl1d(output_tfr, targets)
        # val_tfr_nl1d += nl1d_batch.data.item()
        # print("L1 distances for each sample in the batch:", nl1d_batch)
        # print("L1 distances for each sample in the batch:", val_tfr_nl1d)
        # nl1d_val = torch.sum(nl1d_batch).to(torch.float32)
        # nl1d_fr_val_fr += nl1d_val.data.item()
        loss_val = tfr_loss(output_tfr, targets)
        # loss_val = torch.tensor(loss_val)

        # print('loss_val', loss_val)
        val_tfr_loss += loss_val.data.item()

    train_tfr_loss /= args.n_training
    val_tfr_loss /= args.n_validation
    # val_tfr_nl1d /= args.n_validation

    tb_writer.add_scalar('train_tfr_loss', train_tfr_loss, epoch)
    tb_writer.add_scalar('val_tfr_loss', val_tfr_loss, epoch)
    # tb_writer.add_scalar('val_tfr_nl1d', val_tfr_nl1d, epoch)
    tfa_scheduler.step(val_tfr_loss)
    logger.info("Epochs: %d / %d, Time: %.1f, training loss %.2f, validation loss %.2f",
                epoch, args.n_epochs_tfa, time.time() - epoch_start_time, train_tfr_loss, val_tfr_loss)#， nl1d_validation loss %.3e",, nl1d_fr_val_fr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--n_training', type=int, default=1000)
    parser.add_argument('--n_validation', type=int, default=100)
    parser.add_argument('--random_state', type=int, default=42)
    parser.add_argument('--tfa_module_type', type=str, default='ctn')
    parser.add_argument('--tfr_size', type=int, default=400)
    parser.add_argument('--feature_dim', type=int, default=400)
    parser.add_argument('--in_kernel_size', type=int, default=1)
    parser.add_argument('--red_layers', type=int, default=20)
    parser.add_argument('--red_filters', type=int, default=16)
    parser.add_argument('--cbam_filters', type=int, default=16)
    parser.add_argument('--out_filters', type=int, default=16)
    parser.add_argument('--upsampling', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--n_epochs_tfa', type=int, default=200)
    parser.add_argument('--save_epoch_freq', type=int, default=20)
    parser.add_argument('--output_dir', type=str, default='CTNet/checkpoint')
    parser.add_argument('--no_cuda', action='store_true')


    args = parser.parse_args()

    if torch.cuda.is_available() and not args.no_cuda:
        args.use_cuda = True
    else:
        args.use_cuda = False

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    file_handler = logging.FileHandler(filename=os.path.join(args.output_dir, 'run.log'))
    stdout_handler = logging.StreamHandler(sys.stdout)
    handlers = [file_handler, stdout_handler]
    logging.basicConfig(
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO,
        format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
        handlers=handlers
    )
    tb_writer = SummaryWriter(args.output_dir)
    util.print_args(logger, args)

    tfa_module = CTNetModules.tfa_module(args)
    tfr_loss = Function.l2_distance
    tfr_nl1d = Function.compute_batch_nl1d
    tfa_optimizer, tfa_scheduler = util.optim(args, tfa_module, 'tfa_module')

    '''
    ''Training interrupted, restart
    # Load checkpoint
    # checkpoint = torch.load('./checkpoint/ctn/xxx.pth')  # First deserialize the model
    # fr_module.load_state_dict(checkpoint['model'])
    # fr_optimizer.load_state_dict(checkpoint['optimizer'])
    # start_epoch = checkpoint['epoch']
    ''
    '''
    start_epoch = 1
    # Load training data - the GLCT results
    data_all = LD.load_mat_to_torch('CTNet/Dataset/xxx.mat')
    # print('data_all.shape',data_all.shape)
    train_data = data_all
    # Load valuation data - the ideal results
    ideal_all = LD.load_mat_to_torch('CTNet/Dataset/xxx.mat')
    # print('ideal_all.shape',ideal_all.shape)
    train_label_data = ideal_all

    # Splitting datasets
    train_data, val_data, train_labels, val_labels = train_test_split(train_data, train_label_data,
                                                                      test_size=(args.n_validation / args.n_training ),
                                                                      random_state=args.random_state)
    # Creating instances of datasets.
    train_dataset = CD.CustomDataset(train_data, train_labels)
    val_dataset = CD.CustomDataset(val_data, val_labels)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    logger.info('Number of parameters in the CTNet model : %.4f M' % (util.parameter_calculation(tfa_module) / 1e6))

    def train_and_maybe_save(epoch, args, tfa_module, tfa_optimizer, tfa_scheduler, train_dataloader, val_dataloader,
                             tb_writer, tfr_loss):
        if epoch < args.n_epochs_tfa:
            train_CTNet(args=args, tfa_module=tfa_module, tfa_optimizer=tfa_optimizer, tfa_scheduler=tfa_scheduler,
                        train_dataloader=train_dataloader, val_dataloader=val_dataloader, epoch=epoch,
                        tb_writer=tb_writer, tfr_loss=tfr_loss)

        if epoch % args.save_epoch_freq == 0 or epoch == args.n_epochs_tfa:
            util.save_checkpoint(tfa_module, tfa_optimizer, tfa_scheduler, args, epoch, args.tfa_module_type)


    for epoch in range(start_epoch, args.n_epochs_tfa + 1):
        train_and_maybe_save(epoch, args, tfa_module, tfa_optimizer, tfa_scheduler, train_dataloader, val_dataloader,
                             tb_writer, tfr_loss)