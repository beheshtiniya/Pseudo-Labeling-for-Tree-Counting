import multiprocessing
from multiprocessing import Process
import threading
if __name__ == '__main__':
    import argparse
    import os
    import time
    import warnings
    import torch
    from networks import TreeCountNet
    import train
    import test
    # from networks import TEDNet, SaNet, MCNN,  Comnet, TreeCountNet
    torch.cuda.empty_cache()
    warnings.filterwarnings("ignore")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    torch.cuda.memory_summary(device=None, abbreviated=False)
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataname', type=str, default='TC_dataset')
    parser.add_argument('--model', type=str, default='TreeCountNet', help='')
    parser.add_argument('--batch_size', type=int, default=5, help='batch_size')
    parser.add_argument('--gauss_kernel', type=int, default=15)
    parser.add_argument('--sigma', type=float, default=9)
    parser.add_argument('--factor', type=float, default=5, help='3-16 or 5-256')
    parser.add_argument('--num_epochs', type=int,
                        default=80 , help='number of epochs')
    parser.add_argument('--loss_function', type=str, default='SSIM_L2')
    parser.add_argument('--ssim_weight', type=float, default=0.02)
    parser.add_argument('--adaption_gaussian', type=bool, default=False)
    parser.add_argument('--deepsupervision', type=bool, default=True)
    parser.add_argument('--test_model', type=str, default='-')
    parser.add_argument('--trainable', type=str, default='train_')
    arg = parser.parse_args()

    # if arg.model == 'SaNet':
    #     model = SaNet.SANEet()
    # if arg.model == 'MCNN':
    #     model = MCNN.MCNN()
    # if arg.model == 'Crowdnet':
    #     model = Comnet.com_net()
    if arg.model == 'TreeCountNet':
        model = TreeCountNet.TreeCountNet(arg)
    # if arg.model == 'TEDNet':
    #     model = TEDNet.TEDNet(arg)

    if arg.dataname == 'TC_dataset':
        dataset_path = 'E:\PSSNET2\PSSNET2\TreeCountNet\TreeCounting\datasets\TC'

    if arg.trainable == 'train_':
        time1 = time.strftime('%m-%d-%H-%S', time.localtime(time.time()))
        path = './checkpoint/' + str(time1) + '_' + str(arg.dataname) + '_' + str(arg.model) + '_' + str(
            arg.gauss_kernel) + '_' + str(
            arg.sigma) + '_' + str(arg.factor) + '_' + str(arg.loss_function) + '_' + str(
            arg.deepsupervision) + '_' + str(arg.ssim_weight)
        os.makedirs(path)
        checkpoint_path = path + '/'

        path1 = './results/' + str(time1) + '_' + str(arg.dataname) + '_' + str(arg.model) + '_' + str(
                arg.gauss_kernel) + '_' + str(arg.sigma) + '_' + str(arg.factor) + '_' + str(arg.loss_function) + '_' + str(
            arg.deepsupervision) + '_' + str(arg.ssim_weight)
        os.makedirs(path1)
        result_path = path1 + '/'

        full_path = result_path + 'record.txt'
        with open(full_path, 'a') as file:
            file.write('time' + str(time1) + str(arg.model) + " " + "\n")
            file.write('Model:' + arg.model + " " + "\n")
            file.write('gauss_kernel:' + str(arg.gauss_kernel) + " " + "\n")
            file.write('sigma:' + str(arg.sigma) + " " + "\n")
            file.write('factor:' + str(arg.factor) + " " + "\n")
            file.write('loss_function:' + arg.loss_function + " " + "\n")
            file.write('deepsupervision:' + str(arg.deepsupervision) + " " + "\n")
            file.write('SSIM_WEIGHT:' + str(arg.ssim_weight) + " " + "\n")
            # print("MODEL::",model)
            # print("device::",device)
            #print("datase_path",dataset_path)
            #print("chekpoint_path",checkpoint_path)
            # print("resut_path",result_path)
            # print("arg",arg)
        train.train(model, device, dataset_path, checkpoint_path, result_path, arg)
        print("chekpoint_path", checkpoint_path)


        test.test(model, device, dataset_path, checkpoint_path, result_path, arg)
