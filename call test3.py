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
    import test3
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
                        default=3, help='number of epochs')
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
        checkpoint_path ="E:\\PSSNET2\\PSSNET2\\TreeCountNet\\checkpoint\\old\\10-26-11-35_TC_dataset_TreeCountNet_5_1.5_1_SSIM_L2_True_0.02\\"

        result_path = "./results/report/"
        print("chekpoint_path", checkpoint_path)
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        test3.test(model, device, dataset_path, checkpoint_path, result_path, arg)
