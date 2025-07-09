import argparse
import os
import numpy as np
import cv2
from time import *
import math
import pandas as pd

import torch
from utils.metics import RMSE, mAE, R2, mAPE
import torch.nn as nn
from torchvision.transforms import transforms
from scipy.ndimage import median_filter
from networks import TreeCountNet
# from utils.excel_tool import *
# from networks import TEDNet
from utils.gaussian import choose_map_method
from utils.loss_fun import SSIM

from utils.metics import sum1
import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
arg = parser.parse_args()

x_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

# mask只需要转换为tensor
y_transforms = transforms.ToTensor()
out_num = []
out_ssim = []
T = []
def test(model, device, dataset_path, checkpointpath, out_path, arg):
    model_name = 'weights_best_val_loss.pth'
    pt_name = checkpointpath + model_name

    test_img = 'E:\\PSSNET2\\PSSNET2\\TreeCountNet\\TreeCounting\\datasets\\TC\\test\\IMG'
    test_gt = 'E:\\PSSNET2\\PSSNET2\\TreeCountNet\\TreeCounting\\datasets\\TC\\test\\GT'

    if arg.model == 'TreeCountNet':
        model = TreeCountNet.TreeCountNet(arg)

    model = model.to(device)
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(pt_name))
    model.eval()
    best_size_spot = 8
    size_spot2 = 19  # Set size_spot2 to 19
    name_list = os.listdir(test_img)
    results = []
    name_l = []
    ppre_l = []
    ggt_l = []
    gt_l = []
    pre_l = []
    mae_1 = []
    ssim_1 = []
    r2_1 = []
    mAPE_1 = []
    all_gt_count = []
    all_predcount = []
    sum_gt_count = 0
    sum_pred_after_filter = 0
    step_test = 0
    test_mae = 0
    mse_sum = 0
    rmse_sum = 0
    mape_sum = 0
    GT_SUM = 0
    mae_sum_f = 0

    best_threshold = 0
    best_kernel = 0
    i = 0
    for threshold in range(10, 150, 10):
        for kernel_size in range(3, 75, 5):
            for size_spot in range(3, 24, 2):
                i += 1
                print("i:", i)
                sum_gt_count = 0
                sum_pred_after_filter = 0
                all_gt_count = []
                all_predcount = []
                for name in name_list:
                    pth = os.path.join(test_img, name)
                    pth2 = os.path.join(test_gt, name)
                    gt = cv2.imread(pth2, flags=-1)
                    gt_arr = np.array(gt)
                    gt_count = np.sum(gt_arr == 255)
                    sum_gt_count += gt_count
                    all_gt_count.append(gt_count)
                    gt = gt * int(arg.factor)
                    img = cv2.imread(pth, flags=-1)
                    gt = y_transforms(gt)
                    gt = torch.Tensor(gt)
                    gt = gt.unsqueeze(0)
                    img = x_transforms(img)
                    img = img.unsqueeze(0)
                    with torch.no_grad():
                        x = img.to(device)
                        y = gt.to(device)
                        if arg.deepsupervision:
                            _, _, _, pre1 = model(x)
                        else:
                            pre1 = model(x)
                        out1 = pre1.cpu().numpy()
                        out1 = np.squeeze(out1)
                        outname = out_path + model_name + name
                        cv2.imwrite(outname, out1)
                        image = cv2.imread(outname, cv2.IMREAD_UNCHANGED)
                        if image.dtype != np.uint8:
                            image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                            num_labels, labels_im = cv2.connectedComponents(image)
                            for label in range(1, num_labels):
                                component = (labels_im == label)
                                if np.sum(component) < size_spot:
                                    image[component] = 0
                        _, binary_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
                        binary_image = median_filter(binary_image, size=3)
                        kernel = np.ones((kernel_size, kernel_size), np.uint8)
                        connected_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
                        num_labels, labels_im = cv2.connectedComponents(connected_image)
                        for label in range(1, num_labels):
                            component = (labels_im == label)
                            if np.sum(component) < size_spot2:
                                connected_image[component] = 0
                        output_path = os.path.join(out_path, 'filtered', name)
                        cv2.imwrite(output_path, connected_image)
                        num_labels_filtered, _ = cv2.connectedComponents(connected_image)
                        all_predcount.append(num_labels_filtered - 1)
                        sum_pred_after_filter += (num_labels_filtered - 1)
                all_gt_count = np.array(all_gt_count)
                all_pred_count = np.array(all_predcount)
                mae = np.mean(np.abs(all_gt_count - all_predcount))
                print(f"Mean Absolute Error: {mae}")
                mape = np.mean(np.abs((all_gt_count - all_predcount) / all_gt_count)) * 100
                print(f"Mean Absolute Percentage Error: {mape}%")
                rezult = np.sum(all_pred_count) - np.sum(all_gt_count)
                print("-----------------------------------------------")
                print("result:", rezult)
                print("mape:", mape)
                print("kernel_size:", kernel_size)
                print("threshold:", threshold)
                if i == 1:
                    min_mape = mape
                    best_threshold = threshold
                    best_kernel = kernel_size
                    best_size_spot2 = size_spot2
                    best_size_spot = size_spot
                    print("best_kernel:", best_kernel)
                    print("best_threshold:", best_threshold)
                if abs(mape) < abs(min_mape):
                    print("better find********************")
                    min_mape = mape
                    best_threshold = threshold
                    best_kernel = kernel_size
                    best_size_spot = size_spot
                    best_size_spot2 = size_spot2
                print("best_kernel:", best_kernel)
                print("best_threshold:", best_threshold)
                print("best_size_spot:", best_size_spot)
                print("best_size_spot2:", best_size_spot2)
                print("min_mape:", min_mape)

    print(f'Best threshold: {best_threshold}')
    print(f'Best kernel size: {best_kernel}')
    print(f'Minimum Result: {min_mape}')
    print("best_size_spot:", best_size_spot)
    print("best_size_spot2:", best_size_spot2)
