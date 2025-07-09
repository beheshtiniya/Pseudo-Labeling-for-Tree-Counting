import argparse
import os
import numpy as np
import cv2
import math
import pandas as pd

import torch
from utils.metics import RMSE, mAE, R2, mAPE
import torch.nn as nn
from torchvision.transforms import transforms
from scipy.ndimage import median_filter
from networks import TreeCountNet
from utils.gaussian import choose_map_method
from utils.loss_fun import SSIM
from utils.metics import sum1
import matplotlib
import argparse
import os
import time
import warnings
import torch
from networks import TreeCountNet
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

x_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

y_transforms = transforms.ToTensor()


def test(model, device, dataset_path, checkpointpath, out_path, arg):
    model_name = 'weights_best_val_loss.pth'
    pt_name = checkpointpath + model_name
    test_img = dataset_path
    if arg.model == 'TreeCountNet':
        model = TreeCountNet.TreeCountNet(arg)
    model = model.to(device)
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(pt_name))
    model.eval()
    name_list = os.listdir(test_img)
    results = []
    coordinates = []
    all_predcount = []
    sum_pred_after_filter = 0

    for name in name_list:
        pth = os.path.join(test_img, name)
        img = cv2.imread(pth, flags=-1)
        img = x_transforms(img)
        img = img.unsqueeze(0)
        with torch.no_grad():
            x = img.to(device)
            if arg.deepsupervision:
                _, _, _, pre1 = model(x)
            else:
                pre1 = model(x)
            out1 = pre1.cpu().numpy()
            out1 = np.squeeze(out1)
            out2 = cv2.normalize(out1, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            out2 = out1.astype(np.uint8)
            outname = os.path.join(out_path,name)  # Ensure 'out1' is saved to the specified directory
            cv2.imwrite(outname, out2)
            image = cv2.imread(outname, cv2.IMREAD_UNCHANGED)
            # Display the image using Matplotlib
            # plt.imshow(image, cmap='gray')
            # plt.title('Prediction')
            # plt.show()
            image=out2
            if out1.dtype != np.uint8:
                image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                num_labels, labels_im = cv2.connectedComponents(image)
                for label in range(1, num_labels):
                    component = (labels_im == label)
                    if np.sum(component) < 3:
                        image[component] = 0
            _, binary_image = cv2.threshold(image, 10, 255, cv2.THRESH_BINARY)
            binary_image = median_filter(binary_image, size=3)
            kernel = np.ones((13, 13), np.uint8)
            connected_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
            num_labels, labels_im = cv2.connectedComponents(image)
            for label in range(1, num_labels):
                component = (labels_im == label)
                if np.sum(component) < 19:
                    image[component] = 0
            output_path = os.path.join(r'E:\FASTRCNN\FASTRCNN\dataset\000000origin_data\new syntetic dataset\New_test_dot\filtered', name)
            cv2.imwrite(output_path, image)
            num_labels_filtered, labels_filtered = cv2.connectedComponents(image)
            predict = num_labels_filtered - 1
            results.append([name, predict])
            all_predcount.append(predict)
            sum_pred_after_filter += (num_labels_filtered - 1)
            coords = []
            for label in range(1, num_labels_filtered):
                component = (labels_filtered == label)
                y, x = np.mean(np.where(component), axis=1)
                coords.append([int(x), int(y)])
                xlsx_filename = os.path.splitext(name)[0] + '.xlsx'
                coord_df = pd.DataFrame(coords, columns=['cx', 'cy'])
                coord_df.to_excel(os.path.join(out_path, xlsx_filename), index=False)
    df = pd.DataFrame(results, columns=['image_name', 'predict'])
    df.to_excel(os.path.join(out_path, '_pre_result.xlsx'), index=False)
    coord_df = pd.DataFrame(coordinates, columns=['image_name', 'coordinate'])
    coord_df.to_csv(os.path.join(out_path, '_coordinates.csv'), index=False)

    all_predcount = np.array(all_predcount)
    sum2 = np.sum(all_predcount)
    # rezult = sum2 - sum1
    print(f"Sum of all_gt_count: {sum2}")


import multiprocessing
from multiprocessing import Process
import threading

if __name__ == '__main__':


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
    parser.add_argument('--num_epochs', type=int, default=3, help='number of epochs')
    parser.add_argument('--loss_function', type=str, default='SSIM_L2')
    parser.add_argument('--ssim_weight', type=float, default=0.02)
    parser.add_argument('--adaption_gaussian', type=bool, default=False)
    parser.add_argument('--deepsupervision', type=bool, default=True)
    parser.add_argument('--test_model', type=str, default='-')
    parser.add_argument('--trainable', type=str, default='train_')
    arg = parser.parse_args()

    if arg.model == 'TreeCountNet':
        model = TreeCountNet.TreeCountNet(arg)

    if arg.dataname == 'TC_dataset':
        dataset_path = r'E:\FASTRCNN\FASTRCNN\dataset\000000origin_data\new syntetic dataset\test_images'
        checkpoint_path = "./checkpoint/11-23-16-09_TC_dataset_TreeCountNet_15_9_5_SSIM_L2_True_0.02/"
        result_path = r"E:\FASTRCNN\FASTRCNN\dataset\000000origin_data\new syntetic dataset\New_test_dot"
        print("chekpoint_path", checkpoint_path)
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")

        test(model, device, dataset_path, checkpoint_path, result_path, arg)
