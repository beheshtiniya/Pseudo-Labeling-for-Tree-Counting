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
out_ssim =[]
T = []
def test(model,device,dataset_path,checkpointpath,out_path,arg):
	model_name = 'weights_best_val_loss.pth'
	#print("checkpointpath",checkpointpath)
	pt_name = checkpointpath + model_name
	#print("pt_name",pt_name)
	# if not os.path.exists(pt_name):
	# 	continue
	# else:
	# 	book_name_xls = out_path + '/' + model_name[:-4] + '_pre_result.xls'
	test_img = dataset_path + '\\test\\IMG'
	test_gt = dataset_path + '\\test\\GT'
	# if arg.model == 'MCNN':
	# 	model = MCNN.MCNN()
	# if arg.model == 'Crowdnet':
	# 	model = Comnet.com_net()
	if arg.model == 'TreeCountNet':
		model = TreeCountNet.TreeCountNet(arg)
	# if arg.model == 'TEDNet':
	# 	model = TEDNet.TEDNet(arg)

	model = model.to(device)
	model = nn.DataParallel(model)
	model.load_state_dict(torch.load(pt_name))
	model.eval()
	name_list = os.listdir(test_img)
	results = []
	coordinates = []
	name_l = []
	ppre_l =[]
	ggt_l = []
	gt_l = []
	pre_l = []
	mae_1 = []
	ssim_1 = []
	r2_1=[]
	mAPE_1=[]
	all_gt_count=[]
	all_predcount=[]
	sum_gt_count=0
	sum_pred_after_filter=0
	step_test = 0
	test_mae = 0
	mse_sum = 0
	rmse_sum = 0
	mape_sum = 0
	GT_SUM = 0
	mae_sum_f = 0

	for name in name_list:
		# print("name", name)
		pth = os.path.join(test_img, name)
		pth2 = os.path.join(test_gt, name)
		gt = cv2.imread(pth2, flags=-1)
		# cv2.imshow('gt', gt)

		gt_arr=np.array(gt)
		gt_count=np.sum(gt_arr==255)
		# print('img:',pth)
		# print('gt_count:,',gt_count)
		sum_gt_count +=gt_count
		all_gt_count.append(gt_count)
		# print('arg.factor:',arg.factor)
		gt = gt * int(arg.factor)

		img = cv2.imread(pth, flags=-1)
		# cv2.imshow('Image', img)
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
			# print('batch:',pre1.shape[0])
			out1 = pre1.cpu().numpy()
			out1 = np.squeeze(out1)
			outname =out_path + name

			outname = out_path + name.replace('.tif', '.jpg')  # تغییر پسوند به .jpg

			# تبدیل داده‌های تصویر به محدوده 0-255 و نوع uint8 (ضروری برای JPG)
			if out1.dtype != np.uint8:
				out1 = cv2.normalize(out1, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

			# ذخیره به صورت JPG (با کیفیت 90%)
			cv2.imwrite(outname, out1, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

			# خواندن تصویر JPG
			image = cv2.imread(outname, cv2.IMREAD_UNCHANGED)

			# cv2.imwrite(outname, out1)
			# image = cv2.imread(outname, cv2.IMREAD_UNCHANGED)
			# Remove small components first
			if image.dtype != np.uint8:
				image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
				num_labels, labels_im = cv2.connectedComponents(image)
				for label in range(1, num_labels):
					component = (labels_im == label)
					if np.sum(component) < 3:
						image[component] = 0
			cv2.imshow('predicted dotmap', image)

			_, binary_image = cv2.threshold(image, 30, 255, cv2.THRESH_BINARY)
			# Define the structuring element for connectivity
			binary_image = median_filter(binary_image, size=3)
			kernel = np.ones((13, 13), np.uint8)

			# Perform morphological closing to connect nearby white pixels
			connected_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)

			# _, binary_image = cv2.threshold(image, 0.1, 0.8, cv2.THRESH_BINARY)
			num_labels, labels_im = cv2.connectedComponents(connected_image)
			# Remove small components
			for label in range(1, num_labels):
				component = (labels_im == label)
				if np.sum(component) < 19:
					connected_image[component] = 0


			output_path = os.path.join('E:\\PSSNET2\\PSSNET2\\TreeCountNet\\results\\report\\filtered', name)
			# os.makedirs(output_path, exist_ok=True)
			cv2.imwrite(output_path, connected_image)
			# cv2.imshow('filtered dotmap', connected_image)
			num_labels_filtered,labels_filtered = cv2.connectedComponents(connected_image)
			predict = num_labels_filtered - 1
			results.append([name, predict, gt_count])

			all_predcount.append(predict)
			sum_pred_after_filter += (num_labels_filtered - 1)
			# Save coordinates in a separate CSV file for each image
			coords = []
			for label in range(1, num_labels_filtered):
				component = (labels_filtered == label)
				y, x = np.mean(np.where(component), axis=1)
				coords.append([int(x), int(y)])
				xlsx_filename = os.path.splitext(name)[0] + '.xlsx'
				coord_df = pd.DataFrame(coords, columns=['cx', 'cy'])
		# Save the DataFrame to an Excel file
				coord_df.to_excel(os.path.join(out_path, xlsx_filename), index=False)			# print(f"Number of connected components after filtering: {predict}")

			# Visualize the binary image
			# cv2.imshow('Binary Image', binary_image)
			# cv2.waitKey(0)
			# cv2.destroyAllWindows()
	# Save results to an Excel file
	df = pd.DataFrame(results, columns=['image_name', 'predict', 'gt'])
	df.to_excel(os.path.join(out_path, '_pre_result.xlsx'), index=False)
	coord_df = pd.DataFrame(coordinates, columns=['image_name', 'coordinate'])
	coord_df.to_csv(os.path.join(out_path, '_coordinates.csv'), index=False)




	# print('all_gt:',all_gt_count)
	# print('all_prediction:',all_predcount)
	# Convert lists to NumPy arrays
	all_gt_count = np.array(all_gt_count)
	all_predcount = np.array(all_predcount)
	slope, intercept = np.polyfit(all_gt_count, all_predcount, 1)

	# Calculate MAE
	mae = np.mean(np.abs(all_gt_count - all_predcount))
	print(f"Mean Absolute Error: {mae}")

	# Calculate MAPE
	mape = np.mean(np.abs((all_gt_count - all_predcount) / all_gt_count)) * 100
	print(f"Mean Absolute Percentage Error: {mape}%")

	# Calculate sums
	sum1 = np.sum(all_gt_count)
	sum2 = np.sum(all_predcount)
	rezult=sum2-sum1
	print(f"Sum of all_gt_count: {sum1}, Sum of all_predcount: {sum2}")
	k = (1, 30)
	l = (1, 30)
	fig, ax = plt.subplots()
	plt.scatter(all_gt_count, all_predcount, label="samples")
	ax.plot(k, l, color="red", label="ideal");

	regression_line = slope * all_gt_count + intercept
	ax.plot(all_gt_count, regression_line, color="blue",
			label=f"regression (slope={slope:.2f}, intercept={intercept:.2f})")
	# for xy in zip(x, y):
	# plt.annotate('(%.2f, %.2f)' % xy, xy=xy)# write x,y valu
	ax.legend(fontsize=8)
	ax.set_xlabel("predicted Count")
	ax.set_ylabel("Ground Truth Count")
	fig.suptitle("Performance")
	plt.show()

