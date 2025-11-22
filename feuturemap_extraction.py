import torch
import torch.nn as nn
import cv2
import os
import numpy as np
import argparse
import torchvision.transforms as transforms

from networks.TreeCountNet import TreeCountNet   #  ← درست


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------------------------------------
# 1) تنظیمات مسیرها
# ------------------------------------------------------------

STEP0_CKPT = r"E:\PSSNET2\PSSNET2\TreeCountNet\checkpoint\checkpoint_for _report\old\10-25-14-59_TC_dataset_TreeCountNet_5_1.5_1_SSIM_L2_True_0.02\weights_best_val_loss.pth"
STEP5_CKPT = r"E:\PSSNET2\PSSNET2\TreeCountNet\checkpoint\checkpoint_for _report\11-23-16-09_TC_dataset_TreeCountNet_15_9_5_SSIM_L2_True_0.02\weights_best_val_loss.pth"

TEST_IMAGE = r"E:\PSSNET2\PSSNET2\TreeCountNet\TreeCounting\datasets\TC\test\IMG\1.tif"
SAVE_DIR = r"E:\PSSNET2\PSSNET2\TreeCountNet\results\featuremaps"

os.makedirs(SAVE_DIR, exist_ok=True)

# ------------------------------------------------------------
# 2) پیش‌پردازش تصویر
# ------------------------------------------------------------

import torchvision.transforms as transforms

x_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

# ------------------------------------------------------------
# 3) تعریف لایه‌هایی که می‌خواهیم hook کنیم
# ------------------------------------------------------------

TARGET_LAYERS = [
    "conv0_0",
    "conv1_0",
    "conv2_0",
    "conv3_0",
    "conv4_0",
    "conv0_4",
]

# ------------------------------------------------------------
# 4) تابع استخراج فیچر مپ‌ها
# ------------------------------------------------------------

def extract_maps(ckpt_path, tag):

    print(f"\n============ Extracting Feature Maps: {tag} ============\n")

    # ---------------- Load model ----------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--deepsupervision', type=bool, default=True)
    args = parser.parse_args([])

    model = TreeCountNet(args)
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)
    model.eval()

    # ---- DataParallel fix ----
    net = model.module if isinstance(model, nn.DataParallel) else model

    # ---- ثبت فیچر مپ‌ها ----
    fmap_dict = {}

    def hook_fn(name):
        def hook(m, inp, out):
            fmap_dict[name] = out.detach().cpu().numpy()
        return hook

    modules = dict(net.named_modules())

    for layer_name in TARGET_LAYERS:
        if layer_name in modules:
            modules[layer_name].register_forward_hook(hook_fn(layer_name))
        else:
            print(f"[WARNING] layer not found: {layer_name}")

    # ---------------- Load Image ----------------
    img = cv2.imread(TEST_IMAGE)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor = x_transforms(img_rgb).unsqueeze(0).to(device)

    # ---------------- Forward Pass ----------------
    with torch.no_grad():
        model(tensor)

    # ---------------- Save feature maps ----------------
    out_dir = os.path.join(SAVE_DIR, tag)
    os.makedirs(out_dir, exist_ok=True)

    for layer_name, fmap in fmap_dict.items():
        fmap = np.squeeze(fmap)  # remove batch

        layer_folder = os.path.join(out_dir, layer_name)
        os.makedirs(layer_folder, exist_ok=True)

        print(f"Saving {layer_name}  →  {fmap.shape}")

        # ذخیره هر کانال فیچر مپ
        for i in range(fmap.shape[0]):
            fm = fmap[i]
            fm = cv2.normalize(fm, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            save_path = os.path.join(layer_folder, f"{layer_name}_ch{i}.jpg")
            cv2.imwrite(save_path, fm)

    print(f"\nDONE → saved in: {out_dir}\n")

# ------------------------------------------------------------
# 5) اجرا برای STEP0 و STEP5
# ------------------------------------------------------------

if __name__ == "__main__":
    extract_maps(STEP0_CKPT, "STEP0")
    extract_maps(STEP5_CKPT, "STEP5")
