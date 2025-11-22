
import os
import cv2
import pandas as pd

# مسیر فایل‌های ورودی
excel_path = r"E:\PSSNET2\PSSNET2\TreeCountNet\results\report"
img_path = r"E:\PSSNET2\PSSNET2\TreeCountNet\TreeCounting\datasets\TC\test\IMG"

# مسیر ذخیره خروجی
save_path = r"E:\PSSNET2\PSSNET2\TreeCountNet\results\report\img &dot"
os.makedirs(save_path, exist_ok=True)

# تعداد فایل‌ها
num_files = 972

for i in range(1, num_files + 1):

    excel_file = os.path.join(excel_path, f"{i}.xlsx")
    img_file = os.path.join(img_path, f"{i}.tif")

    # چک وجود فایل
    if not os.path.exists(excel_file) or not os.path.exists(img_file):
        print(f"Skip → file {i} not found.")
        continue

    # خواندن اکسل
    df = pd.read_excel(excel_file)

    # خواندن تصویر
    img = cv2.imread(img_file)

    if img is None:
        print(f"Cannot read image {img_file}")
        continue

    # رسم نقاط روی تصویر
    for idx, row in df.iterrows():
        cx = int(row["cx"])
        cy = int(row["cy"])

        # رسم دایره زرد با شعاع ۵ پیکسل
        cv2.circle(img, (cx, cy), 5, (0, 255, 255), -1)  # BGR = زرد

    # مسیر ذخیره تصویر خروجی
    out_file = os.path.join(save_path, f"{i}.tif")

    cv2.imwrite(out_file, img)
    print(f"Saved → {out_file}")

print("تمام شد.")
