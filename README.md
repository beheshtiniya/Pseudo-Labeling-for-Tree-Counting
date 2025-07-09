
📘 Iterative Pseudo-Labeling for Tree Counting in Dense Forests Using Aerial Imagery and Weak Point Annotations
These codes is adapted from the GitHub repository at:
🔗 https://github.com/sddpltwanqiu/TreeCountNet/tree/main

The script <span style="color:#007acc"><strong><em>train.py</em></strong></span> is executed via <span style="color:#007acc"><strong><em>main.py</em></strong></span>.

If your goal after training is solely to predict the center coordinates of trees, you can simply run the script:
<span style="color:#d38b00"><strong><em>test just for predict.py</em></strong></span>

The final outputs of this work—including the optimized dataset, trained model, and corresponding weights—are available at:
🔗 https://yun.ir/0bl1a3


# 🔧 Hyperparameter Optimization and Final Evaluation for Tree Counting

This documentation describes the workflow of two important evaluation scripts:

1. `test2_2 find best 4 parameter.py` – Automatic hyperparameter search
2. `test3.py` – Final prediction and coordinate extraction using the best parameters

---

## 📌 1. `test2_2 find best 4 parameter.py`

This script performs an **exhaustive grid search** over four postprocessing parameters to optimize the tree counting accuracy (based on MAPE):

### 🎯 Goal:
To find the best values for:
- `threshold`: binarization threshold for predicted density maps
- `kernel_size`: morphological kernel size to connect nearby regions
- `size_spot`: minimum connected component size to keep (initial filtering)
- `size_spot2`: minimum region size after morphological processing (final cleanup)

### 🔁 Search Ranges:
| Parameter     | Range               |
|---------------|---------------------|
| `threshold`   | 50 to 140 (step 10) |
| `kernel_size` | 35 to 70 (step 5)   |
| `size_spot`   | 6 to 22 (step 2)    |
| `size_spot2`  | 18 to 22 (step 2)   |

For each combination, the script:
- Applies the postprocessing pipeline on test images
- Computes MAE and MAPE compared to ground truth
- Prints and tracks the best configuration (i.e., lowest MAPE)

### ✅ Output:
The best parameter set is printed at the end, e.g.:
```

Best threshold: 30
Best kernel size: 13
best\_size\_spot: 8
best\_size\_spot2: 19
Minimum MAPE: 7.73%

````

---

## 📌 2. Applying Optimal Parameters in `test3.py`

Once the best parameters are found, they are **manually copied into `test3.py`**, replacing the default values in its postprocessing section:

### Example (in `test3.py`):

```python
# Replace with optimal values from test2_2.py
threshold = 30
kernel = np.ones((13, 13), np.uint8)
size_spot = 3
size_spot2 = 19
````

---

## 📌 3. `test3.py` – Final Evaluation and Coordinate Extraction

This script performs the **final evaluation phase** using the optimized settings.

### 🧠 Main Functions:

* Loads the trained model (`weights_best_val_loss.pth`)
* Applies it to all test images
* Applies the binarization, morphological filtering, and connected component analysis
* Computes prediction metrics (MAE, MAPE)
* Saves filtered dotmaps and center coordinates of each detected tree

---

## 📤 Output Files of `test3.py`

| File / Folder      | Description                                             |
| ------------------ | ------------------------------------------------------- |
| `filtered/`        | Binary dotmaps after filtering                          |
| `_pre_result.xlsx` | Summary: image names, predicted vs. GT count            |
| `*.xlsx` per image | List of `(cx, cy)` coordinates of detected tree centers |
| Console output     | MAE, MAPE, scatter plot, regression line                |

### 📈 Visualization:

* Plots predicted vs. actual counts
* Shows both ideal line and linear regression fit
* Helps visually assess prediction accuracy and bias

---

## ✅ Final Result:

By combining the **optimized hyperparameters from `test2_2.py`** with the **structured outputs of `test3.py`**, we obtain:

* High-accuracy tree count predictions
* Accurate center coordinates of detected trees
* Publication-ready visual and tabular outputs

This pipeline forms the **final stage of the iterative pseudo-labeling framework** for tree counting in dense forests using TreeCountNet.



