# 🧠 My Machine Learning Journey

> This repository is not a polished project. It is a log — a record of every concept I picked up, every model I built, and every mistake I made while learning machine learning from scratch.

If you're here expecting clean, production-ready code, this isn't it. But if you're someone who's also learning, or someone curious about what a real learning path looks like — messy, incremental, and honest — then you're in the right place.

---

## 📖 What this repo is

Every folder and file here represents a step in my learning. Some notebooks are clean and complete. Some are half-finished. A few have bugs I didn't catch until later. I've left them as they are, because fixing them retroactively would erase the point — this is what learning actually looks like.

The topics roughly follow the order I learned them in, starting from basic statistics and visualization, moving through classical ML algorithms, and eventually reaching deep learning with PyTorch.

---

## 🗺️ The Path

### 📊 Statistics & Visualization
Where it all started — understanding data before trying to model it.

| File | What I learned |
|---|---|
| `Normal_Distribution.py` | Generating and visualizing normal distributions with NumPy and Seaborn |
| `HeatMap.py` | First attempt at heatmaps using Matplotlib (has a bug — `imshow` expects 2D, not 1D) |
| `Blood_Pressure_Hypothesis.py` | Paired t-test for hypothesis testing; also manually computed the t-statistic to verify |

---

### 🔍 Exploratory Data Analysis
Learning to understand a dataset before touching any model.

| File | Dataset | What I practiced |
|---|---|---|
| `EDA.ipynb` | WineQT (1143 rows, 13 features) | `.info()`, `.describe()`, correlation heatmap |

---

### 📈 Regression
My first real models — predicting continuous values.

| File | Dataset | Algorithm | Notes |
|---|---|---|---|
| `Linear_Regression.ipynb` | Test1.csv (Age → Salary) | Linear Regression | R² came out negative — learned why single-feature models fail |
| `Price_Pridiction.ipynb` | Book1.csv | Linear Regression | Multi-feature prediction with ₹ formatting on plots |
| `Rainfall_Predictioln.ipynb` | Weather dataset (1664 rows) | Linear Regression | 4 features, MSE + R² evaluation, residual plot |
| `Box_Office_Revenue.ipynb` | Box office dataset | XGBoost | Most complex regression project — full preprocessing, genre text encoding, train/val/test split |

**Regression diagnostics I explored:**

| File | What I learned |
|---|---|
| `Multicollinearity_Test.ipynb` | VIF analysis — found Height and Weight both exceed VIF > 10 |
| `Regularization_Techniques.ipynb` | Lasso (L1), Ridge (L2), and Elastic Net to combat overfitting |

---

### 🏷️ Classification
The bulk of my classical ML work — binary and multi-class problems.

| File | Dataset | Algorithm | Accuracy |
|---|---|---|---|
| `Logistic_Regressions.ipynb` | Breast Cancer (Binomial) | Logistic Regression | 95.61% |
| `Logistic_Regressions.ipynb` | Digits (Multinomial) | Logistic Regression | 96.66% |
| `Gaussian_NBC.ipynb` | Iris | Gaussian Naïve Bayes | 97.78% |
| `MNB.ipynb` | Hand-crafted spam (10 samples) | Multinomial NB | 66.67% (tiny dataset, expected) |
| `Text_Classisfication.ipynb` | Synthetic text data | Multinomial NB | Topic classification (Technology, etc.) |
| `Random_Forest.ipynb` | Titanic | Random Forest (100 trees) | — |
| `Heart_Disease_Prediction.ipynb` | Framingham CHD dataset | Logistic Regression | — |
| `Heart_Disease_Prediction.ipynb` | Heart prediction dataset | Random Forest | — |
| `POST_and_PRE_Pruning_DT.ipynb` | Iris | Decision Tree | ❌ Not completed |
| `Credit_Card_Fraud_Detection.ipynb` | 284,807 transactions | Random Forest | ❌ Model too large to run to completion |
| `Customer_Default_Prediction.ipynb` | Loan dataset (32,586 rows) | AdaBoost | Best preprocessing pipeline I wrote |

**Model evaluation practice:**

| File | What I learned |
|---|---|
| `Confusion_Matrix_Implementation.ipynb` | Confusion matrix, classification report, `ConfusionMatrixDisplay` |

---

### ⏳ Time Series
A detour into sequential data.

| File | Dataset | What I practiced |
|---|---|---|
| `time_series_analysis.py` | Stock price data | Line plots, ACF, ADF stationarity test, differencing, rolling moving average |

---

### 🤖 Deep Learning (PyTorch)
The most recent and ongoing part of the journey.

| File | What I built |
|---|---|
| `PyTorch.ipynb` | Tensor basics, autograd/backpropagation on 4 functions, full ANN for diabetes prediction |
| `ANN_model.pt` | Saved model from the ANN above (input: 8 → hidden: 20 → hidden: 20 → output: 2) |
| `MLP.ipynb` | Started an MLP for MNIST — PyTorch 2.14 + CUDA confirmed, architecture not yet written ❌ |
| `CNN_MNIST_CUDA_Assignment.ipynb` | Full CNN pipeline on MNIST — the most complete deep learning project in this repo |

**CNN MNIST — details:**

This was a practical assignment and the most structured notebook I've written so far. It runs fully on GPU (NVIDIA RTX 3050, CUDA 12.6).

Architecture — `MNISTCNN`:
```
Input: 1 × 28 × 28
→ Conv2d(1, 32, 3) + BatchNorm + ReLU + MaxPool   →  32 × 14 × 14
→ Conv2d(32, 64, 3) + BatchNorm + ReLU + MaxPool  →  64 × 7 × 7
→ Flatten → Linear(3136, 128) + ReLU + Dropout(0.3)
→ Linear(128, 10)
Total parameters: 421,834
```

Training config: 8 epochs · batch size 128 · Adam (lr=0.001) · CrossEntropyLoss · best-checkpoint saving

Results on 10,000 test images:

| Metric | Value |
|---|---|
| Overall test accuracy | **99.10%** |
| Overall error rate | 0.90% |
| Best validation accuracy | 98.98% |
| Hardest digit | 6 (2.19% error rate, 21 mistakes) |
| Most common confusion | 2 → 7 (7 images) |

Per-digit error rates:

| Digit | Test Images | Misclassified | Error Rate |
|---|---|---|---|
| 0 | 980 | 5 | 0.51% |
| 1 | 1135 | 2 | 0.18% |
| 2 | 1032 | 10 | 0.97% |
| 3 | 1010 | 5 | 0.50% |
| 4 | 982 | 3 | 0.31% |
| 5 | 892 | 4 | 0.45% |
| 6 | 958 | 21 | **2.19%** |
| 7 | 1028 | 6 | 0.58% |
| 8 | 974 | 14 | 1.44% |
| 9 | 1009 | 20 | 1.98% |

Also includes: training/validation loss & accuracy curves, confusion matrix heatmap, top-10 confusion pairs table, 3 misclassified image visualizations with written analysis, and saved outputs (`mnist_cnn_best.pt`, `mnist_per_digit_error_rates.csv`, `mnist_confusion_matrix.npy`).

---

## 📦 Datasets used

Most datasets are local and not included in this repo. They include:

- WineQT (wine quality)
- PIMA Indians Diabetes
- Titanic survival
- Framingham Heart Study (CHD risk)
- Breast Cancer Wisconsin
- Scikit-learn built-ins: Iris, Digits
- Stock price data
- Weather/rainfall data
- Box office revenue
- Credit card fraud (Kaggle)
- Loan default dataset
- BMI dataset
- MNIST handwritten digits (downloaded via torchvision)

---

## 🛠️ Stack

```
Python 3.11
NumPy · Pandas · Matplotlib · Seaborn
Scikit-learn · Statsmodels · SciPy · XGBoost
PyTorch 2.14 (CUDA)
Jupyter Notebook
```

---

## 🪲 Known bugs and incomplete files

I'm keeping these here on purpose — they're part of the learning record.

| File | Issue |
|---|---|
| `HeatMap.py` | `plt.imshow()` on a 1D Series — should use `sns.heatmap()` |
| `Linear_Regression.ipynb` | R² = -0.22 (Age is too weak a predictor for Salary alone) |
| `Regularization_Techniques.ipynb` | Elastic Net code is in a Markdown cell — never executed |
| `Box_Office_Revenue.ipynb` | `max_dapth=3` typo in XGBRegressor — `max_depth` was never set |
| `time_series_analysis.py` | `from matplotlib.lines import lineStyles` — this doesn't exist |
| `POST_and_PRE_Pruning_DT.ipynb` | Only loads Iris — pruning never implemented |
| `MLP.ipynb` | Only MNIST transform defined — rest is empty (the CNN notebook is the completed version of this idea) |
| `Heart_Disease_Prediction.ipynb` (v2) | Uses `RandomForestRegressor` for a classification task |

---

## 💬 A note

Every file in this repo was written while I was still figuring things out. The progress isn't linear, the naming isn't consistent, and not everything works perfectly. That's intentional — this is a learning log, not a portfolio.

If something here helped you, or if you spot something I should fix or understand better, feel free to open an issue.

---

*Learning in public — one notebook at a time.*
