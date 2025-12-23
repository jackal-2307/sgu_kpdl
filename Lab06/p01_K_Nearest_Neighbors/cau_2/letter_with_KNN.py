
# Câu 2: Nhận dạng ký tự dùng K Nearest Neighbors 
# Database: UCI Letter Recognition
# 26 lớp A..Z, 16-D feature vectors, 20 000 samples

import pandas as pd
import random
import numpy as np
from K_Nearest_Neighbors import K_Nearest_Neighbors as KNN

# =========================
# 1. Đọc dữ liệu
# =========================
# Mỗi dòng: label (A..Z), 16 đặc trưng số nguyên
df = pd.read_csv("letter-recognition.data", header=None)

print("5 dòng đầu của dữ liệu:")
print(df.head(), "\n")

X_all = df.iloc[:, 1:].values.tolist()   # 16-D features
y_all = df.iloc[:, 0].values.tolist()    # label A..Z

# =========================
# 2. Chia train/test (80/20)
# =========================
data = list(zip(X_all, y_all))
random.seed(0)          # để lần nào chạy cũng giống nhau
random.shuffle(data)
X_all, y_all = zip(*data)
X_all = list(X_all)
y_all = list(y_all)

split = int(0.8 * len(X_all))   # 80% train, 20% test
X_train, X_test = X_all[:split], X_all[split:]
y_train, y_test = y_all[:split], y_all[split:]

print(f"Số mẫu train: {len(X_train)}, số mẫu test: {len(X_test)}\n")

# =========================
# 3. Chuẩn bị dữ liệu cho class KNN của thầy
#    KNN nhận dạng: {label: [[features], ...]}
# =========================
labels = sorted(set(y_all))               # 'A'..'Z'
train_data = {label: [] for label in labels}
test_data  = {label: [] for label in labels}

for features, label in zip(X_train, y_train):
    train_data[label].append(features)

for features, label in zip(X_test, y_test):
    test_data[label].append(features)

# Hàm tính accuracy cho 1 giá trị k
def accuracy_for_k(k: int) -> float:
    model = KNN(train_data, k=k)
    correct = 0
    total = 0
    for group in test_data:
        for feature_set in test_data[group]:
            pred = model.predict(feature_set)
            if pred == group:
                correct += 1
            total += 1
    return correct / total

# =========================
# 4. Thử một vài giá trị k (hyperparameter tuning nhẹ)
# =========================
k_values = [1, 3, 5, 7, 9]
acc_values = []

print("Tuning k (Letter Recognition):")
for k in k_values:
    acc_k = accuracy_for_k(k)
    acc_values.append(acc_k)
    print(f"  k = {k:2d} -> accuracy = {acc_k:.4f}")

best_index = int(np.argmax(acc_values))
best_k = k_values[best_index]
best_acc = acc_values[best_index]
print(f"\nBest k = {best_k}, accuracy = {best_acc:.4f}\n")

# =========================
# 5. Train model cuối cùng & đánh giá
# =========================
final_knn = KNN(train_data, k=best_k)

correct = 0
total = 0

print("Một vài dự đoán trên tập test:")
for i, (features, true_label) in enumerate(zip(X_test, y_test)):
    pred = final_knn.predict(features)
    if pred == true_label:
        correct += 1
    total += 1
    if i < 10:  # in 10 mẫu đầu
        print(f"  Sample {i}: true = {true_label}, pred = {pred}")

final_acc = correct / total
print(f"\nFinal accuracy (Letter, dùng KNN của thầy) với k = {best_k}: {final_acc:.4f}")

# Demo dự đoán 1 mẫu bất kỳ trong test
sample = X_test[0]
true_label = y_test[0]
pred_new = final_knn.predict(sample)
print("\nDemo 1 mẫu:")
print("  True label   :", true_label)
print("  Predicted    :", pred_new)
print("  Confidence   :", final_knn.confidence)
