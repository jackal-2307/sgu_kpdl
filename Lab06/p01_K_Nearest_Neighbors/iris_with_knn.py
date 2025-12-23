import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt
from K_Nearest_Neighbors import K_Nearest_Neighbors as KNN

df = pd.read_csv("iris.csv")

# Một số file có thêm cột Id
if "Id" in df.columns:
    df.drop(["Id"], axis=1, inplace=True)

print("5 dòng đầu của dữ liệu:")
print(df.head(), "\n")

# ---- Visualize: scatter 2 thuộc tính Petal.Length & Petal.Width ---
feat_names = df.columns.tolist()      # ['Sepal.Length', ..., 'Species']
X_all = df.iloc[:, :-1].values        # 4 features
y_all = df.iloc[:, -1].values         # nhãn

ix, iy = 2, 3  # Petal.Length, Petal.Width

plt.figure()
for label in sorted(set(y_all)):
    mask = (y_all == label)
    plt.scatter(
        X_all[mask, ix],
        X_all[mask, iy],
        label=label,
        alpha=0.8
    )

plt.xlabel(feat_names[ix])
plt.ylabel(feat_names[iy])
plt.title("Iris dataset - scatter by class")
plt.legend()
plt.tight_layout()
plt.show()

# 1. FEATURE ENGINEERING
#    (ở đây dữ liệu đã sạch → chỉ cần tách X, y)

X = X_all.tolist()          # chuyển sang list of list cho tiện dùng về sau
y = y_all.tolist()

# 2. SPLITTING THE DATA (chia train/test)
data = list(zip(X, y))
random.seed(0)          # để lần nào chạy cũng chia giống nhau
random.shuffle(data)
X, y = zip(*data)
X = list(X)
y = list(y)

split = int(0.8 * len(X))   # 80% train, 20% test 
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print(f"Số mẫu train: {len(X_train)}, số mẫu test: {len(X_test)}\n")


# 3. CHUẨN BỊ DỮ LIỆU CHO CLASS KNN CỦA THẦY (TRAIN THE MODEL)
#    KNN của thầy nhận dạng: {label: [[features], ...]}
labels = sorted(set(y))
train_data = {label: [] for label in labels}
test_data  = {label: [] for label in labels}

for features, label in zip(X_train, y_train):
    train_data[label].append(features)

for features, label in zip(X_test, y_test):
    test_data[label].append(features)

# Hàm đánh giá accuracy cho 1 giá trị k bất kỳ
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


# 4. HYPERPARAMETER TUNING (TÌM k TỐT NHẤT) + LINE PLOT

k_values = list(range(1, 16, 2))   # 1,3,5,...,15
acc_values = []

print("Tuning k:")
for k in k_values:
    acc_k = accuracy_for_k(k)
    acc_values.append(acc_k)
    print(f"  k = {k:2d}  ->  accuracy = {acc_k:.4f}")

# Vẽ đường accuracy theo k (line plot)
plt.figure()
plt.plot(k_values, acc_values, marker="o")
plt.xlabel("k")
plt.ylabel("Accuracy")
plt.title("KNN (thầy) - Accuracy vs k trên Iris")
plt.xticks(k_values)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Chọn k tốt nhất
best_index = int(np.argmax(acc_values))
best_k = k_values[best_index]
best_acc = acc_values[best_index]
print(f"\nBest k = {best_k}, accuracy = {best_acc:.4f}\n")


# 5. TRAIN THE MODEL VỚI k TỐT NHẤT & ASSESS PERFORMANCE

final_knn = KNN(train_data, k=best_k)

# Tự đánh giá lại (giống accuracy_for_k nhưng đồng thời in vài mẫu)
correct = 0
total = 0

print("Một vài dự đoán trên tập test:")
for i, (features, true_label) in enumerate(zip(X_test, y_test)):
    pred = final_knn.predict(features)
    if pred == true_label:
        correct += 1
    total += 1
    if i < 5:  # in 5 mẫu đầu
        print(f"  Sample {i}: true = {true_label:10s}, pred = {pred:10s}")

final_acc = correct / total
print(f"\nFinal accuracy with k = {best_k}: {final_acc:.4f}")

# Demo dự đoán điểm mới
sample = [5.0, 3.4, 1.5, 0.2]
pred_new = final_knn.predict(sample)
print("\nPredict for new sample", sample, "->", pred_new,
      ", confidence =", final_knn.confidence)

