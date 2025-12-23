# Câu 2: Nhận dạng ký tự dùng KNN (tự cài đặt, không dùng class KNN)
# Database: UCI Letter Recognition

import pandas as pd
import random
import math

# 1. Đọc dữ liệu
df = pd.read_csv("letter-recognition.data", header=None)

print("5 dòng đầu của dữ liệu:")
print(df.head(), "\n")

X_all = df.iloc[:, 1:].values.tolist()   # 16-D features
y_all = df.iloc[:, 0].values.tolist()    # label A..Z

# 2. Chia train/test giống file dùng KNN của thầy

data = list(zip(X_all, y_all))
random.seed(0)
random.shuffle(data)
X_all, y_all = zip(*data)
X_all = list(X_all)
y_all = list(y_all)

split = int(0.8 * len(X_all))
X_train, X_test = X_all[:split], X_all[split:]
y_train, y_test = y_all[:split], y_all[split:]

print(f"Số mẫu train: {len(X_train)}, số mẫu test: {len(X_test)}\n")

# 3. Hàm khoảng cách Euclid & KNN tính tay
def euclidean_distance(a, b):
    s = 0.0
    for i in range(len(a)):      # 16 thuộc tính
        diff = a[i] - b[i]
        s += diff * diff
    return math.sqrt(s)

def predict_one_manual(x, k):
    # Tính khoảng cách tới tất cả điểm train
    distances = []  # (d, label, index_train)
    for idx, x_tr in enumerate(X_train):
        d = euclidean_distance(x, x_tr)
        distances.append((d, y_train[idx], idx))

    # Sắp xếp tăng dần theo khoảng cách và lấy k hàng xóm
    distances.sort(key=lambda t: t[0])
    k_near = distances[:k]

    # Bỏ phiếu
    votes = {}
    for d, lbl, _ in k_near:
        votes[lbl] = votes.get(lbl, 0) + 1

    # Chọn nhãn có số phiếu lớn nhất
    best_label = max(votes, key=votes.get)
    return best_label, votes, k_near

def score_manual(k):
    correct = 0
    for i, x in enumerate(X_test):
        pred, _, _ = predict_one_manual(x, k)
        if pred == y_test[i]:
            correct += 1
    return correct / len(X_test)

# 4. Demo tính tay cho Test 0 (mô phỏng "tính tay")
K = 1  
pred0, votes0, neighbors0 = predict_one_manual(X_test[0], K)
print(f"Test 0: true = {y_test[0]}, pred = {pred0}, votes = {votes0}")
print("Các láng giềng gần nhất của Test 0:")

for rank, (d, lbl, idx_tr) in enumerate(neighbors0, start=1):
    print(f"  Neighbor {rank}: dist = {d:.3f}, label = {lbl}, train_index = {idx_tr}")

print()