
import pandas as pd
import random
import math

# ===============================
# 1. Đọc dữ liệu iris.csv
# ===============================
df = pd.read_csv("iris.csv")

if "Id" in df.columns:
    df.drop(["Id"], axis=1, inplace=True)

print("5 dòng đầu của dữ liệu:")
print(df.head(), "\n")

# tách X (4 thuộc tính) và y (nhãn)
X = df.iloc[:, :-1].values.tolist()
y = df.iloc[:, -1].values.tolist()

# ===============================
# 2. Chia train/test giống iris_with_knn.py
#    (shuffle với seed=0, 80% train, 20% test)
# ===============================
data = list(zip(X, y))
random.seed(0)
random.shuffle(data)
X, y = zip(*data)
X = list(X)
y = list(y)

split = int(0.8 * len(X))   # 120 train, 30 test
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print(f"Số mẫu train: {len(X_train)}, số mẫu test: {len(X_test)}\n")

# ===============================
# 3. Hàm tính khoảng cách Euclid (tự code)
# ===============================
def euclidean_distance(a, b):
    s = 0.0
    for i in range(len(a)):
        diff = a[i] - b[i]
        s += diff * diff
    return math.sqrt(s)

# chọn K (có thể để 1 cho trùng best_k bên iris_with_knn, hoặc 7 tùy bạn)
K = 1

# ===============================
# 4. KNN tính tay cho toàn bộ test
# ===============================
correct = 0

for idx, x in enumerate(X_test):
    true_label = y_test[idx]

    # 4.1. Tính khoảng cách từ x đến toàn bộ train
    distances = []  # (distance, label, index_train)
    for j, x_train in enumerate(X_train):
        d = euclidean_distance(x, x_train)
        distances.append((d, y_train[j], j))

    # 4.2. Sắp xếp và lấy K láng giềng gần nhất
    distances.sort(key=lambda t: t[0])
    k_nearest = distances[:K]

    # 4.3. Bỏ phiếu
    votes = {}  # {label: count}
    for d, lbl, _ in k_nearest:
        votes[lbl] = votes.get(lbl, 0) + 1

    # 4.4. Chọn nhãn có số phiếu lớn nhất
    best_label = max(votes, key=votes.get)

    if best_label == true_label:
        correct += 1

    # In 3 mẫu đầu cho dễ xem
    if idx < 3:
        print(f"Test {idx}: true = {true_label}, pred = {best_label}, votes = {votes}\n")

    # === MÔ PHỎNG TÍNH TAY CHI TIẾT CHO TEST 0 ===
    if idx == 0:
        print("===== Mô phỏng tính tay cho Test 0 =====")
        print("Điểm test X_test[0] =", x, ", nhãn thật =", true_label, "\n")
        for rank, (d, lbl, j) in enumerate(k_nearest, start=1):
            x_train = X_train[j]
            print(f"Láng giềng {rank}: train_index = {j}, label = {lbl}")
            s = 0.0
            for f in range(len(x)):
                diff = x[f] - x_train[f]
                diff2 = diff * diff
                s += diff2
                print(f"  Thuộc tính {f+1}: ({x[f]} - {x_train[f]})^2 = {diff2}")
            print(f"  Tổng bình phương = {s} -> khoảng cách = sqrt({s}) = {math.sqrt(s)}\n")
        print("Votes cho Test 0:", votes)
        print("Nhãn dự đoán cho Test 0:", best_label)
        print("========================================\n")

# ===============================
# 5. Accuracy cuối cùng
# ===============================
accuracy = correct / len(X_test)
print(f"Accuracy (Iris – KNN tính tay, K = {K}) = {accuracy:.4f}")
