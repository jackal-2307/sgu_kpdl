# Câu 2: Nhận dạng ký tự dùng thuật toán Naive Bayes
# Database: UCI Letter Recognition (26 lớp A..Z, 16-D feature, 20 000 mẫu)

import pandas as pd
import numpy as np
import random
import math
# 1. Đọc dữ liệu
# Mỗi dòng: label (A..Z), 16 đặc trưng số nguyên
df = pd.read_csv("data2/letter-recognition.data", header=None)

print("5 dòng đầu của dữ liệu:")
print(df.head(), "\n")

X_all = df.iloc[:, 1:].values          # 16 chiều
y_all = df.iloc[:, 0].values           # nhãn A..Z

# 2. Chia train/test (80/20)
data = list(zip(X_all, y_all))
random.seed(0)
random.shuffle(data)
X_all, y_all = zip(*data)
X_all = np.array(X_all)
y_all = np.array(y_all)

split = int(0.8 * len(X_all))          # 80% train, 20% test
X_train, X_test = X_all[:split], X_all[split:]
y_train, y_test = y_all[:split], y_all[split:]

print(f"Số mẫu train: {len(X_train)}, số mẫu test: {len(X_test)}\n")

classes = sorted(list(set(y_train)))

# 3. TÍNH THÔNG SỐ NAIVE BAYES
#    - prior P(class)
#    - mean, variance cho từng feature trong từng class
prior = {}
mean = {}
var = {}

for c in classes:
    X_c = X_train[y_train == c]        # các mẫu thuộc lớp c
    n_c = X_c.shape[0]
    prior[c] = n_c / len(X_train)
    mean[c] = X_c.mean(axis=0)
    var[c] = X_c.var(axis=0)

# Hàm mật độ Gaussian cho vector (tính theo từng feature)
def gaussian_pdf_vec(x, mu, sigma2):
    eps = 1e-6
    sigma2 = sigma2 + eps
    coef = 1.0 / np.sqrt(2 * math.pi * sigma2)
    return coef * np.exp(- (x - mu) ** 2 / (2 * sigma2))

# Dự đoán 1 mẫu x (16-D) bằng Naive Bayes
def predict_one(x):
    posteriors = {}
    for c in classes:
        # log P(c)
        log_p = math.log(prior[c])
        # log P(x | c) = tổng log Gaussian trên từng feature (giả sử độc lập)
        pdf = gaussian_pdf_vec(x, mean[c], var[c])
        log_p += np.sum(np.log(pdf))
        posteriors[c] = log_p
    # chọn lớp có log-posterior lớn nhất
    best_class = max(posteriors, key=posteriors.get)
    return best_class, posteriors

# 4. Đánh giá trên tập test

correct = 0
preds = []

for x_i, y_i in zip(X_test, y_test):
    pred_i, _ = predict_one(x_i)
    preds.append(pred_i)
    if pred_i == y_i:
        correct += 1

accuracy = correct / len(y_test)
print(f"Accuracy (Letter – Naive Bayes) = {accuracy:.4f}\n")

# In vài kết quả đầu cho đẹp
print("Một vài dự đoán trên tập test:")
for i in range(10):
    print(f"  Sample {i}: true = {y_test[i]}, pred = {preds[i]}")


# 5. Phân tích chi tiết cho Test 0 (tùy, để ghi báo cáo)
x0 = X_test[0]
true0 = y_test[0]
pred0, posts0 = predict_one(x0)

print("\n===== Phân tích Test 0 =====")
print("Feature vector x0 (16-D):", x0)
print("True label:", true0)
print("Predicted :", pred0)

# Lấy top-3 lớp có posterior cao nhất để xem
sorted_posts = sorted(posts0.items(), key=lambda kv: kv[1], reverse=True)
print("\nTop-3 classes theo log-posterior:")
for c, lp in sorted_posts[:3]:
    print(f"  class {c}: log-posterior = {lp:.3f}")
