# Naive Bayes tính tay cho Iris 

import pandas as pd
import random
import math

# 1. Đọc dữ liệu Iris
df = pd.read_csv("data/iris.csv")
df.drop(["Id"], axis=1, inplace=True, errors="ignore")

print("5 dòng đầu của dữ liệu:")
print(df.head(), "\n")

X = df.iloc[:, :-1].values.tolist()   # 4 thuộc tính
y = df.iloc[:, -1].values.tolist()    # nhãn: setosa, versicolor, virginica

# 2. Chia train/test 80/20
data = list(zip(X, y))
random.seed(0)
random.shuffle(data)

X, y = zip(*data)
X = list(X)
y = list(y)

split = int(0.8 * len(X))   # 120 / 30
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print(f"Số mẫu train: {len(X_train)}, số mẫu test: {len(X_test)}\n")

# 3. TÍNH THÔNG SỐ NAIVE BAYES BẰNG TAY
#    - prior P(class)
#    - mean, variance cho từng feature trong từng class
classes = sorted(set(y_train))

# prior, mean, var: dict[label] = list theo từng feature
prior = {}
mean = {}
var = {}

for c in classes:
    # lấy tất cả mẫu thuộc class c
    X_c = [x for x, lab in zip(X_train, y_train) if lab == c]
    n_c = len(X_c)
    prior[c] = n_c / len(X_train)

    # số feature
    d = len(X_c[0])
    # tính mean thủ công
    m = [0.0] * d
    for x in X_c:
        for j in range(d):
            m[j] += x[j]
    m = [v / n_c for v in m]
    mean[c] = m

    # tính variance thủ công
    v = [0.0] * d
    for x in X_c:
        for j in range(d):
            diff = x[j] - m[j]
            v[j] += diff * diff
    v = [vv / n_c for vv in v]
    var[c] = v

# Hàm mật độ Gaussian cho 1 feature
def gaussian_pdf(x, mu, sigma2):
    eps = 1e-6
    sigma2 = sigma2 + eps
    coef = 1.0 / math.sqrt(2 * math.pi * sigma2)
    return coef * math.exp(-(x - mu) ** 2 / (2 * sigma2))

# Dự đoán 1 mẫu x bằng Naive Bayes tính tay
def predict_one_manual(x):
    posteriors = {}
    for c in classes:
        # log prior
        log_p = math.log(prior[c])
        # cộng log của từng feature
        for j in range(len(x)):
            p_xj = gaussian_pdf(x[j], mean[c][j], var[c][j])
            log_p += math.log(p_xj)
        posteriors[c] = log_p
    # chọn lớp có posterior lớn nhất
    best_class = max(posteriors, key=posteriors.get)
    return best_class, posteriors

# 4. Mô phỏng "tính tay" chi tiết cho Test 0
x0 = X_test[0]
true0 = y_test[0]
pred0, posts0 = predict_one_manual(x0)

print("===== Phân tích tính tay cho Test 0 =====")
print("x0 =", x0)
print("True label =", true0)
print("Predicted  =", pred0)

for c in classes:
    print(f"\n--- Lớp {c} ---")
    print("mean :", mean[c])
    print("var  :", var[c])

    # tính pdf từng thuộc tính
    pdfs = []
    for j in range(len(x0)):
        p_xj = gaussian_pdf(x0[j], mean[c][j], var[c][j])
        pdfs.append(p_xj)
    print("PDF từng thuộc tính:", pdfs)

    log_prior = math.log(prior[c])
    log_likelihood = sum(math.log(p) for p in pdfs)
    log_posterior = log_prior + log_likelihood
    print("log prior     =", log_prior)
    print("log likelihood=", log_likelihood)
    print("log posterior =", log_posterior)

# 5. Tính accuracy trên toàn bộ tập test
correct = 0
for x_i, y_i in zip(X_test, y_test):
    pred_i, _ = predict_one_manual(x_i)
    if pred_i == y_i:
        correct += 1

acc = correct / len(X_test)
print(f"\nAccuracy (Iris – Naive Bayes tính tay) = {acc:.4f}")
