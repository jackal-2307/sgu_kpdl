import pandas as pd
import random
from Naive_Bayes import Naive_Bayes

# DATA
df = pd.read_csv("data/iris.csv")
df.drop(['Id'], axis=1, inplace=True, errors='ignore')

print("5 dòng đầu của dữ liệu:")
print(df.head(), "\n")

data_set = df.values.tolist()

# shuffle data set
random.seed(0)
random.shuffle(data_set)
random.shuffle(data_set)

# divide set in to training and test data
train_data = pd.DataFrame(data_set[:120])
test_data = pd.DataFrame(data_set[120:])
print(f"Số mẫu train: {train_data.shape[0]}, số mẫu test: {test_data.shape[0]}\n")

# CLASSIFIER
nb = Naive_Bayes(train_data)
nb.test(test_data)

# Demo: Expected for 1 sample in file test
# Get first line in test

row0 = test_data.iloc[0]
features0 = row0.values[0:4]  
true_label0 = row0.values[4] 

pred0 = nb.predict(features0)

pred0 = nb.predict(features0)

print("\n===== Demo prediction for test sample[0] =====")
print("Features:", features0)
print("True label :", true_label0)
print("Predicted  :", pred0)

#  Demo: prediction for any new model
# =========================
new_sample = [5.0, 3.4, 1.5, 0.2]   
pred_new = nb.predict(new_sample)

print("\n===== Predictions for the new model =====")
print("New sample:", new_sample)
print("Predicted :", pred_new)