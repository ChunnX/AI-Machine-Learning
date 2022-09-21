from sklearn.semi_supervised import _label_propagation
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from icecream import ic
import numpy as np


digits = load_digits()
x = digits.data
y = digits.target

n_total_sample = len(digits.data)
ic(n_total_sample)
# 只有10%有label，其余都没有label
n_labeled_points = int(n_total_sample * 0.1)
ic(n_labeled_points)


model = LogisticRegression()
model.fit(x[:n_labeled_points], y[:n_labeled_points])
predict_y = model.predict(x[n_labeled_points:])
true_y = y[n_labeled_points:]
print("准确率", (predict_y == true_y).sum()/len(true_y))

y_train = np.copy(y)
y_train[n_labeled_points:] = -1
lp_model = _label_propagation.LabelSpreading(gamma=0.25, max_iter=5)
lp_model.fit(x, y_train)
predict_y = lp_model.predict(x[n_labeled_points:])
true_y = y[n_labeled_points:]
print("准确率", (predict_y == true_y).sum()/len(true_y))