from sklearn import svm
from matplotlib import pyplot as plt
import numpy as np

svc = svm.SVC(kernel='linear', C=2000)  # C means penalty, large C means hard margin
x = [[1.0, 2.0], [2.0, 3.0], [3.0, 3.0], [2.0, 1.0], [3.0, 2.0]]
y = [1, 1, 1, -1, -1]
svc.fit(x, y)

w = svc.dual_coef_[0]  # w is (yiai) of support vectors
b = svc.intercept_[0]
print(w)
print(b)
print(svc.support_vectors_)

s = svc.support_vectors_
A = np.sum([w[i] * s[i][0] for i in range(len(w))])
B = np.sum([w[i] * s[i][1] for i in range(len(w))])
C = svc.intercept_[0]
print(A, B, C)
k = -A / B
b1 = -C / B
bdown = s[0][1] - k * s[0][0]
bup = s[1][1] - k * s[1][0]

plt.scatter([x[i][0] for i in range(len(x))], [x[i][1] for i in range(len(x))])
xfit = np.linspace(0, 4)

plt.plot(xfit, k * xfit + b1)
plt.plot(xfit, k * xfit + bup)
plt.plot(xfit, k * xfit + bdown)
plt.show()
