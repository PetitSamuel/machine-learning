from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerBase

df = pd.read_csv("week2.csv", comment='#')
print(df.head())
X1 = df.iloc[:, 0]
X2 = df.iloc[:, 1]
X = np.column_stack((X1, X2))
y = df.iloc[:, 2]

# Plot the cost function evolution
plt.rc('font', size=20)
plt.rcParams['figure.constrained_layout.use'] = True
neg = plt.scatter(X1[y < 0], X2[y < 0], color='red', marker=".")
pos = plt.scatter(X1[y > 0], X2[y > 0], color='blue', marker="+")
plt.xlabel("X1 (first feature)")
plt.ylabel("X2 (second feature)")
plt.legend((neg, pos), ["positive", "negative"],
           scatterpoints=1,
           loc='lower right',
           ncol=2,
           fontsize=12)
# plt.show()

ytrain = np.sign(y)
model = LogisticRegression(penalty='none', solver='saga')
model.fit(X, ytrain)
print("intercept, slope", model.intercept_, model.coef_)

ypred = model.predict(X)
pos = plt.scatter(X1[ypred > 0], X2[ypred > 0],
                  color='black', marker="+")
neg = plt.scatter(X1[ypred < 0], X2[ypred < 0],
                  color='orange', marker=".")
plt.show()
