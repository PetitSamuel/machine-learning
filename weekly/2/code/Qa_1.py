import matplotlib.pyplot as plt
from util import get_data

# Read in data
X1, X2, X, y, ytrain = get_data()

# Plot the data
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
plt.show()
