import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

from pandas.plotting import scatter_matrix
# from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn import tree

print("Dataset:")
dataset = pd.read_csv('lung_cancer_examples.csv')
print(len(dataset))
print(dataset.head())

scatter_matrix(dataset)
# pyplot.show()

A = dataset[dataset.Result == 1]
B = dataset[dataset.Result == 0]

plt.scatter(A.Age, A.Smokes, color="Black", label="1", alpha=0.4)
plt.scatter(B.Age, B.Smokes, color="Blue", label="0", alpha=0.4)

plt.xlabel("Age")
plt.ylabel("Smoke")
plt.legend()
plt.title("Smoke vs Age")
# plt.show()

plt.scatter(A.Age, A.Alcohol, color="Black", label="1", alpha=0.4)
plt.scatter(B.Age, B.Alcohol, color="Blue", label="0", alpha=0.4)

plt.xlabel("Age")
plt.ylabel("Alcohol")
plt.legend()
plt.title("Alcohol vs Age")
# plt.show()

plt.scatter(A.Smokes, A.Alcohol, color="Black", label="1", alpha=0.4)
plt.scatter(B.Smokes, B.Alcohol, color="Blue", label="0", alpha=0.4)

plt.xlabel("Smoke")
plt.ylabel("Alcohol")
plt.legend()
plt.title("Alcohol vs Smoke")
# plt.show()

X = dataset.iloc[:, 3:5]
Y = dataset.iloc[:, 6]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=0, test_size=0.28)

sc_x = StandardScaler()
X_train = sc_x.fit_transform(X_train)
X_test = sc_x.transform(X_test)

print("-----KNN ALGORITHM----")
a = math.sqrt(len(Y_train))
print(a)

classifier = KNeighborsClassifier(n_neighbors=5, p=2, metric='euclidean')

classifier.fit(X_train, Y_train)

Y_pred = classifier.predict(X_test)
print(Y_pred)

cm = confusion_matrix(Y_test, Y_pred)
print("Confusion Matrix:  ")
print(cm)
print("In confusion Matrix :----")
print("Position 1.1 shows the patients don't have Cancer, In this case = ", cm[0][0])
print("Position 1.2 shows the number of patients that have high risk of Cancer, In this case = ", cm[0][1])
print("Position 2.1 shows the Incorrect Value, In this case = ", cm[1][0])
print("Position 2.2 shows the correct number of patients that have Cancer, In this case = ", cm[1][1])

print('F1 Score: ', (f1_score(Y_test, Y_pred))*100)
print('Accuracy: ', (accuracy_score(Y_test, Y_pred))*100)

print("-------Using Decision Tree-------")
c = tree.DecisionTreeClassifier()
c.fit(X_train, Y_train)
accu_train = np.sum(c.predict(X_train) == Y_train) / float(Y_train.size)
accu_test = np.sum(c.predict(X_test) == Y_test) / float(Y_test.size)

print('Classifier Accuracy on train', accu_train*100)
print('Classifier Accuracy on test', accu_test*100)
