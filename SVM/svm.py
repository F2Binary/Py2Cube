import pandas as pd 
import numpy as np
from sklearn import svm,metrics
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
df = pd.read_csv("new_dataset.csv")
clf = svm.SVC(kernel='linear')
columns = "Time Length_Scramble Length_Solution TPS".split()
x  = df.iloc[:, [1,3,5,6]] 
target  = df.iloc[:, 7] # Label column i.e 'Target'
data = pd.DataFrame(x,columns=columns)
y = target
X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.3)
#print(X_train.shape,y_train.shape)
#print(X_test.shape, y_test.shape)
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
predictions = clf.predict(X_test)
#plt.scatter(y_test,predictions)
#plt.xlabel("True Values")
#plt.ylabel("Predictions")
#plt.show()
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred,average ='macro'))
print("F1-Score:",metrics.f1_score(y_test,y_pred))
metrics.plot_roc_curve(clf, X_test, y_test)
plt.show()
