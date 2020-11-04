import pandas as pd 
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
model = pd.read_csv("new_dataset.csv")
x = df.iloc[:, [1,3,5,6]] # Will give you columns 2 and 3 i.e 'petal_length' and 'petal_width'
y = df.iloc[:, 7] # Label column i.e 'Target'
#X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, test_size=0.2,random_state=109
