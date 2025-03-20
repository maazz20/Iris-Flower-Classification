# Importing Libraries

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline





# Loading Dataset

iris = pd.read_csv('C:/Users/Maaz Javed/Downloads/IRIS.csv')
iris.head()





# Renaming the columns 

iris = iris.rename ( columns = {'sepal_length': 'Sepal_length' , 
                            'sepal_width' : 'Sepal_width' , 
                            'petal_length': 'Petal_length' , 
                            'petal_width': 'Petal_width' ,
                            'species': 'Species'})
iris.head()



# Visualise

iris.describe()
data=iris.values



# slicing the matrices
X=data[:,0:4]
Y=data[:,4]

print(X.shape)
print(X)

print(Y.shape)
print(Y)

# Split into testing and training



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X,Y, test_size=0.2)



print(X_train.shape)
print(X_train)

print(y_train.shape)
print(y_train)

print(X_test.shape)
print(X_test)

print(y_test.shape)
print(y_test)





# Pairplot visualization

sns.pairplot(iris, hue="Species")

# Model 1: Decision Tree Model

from sklearn.tree import DecisionTreeClassifier
model_DTC = DecisionTreeClassifier()
model_DTC.fit(X_train, y_train)

prediction1= model_svc.predict(X_test)

#calculate the accuracy
from sklearn.metrics import accuracy_score
print("Model 1 Accuracy=" , accuracy_score(y_test, prediction1))



# Model 2: Logistic Regression

from sklearn.linear_model import LogisticRegression  

model_LR = LogisticRegression(max_iter=200)
model_LR.fit(X_train, y_train)
prediction2 = model_LR.predict(X_test)

from sklearn.metrics import accuracy_score
print("Logistic Regression Accuracy:", accuracy_score(y_test, prediction2))



# New data for prediction
X_new = np.array([[3, 2, 1, 0.2], [4.9, 2.2, 3.8, 1.1], [5.3, 2.5, 4.6, 1.9]])

# Predicting the sizes of the iris flowers
predicted_sizes = model_DTC.predict(X_new)

# Output the predicted sizes
print(predicted_sizes)

