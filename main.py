import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,plot_confusion_matrix, confusion_matrix, accuracy_score
from sklearn import preprocessing

# Reading dataset
df = pd.read_csv("breast-cancer.csv") 

# Seting features and label
X = df.drop(['diagnosis'],axis=1)
y = df['diagnosis']

# Scaling values between 0 and 1
min_max_scaler = preprocessing.MinMaxScaler()
X = min_max_scaler.fit_transform(X)

# Spliting dataset into training and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training model
classifier=LogisticRegression().fit(X_train, y_train) 

# Ploting results
label_predicted = classifier.predict(X_test)
print(classification_report(y_test,label_predicted))
print('accuracy = ',round(accuracy_score(y_test,label_predicted),2),'\n\n')
plot_confusion_matrix(classifier,X_test,y_test,
                                 cmap=plt.cm.Blues,
                                 normalize='true')
plt.title('Matriz de confusao')
plt.show()