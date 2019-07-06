import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
name=['sepal-length','sepal-width','petal-length','petal-width','class']
url='http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
dataset=pandas.read_csv(url,names=name)
array=dataset.values
X=array[:,0:4]
Y=array[:,4]
test_size=0.20
seed=3
X_train,X_test,Y_train,Y_test=model_selection.train_test_split(X,Y,test_size=test_size,random_state=seed)
knn=KNeighborsClassifier()
knn.fit(X_train,Y_train)
#mean accuracy on the given test data
print(knn.score(X_test,Y_test))







