import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
from Preprocess import *
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC  
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

irisdata = pd.read_csv(r"C:\Users\arkha\OneDrive\Desktop\ML\Support Vector Machines\iris.csv") 

#Preprocessing
encode_numeric_zscore(irisdata, 'sepal_l')
encode_numeric_zscore(irisdata, 'sepal_w')
encode_numeric_zscore(irisdata, 'petal_l')
encode_numeric_zscore(irisdata, 'petal_w')

species = encode_text_index(irisdata,'species')

x = irisdata.drop('species', axis=1)  
y = irisdata['species'] 

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)  

#Different Kernels Available
#svclassifier = SVC(kernel='linear')
#svclassifier = SVC(kernel='poly', degree=8)
svclassifier = SVC(kernel='rbf') 
#svclassifier = SVC(kernel='sigmoid')  

svclassifier.fit(x_train, y_train) 

pred = svclassifier.predict(x_test)

score = metrics.accuracy_score(y_test, pred)
print("Final accuracy: {}".format(score))

def plot_confusion_matrix(cm, names, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(names))
    plt.xticks(tick_marks, names)
    plt.yticks(tick_marks, names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


################################################################################
#Plot Normalised Confusion Matrix

cm = confusion_matrix(y_test, pred)
np.set_printoptions(precision=2)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print('Normalized confusion matrix')
print(cm_normalized)
plt.figure()
plot_confusion_matrix(cm_normalized, species, title='Normalized confusion matrix')
plt.show()

################################################################################