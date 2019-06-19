import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
from Preprocess import *
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC  
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')

bankdata = pd.read_csv(r"C:\Users\arkha\OneDrive\Desktop\ML\Support Vector Machines\bill_authentication.csv") 

#Preprocessing
#encode_numeric_zscore(bankdata, 'Variance')
#encode_numeric_zscore(bankdata, 'Skewness')
#encode_numeric_zscore(bankdata, 'Curtosis')
#encode_numeric_zscore(bankdata, 'Entropy')

x = bankdata.drop('Class', axis=1)  
y = bankdata['Class'] 

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20)  

svclassifier = SVC(kernel='linear')  
svclassifier.fit(x_train, y_train) 

pred = svclassifier.predict(x_test)

score = metrics.accuracy_score(y_test, pred)
print("Final accuracy: {}".format(score))

def plot_confusion_matrix(cm, names, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(names))
    plt.xticks(tick_marks, names, rotation=45)
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
plot_confusion_matrix(cm_normalized, y.unique(), title='Normalized confusion matrix')
plt.show()

################################################################################