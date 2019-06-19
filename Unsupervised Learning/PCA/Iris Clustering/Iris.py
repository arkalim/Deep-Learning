import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

irisdata = pd.read_csv(r"C:\Users\arkha\OneDrive\Desktop\ML\PCA\Iris Classification\iris.csv") 

#Separating the data into features and labels
x = irisdata.drop('species', axis=1)  
y = irisdata['species']

#Preprocessing
x = StandardScaler().fit_transform(x)

#PCA Algorithm 
pca = PCA(n_components=2)                     #No. of Dimensions Needed = 2

#Getting the principle features
#Fit and Transform(map) the Features 
principal_features = pca.fit_transform(x) 

#Converting principle features into dataframe 
principal_df = pd.DataFrame(data = principal_features, columns = ['Feature 1', 'Feature 2'])

#Concatenating the features and the labels
final_df = pd.concat([principal_df, y], axis = 1)

#Saving the edited .csv file
final_df.to_csv(r"C:\Users\arkha\OneDrive\Desktop\ML\PCA\Iris Classification\iris_edit.csv")

#Converting the principal features into array
final_data = final_df.values

#Calculating the Variance
#Variance represents the amount of data contained in the Principle Features
#Adding the variance of the principal features, we get the total Variance retained
variance = pca.explained_variance_ratio_
print(variance)
print("Variance Retained:",variance[0]+variance[1])

######################################################################################################
#Plotting the data in 2D
labels = y.unique()
colors = ['y', 'r', 'g','c','k','b']

for color, label in zip(colors,labels):
    selected_rows = (y == label) #creates a list of boolean values representing points belonging to a particular label
    plt.scatter( final_data[selected_rows,0],final_data[selected_rows,1], c=color , s = 50)
    
plt.xlabel('Principal Feature 1')
plt.ylabel('Principal Feature 2')
plt.title('2 Component PCA')
plt.legend(labels)   
plt.show()
######################################################################################################
