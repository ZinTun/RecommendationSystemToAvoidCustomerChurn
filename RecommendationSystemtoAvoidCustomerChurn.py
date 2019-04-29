# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os
os.chdir('/Users/zintun/Documents/MSBA/4th Sem/capstone')


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.backends.backend_pdf
import seaborn as sns
from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from itertools import cycle
from sklearn.metrics import r2_score

from sklearn.metrics import log_loss
random_state = 1001

filename = 'test.xlsx'
sheetname = 'train_data'

xl = pd.ExcelFile(filename)
data = xl.parse(sheetname)
data.head()


dataNVOL = data[data.inVOL ==0]
dataNVOL['label'] = 4 #"NVOL"
dataBilled = data[(data.Billed == 1)]
dataBilled['label']= 3#"BILLED"
dataVE = data[data.regionCode == 1]
dataVE['label']= 2 #"VE"

categorical_features = ['regionCode']
#continuous_features = ['numofuserperorg', 'Org_totalnumOfuserSessionperuser', 'Org_Totalnumofprocessor', 'Org_totalissuerProfile', 'Org_sessioninLast3month', 'Org_sessiontinlast6month','Org_lastaccess','Org_SinceYear']


datax = data.drop(data[data.inVOL==0].index)
datax = datax.drop(datax[datax.Billed == 1].index)
datax = datax.drop(datax[datax.regionCode == 1].index)


#datax = datax.drop(['inVOL','orgid','total_usersession','vpalastaccesscount','vpalastaccesscount_month','Billed','last6month','sinceyear','regionCode'],axis = 1)
#datax.head()
#dataNVOL = dataNVOL.drop(['inVOL','orgid','total_usersession','vpalastaccesscount','vpalastaccesscount_month','Billed','last6month','sinceyear','regionCode'],axis = 1)
#dataBilled = dataBilled.drop(['inVOL','orgid','total_usersession','vpalastaccesscount','vpalastaccesscount_month','Billed','last6month','sinceyear','regionCode'],axis = 1)


##check data
np.any(np.isnan(datax))
np.all(np.isfinite(datax))


### for separating categorical 
#for col in categorical_features:
#    dummies = pd.get_dummies(datax[col], prefix=col)
#    datax = pd.concat([datax, dummies], axis=1)
#    datax.drop(col, axis=1, inplace=True)
#datax.head()

datax = datax.drop([ 'lastaccesscount'],axis = 1)

mms = MinMaxScaler()
mms.fit(datax)
data_transformed = mms.transform(datax)
data_transformed = pd.DataFrame(data_transformed, index = datax.index, columns= datax.columns)  

###['Billed', 'orgid', 'asset_modified', 'lastaccesscount', 'sinceyear','inVOL', 'frequency(3mth)', 'Recency(days)', 'regionCode']
data_cluster = data_transformed.drop(['Billed', 'orgid','sinceyear', 
                                      'inVOL', 'regionCode'],axis = 1)

### for separating categorical 
#data_transformed.columns = ['Billed', 'sinceyear', 'inVOL',
#                            'total_usersession_1','last6month','regionCode_1.0', 'regionCode_2.0',
#                            'regionCode_3.0', 'regionCode_4.0','regionCode_5.0', 'regionCode_6.0',
#                            'regionCode_7.0']

#data_transformed.columns = [ 'lastaccesscount_day',
                           # 'total_usersession_1']

cor = data_cluster.corr()
sns.heatmap(cor, annot=True)
sns.pairplot(data_cluster)


#data_transformed_drop= data_transformed.drop(['total_usersession_1'], axis=1)
#cor = data_transformed_drop.corr()

pd.scatter_matrix(data_cluster, figsize=(6, 6))
plt.show()

#### K means clustering

Sum_of_squared_distances = []
K = range(1,15)
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(data_cluster)
    Sum_of_squared_distances.append(km.inertia_)
    

plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.show()



kmeans = KMeans(n_clusters=2)
kmodel = kmeans.fit(data_cluster)
labels = kmeans.predict(data_cluster)

centroids = kmeans.cluster_centers_

fig = plt.figure(figsize=(5, 5))

colors = map(lambda x: colmap[x+1], labels)

pdf = matplotlib.backends.backend_pdf.PdfPages("kmeans.pdf")


for i in range(0,len(data_cluster.columns)):
    for j in range(i+1, len(data_cluster.columns)):
        fig = plt.figure() 
        plt.scatter(data_cluster[data_cluster.columns[i]], data_cluster[data_cluster.columns[j]], c=labels,cmap = 'rainbow')
        plt.title("K Means")
        plt.xlabel(data_cluster.columns[i])
        plt.ylabel(data_cluster.columns[j])
        plt.show()
        pdf.savefig(fig)
pdf.close()

data_combined = datax

data_combined['label'] = labels

## combine all the datasets
data_label = pd.concat([data_combined,dataBilled,dataNVOL,dataVE])
data_label.to_csv("dataWithLabel.csv", sep=',', encoding='utf-8')


data_label['label'].value_counts()

ave = data_label.groupby('label').mean()

### resample
##L = ["NVOL","BILLED","VE","0","1"]

##NVOL_count = len(np.where(data_label['label']=="NVOL")[0])
###billed_upsampled = np.random.choice(dataBilled, size=NVOL_count, replace=True)

from sklearn.utils import resample

data_label_majority = data_label[data_label.label == 0]
sample_count = len(data_label_majority)
#sample_count = sample_count/3

data_label_minority_1 = data_label[data_label.label==1]
data_label_minority_2 = data_label[data_label.label==2]
data_label_minority_3 = data_label[data_label.label==3]
data_label_minority_4 = data_label[data_label.label==4]

# Upsample minority class

data_label_minority_1_up = resample(data_label_minority_1, 
                                 replace=True,     # sample with replacement
                                 n_samples=int(sample_count),    # to match majority class
                                 random_state=random_state) # reproducible results
 

data_label_minority_2_up = resample(data_label_minority_2, 
                                 replace=True,     # sample with replacement
                                 n_samples=int(sample_count),    # to match majority class
                                 random_state=random_state) # reproducible results
 

data_label_minority_3_up = resample(data_label_minority_3, 
                                 replace=True,     # sample with replacement
                                 n_samples=int(sample_count),    # to match majority class
                                 random_state=random_state) # reproducible results
 

data_label_minority_4_up = resample(data_label_minority_4, 
                                 replace=True,     # sample with replacement
                                 n_samples=int(sample_count),    # to match majority class
                                 random_state=random_state) # reproducible results
 
 
# Combine majority class with upsampled minority class
data_label = pd.concat([data_label_majority, data_label_minority_1_up,data_label_minority_2_up,data_label_minority_3_up,data_label_minority_4_up])
 

data_Y = data_label['label']
##[ 'asset_modified','frequency(3mth)', 'Recency(days)', 'sinceyear','inVOL','Billed']
data_X = data_label.drop(['orgid', 'lastaccesscount', 'sinceyear',
                           'label'],
                        axis = 1)
# Display new class counts
data_label.label.value_counts()

#############################################################################################
#############################################################################################
##################### smote ##################################################################
#############################################################################################
#############################################################################################

from imblearn.over_sampling import SMOTE

data_Y = data_label['label']
##[ 'asset_modified','frequency(3mth)', 'Recency(days)', 'sinceyear','inVOL','Billed']
data_X = data_label.drop(['orgid', 'lastaccesscount', 'sinceyear',
                           'label'],
                        axis = 1)


sm = SMOTE(random_state=random_state)
X_res, y_res = sm.fit_sample(data_X, data_Y)

data_X =pd.DataFrame(X_res,columns=data_X.columns)
data_Y =pd.Series(y_res)
data_Y.value_counts()

#############################################################################################
#############################################################################################
#############################################################################################


#############################################################################################
#############################################################################################
################### plot roc_auc function####################################################
#############################################################################################
#############################################################################################

def plot_roc_auc(predict,actual,name,classifiername):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(5):
        fpr[i], tpr[i], _ = roc_curve(actual[:, i], predict[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(actual.ravel(), predict.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    # Compute macro-average ROC curve and ROC area
    
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(5)]))
    
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(5):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    
    # Finally average it and compute AUC
    mean_tpr /= 5
    
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    # Plot all ROC curves
    fig = plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.5f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)
    
    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.5f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)
    
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue','purple','green'])
    for i, color in zip(range(5), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='ROC curve of class {0} (area = {1:0.5f})'
                 ''.format(i, roc_auc[i]))
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('AUC-ROC curve for '+ classifiername )
    plt.legend(loc="lower right")
    plt.show()
    fig.savefig(name)
#############################################################################################
#############################################################################################
#############################################################################################
    

#############################################################################################
#############################################################################################
################### auc_r2_rmse ####################################################
#############################################################################################
#############################################################################################
 
def auc_r2_rmse(y1,y1_train,y2,y2_test,name):
    
    print(name + " TRAIN")
    print("Accuracy:",metrics.accuracy_score(y1, y1_train))
    print('R2:', r2_score(y1, y1_train))  
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y1, y1_train)))  
    print('Log loss:', log_loss(y1, y1_train))  


    # Model Accuracy, how often is the classifier correct?
    print(name + " TEST")
    print("Accuracy:",metrics.accuracy_score(y2, y2_test))
    print('R2:', r2_score(y2, y2_test))  
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y2, y2_test)))  
    print('Log loss:', log_loss(y2, y2_test))  

#############################################################################################
#############################################################################################
#############################################################################################
 
    
#### models


#############################################################################################
#############################################################################################
###################### train-test data splitting ############################################
#############################################################################################
#############################################################################################
    
from sklearn.preprocessing import label_binarize

def train_test_split_for_oneclass (data_X, data_Y):
    y2 = label_binarize(data_Y, classes = [0,1,2,3,4])
    return train_test_split(data_X, y2, test_size=0.3,random_state =random_state)
    
    
def train_test_split_for_models (data_X,data_Y):
    return train_test_split(data_X, data_Y, test_size=0.3,random_state =random_state)
    
plt.hist(y_test)
plt.hist(y_train)

#############################################################################################
#############################################################################################
#X_train, X_test, y_train, y_test = train_test_split(data_X, data_Y, test_size=0.3,random_state =100)

##linear model
from sklearn import  linear_model
lm = linear_model.LinearRegression()
model = lm.fit(X_train, y_train)
predictions = lm.predict(X_test)
print("Accuracy:",model.score(X_test, y_test))


#############################################################################################
#############################################################################################
########################## Feature Importance ###############################
#############################################################################################
#############################################################################################
from sklearn.ensemble import RandomForestClassifier
X_train, X_test, y_train, y_test = train_test_split_for_models(data_X,data_Y)
clf=RandomForestClassifier(n_estimators=100)
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
errors = abs(y_pred - y_test)

# Calculate mean absolute percentage error (MAPE)
mape = np.mean(100 * (errors / y_test))


##Feature importance
feature_imp = pd.Series(clf.feature_importances_,index=X_train.columns).sort_values(ascending=False)
feature_imp
import matplotlib.pyplot as plt
# Creating a bar plot
sns.barplot(x=feature_imp, y=feature_imp.index)
# Add labels to your graph
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.legend()
plt.show()

#############################################################################################
#############################################################################################


#############################################################################################
#############################################################################################
##########################grid search cv random forest###############################
#############################################################################################
#############################################################################################
X_train, X_test, y_train, y_test =  train_test_split_for_oneclass(data_X,data_Y)

rfc = RandomForestClassifier(n_jobs=-1,max_features= 'sqrt' ,n_estimators=500, random_state = 1) 

param_grid = { 
    'n_estimators': [ 400 ,500, 600, 700],
    'max_features': ['sqrt','log2'],
    'max_depth': [ 10 ],
    'min_samples_split': [100], 
    'min_samples_leaf':[2],
    'random_state': [random_state]
}

CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 10,scoring='roc_auc')
print(datetime.datetime.now())
CV_rfc.fit(X_train,y_train)

import gc; gc.collect()
print(datetime.datetime.now())


print(CV_rfc.best_params_)
y_pred=CV_rfc.predict(X_test)
y_pred_train = CV_rfc.predict(X_train)

auc_r2_rmse(y_train,y_pred_train,y_test,y_pred,"Random Forest")

classifier = OneVsRestClassifier(CV_rfc.best_estimator_)
y_score = classifier.fit(X_train, y_train).predict_proba(X_test)
y_score = classifier.predict_proba(X_test)

y_score_train = classifier.predict_proba(X_train)
plot_roc_auc(y_score,y_test,'random_forest_auc_roc.png', 'Random Forest (test)')
plot_roc_auc(y_score_train,y_train,'random_forest_train_auc_roc.png', 'Random Forest (train)')

#############################################################################################
#############################################################################################


#############################################################################################
#############################################################################################
##########################plot randomforest decision###############################
#############################################################################################
#############################################################################################
X_train, X_test, y_train, y_test =  train_test_split_for_models(data_X,data_Y)

from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  
from IPython.display import Image  
import pydot 
dot_data = StringIO()

rfc = RandomForestClassifier(n_jobs=-1,max_features= 'sqrt',min_samples_leaf=2,min_samples_split=100 ,max_depth = 10,n_estimators=600, random_state = random_state) 
rfc.fit(X_train,y_train)

export_graphviz(rfc.estimators_[1], out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,feature_names = X_train.columns, class_names = list(map(str,np.unique(y_test))))
graph = pydot.graph_from_dot_data(dot_data.getvalue()) 
graph[0].write_png("randomforesttree.png")
Image(graph[0].create_png())
#############################################################################################
#############################################################################################


#############################################################################################
#############################################################################################
############# decision tree ##################################################################
#############################################################################################
#############################################################################################
from sklearn.tree import tree

X_train, X_test, y_train, y_test =  train_test_split_for_oneclass(data_X,data_Y)

#making the instance
model_DT= tree.DecisionTreeClassifier(criterion='gini')
#Hyper Parameters Set
params = {'max_features': ['sqrt', 'log2'],
          'max_depth': [ 2, 3, 4, 5, 10 ,20],
          'min_samples_split': [2,3,4,5,10,50,100,200], 
          'min_samples_leaf':[2,3,4,5,10,100],
          'random_state': [random_state]}

#Making models with hyper parameters sets
model_DT = GridSearchCV(model_DT, param_grid=params, n_jobs=-1,cv= 10,scoring = 'roc_auc')
#Learning
model_DT.fit(X_train,y_train)
print("Best Hyper Parameters:",model_DT.best_params_)
y_pred=model_DT.predict(X_test)
y_pred_train = model_DT.predict(X_train)

auc_r2_rmse(y_train,y_pred_train,y_test,y_pred,"Decision Tree")

classifier_DT = OneVsRestClassifier(model_DT.best_estimator_)
y_score = classifier_DT.fit(X_train, y_train).predict_proba(X_test)
y_score = classifier_DT.predict_proba(X_test)

y_score_train = classifier_DT.predict_proba(X_train)
plot_roc_auc(y_score,y_test,'decision_tree_auc_roc.png', 'Decision Tree (test)')
plot_roc_auc(y_score_train,y_train,'decision_tree_train_auc_roc.png', 'Decision Tree (train)')

#############################################################################################
#############################################################################################


#############################################################################################
#############################################################################################
##########################plot decision tree###############################
#############################################################################################
#############################################################################################
X_train, X_test, y_train, y_test =  train_test_split_for_models(data_X,data_Y)

from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  
from IPython.display import Image  
import pydot 
dot_data = StringIO()

export_graphviz(model_DT.best_estimator_, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,feature_names = X_train.columns, class_names = list(map(str,np.unique(y_test))))
graph = pydot.graph_from_dot_data(dot_data.getvalue()) 
graph[0].write_png("decision tree1.png")
Image(graph[0].create_png())
#############################################################################################
#############################################################################################



#############################################################################################
#############################################################################################
############# SVM ###########################################################################
#############################################################################################
#############################################################################################
from sklearn import svm

mms = MinMaxScaler()
mms.fit(data_X)
data_normal = mms.transform(data_X)
data_normal = pd.DataFrame(data_normal, index = data_X.index, columns= data_X.columns)  

#X_train, X_test, y_train, y_test =  train_test_split_for_oneclass(data_X,data_Y)
X_train, X_test, y_train, y_test = train_test_split_for_models(data_normal, data_Y)

#making the instance
modelsvm=svm.SVC()
#Hyper Parameters Set
params = {'C': list(range(1,100)), 
          'kernel': ['rbf'],
          'gamma': [0.1, 0.01, 0.001, 0.0001, 0.00001]}

#Making models with hyper parameters sets
modelsvm = GridSearchCV(modelsvm, param_grid=params, n_jobs=-1, cv=10)
#Learning
modelsvm.fit(X_train,y_train)
#The best hyper parameters set
print("Best Hyper Parameters:\n",modelsvm.best_params_)
y_pred=modelsvm.predict(X_test)
y_pred_train = modelsvm.predict(X_train)

y_train1 = label_binarize(y_train, classes = [0,1,2,3,4])
y_pred_train1 = label_binarize(y_pred_train, classes = [0,1,2,3,4])
y_pred1 = label_binarize(y_pred, classes = [0,1,2,3,4])
y_test1 = label_binarize(y_test, classes = [0,1,2,3,4])

auc_r2_rmse(y_train1,y_pred_train1,y_test1,y_pred1,"svm")

classifier_svm = OneVsRestClassifier(modelsvm.best_estimator_)
y_score = classifier_svm.fit(X_train, y_train1).decision_function(X_test)
y_score_train = classifier_svm.decision_function(X_train)
plot_roc_auc(y_score,y_test1,'svm_auc_roc.png', 'SVM (test)')
plot_roc_auc(y_score_train,y_train1,'svm_train_auc_roc.png', 'SVM (train)')

#############################################################################################
#############################################################################################


#############################################################################################
#############################################################################################
############# NN - 1 hidden layer ####################################################################
#############################################################################################
#############################################################################################
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
import keras
import datetime
mms = MinMaxScaler()
mms.fit(data_X)
data_normal = mms.transform(data_X)
data_normal = pd.DataFrame(data_normal, index = data_X.index, columns= data_X.columns)  

X_train, X_test, y_train, y_test = train_test_split_for_models(data_normal, data_Y)


# define baseline model
def baseline_model1(neurons = 30):
	# create model
    model = Sequential()
    model.add(Dense(neurons, input_dim=6, activation='relu'))
    model.add(Dense(5, activation='sigmoid'))
	# Compile model
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

estimator1 = KerasClassifier(build_fn=baseline_model1, epochs=10, batch_size=5, verbose=0)

neurons = [1, 5, 10, 15, 20, 25, 30]
param_grid = dict(neurons=neurons)

grid1 = GridSearchCV(estimator=estimator1, param_grid=param_grid, cv=10)
print(datetime.datetime.now())
grid_result1 = grid1.fit(X_train, y_train)
import gc; gc.collect()
print(datetime.datetime.now())

print("Best Hyper Parameters:\n",grid_result1.best_params_)
y_pred=grid_result1.predict(X_test)
y_pred_train = grid_result1.predict(X_train)

y_train1 = label_binarize(y_train, classes = [0,1,2,3,4])
y_pred_train1 = label_binarize(y_pred_train, classes = [0,1,2,3,4])
y_pred1 = label_binarize(y_pred, classes = [0,1,2,3,4])
y_test1 = label_binarize(y_test, classes = [0,1,2,3,4])

auc_r2_rmse(y_train1,y_pred_train1,y_test1,y_pred1,"NN 1 layer")

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_y_1 = LabelEncoder()
y_train3 = labelencoder_y_1.fit_transform(y_train)

classifier_NN1 = OneVsRestClassifier(KerasClassifier(build_fn=baseline_model1, epochs=10, batch_size=5, verbose=0))
classifier_NN1.fit(X_train, y_train3)
y_score = classifier_NN1.predict_proba(X_test)
y_score_train = classifier_NN1.predict_proba(X_train)

plot_roc_auc(y_score,y_test1,'nn1_auc_roc.png', 'SLP (test)')
plot_roc_auc(y_score_train,y_train1,'nn1_train_auc_roc.png', 'SLP (train)')


from ann_visualizer.visualize import ann_viz;
ann_viz(model, title="My first neural network")


#############################################################################################
#############################################################################################



#############################################################################################
#############################################################################################
############# NN - keras ####################################################################
#############################################################################################
#############################################################################################
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
import keras
import datetime
mms = MinMaxScaler()
mms.fit(data_X)
data_normal = mms.transform(data_X)
data_normal = pd.DataFrame(data_normal, index = data_X.index, columns= data_X.columns)  

X_train, X_test, y_train, y_test = train_test_split_for_models(data_normal, data_Y)

from keras.layers import Dropout

# define baseline model
def baseline_model():
	# create model
    model = Sequential()
    model.add(Dense(8, input_dim=6, activation='relu'))
    model.add(Dense(7, activation='relu'))
    #model.add(Dropout(dropout_rate))
    model.add(Dense(5, activation='softmax'))
	# Compile model
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

estimator = KerasClassifier(build_fn=baseline_model, epochs=10, batch_size=5, verbose=0)

batch_size = [32, 64, 128]
epochs = [10, 50, 100]
#dropout_rate = [0.0, 0.1, 0.2]

param_grid = dict(batch_size=batch_size, epochs=epochs)

grid = GridSearchCV(estimator=estimator, param_grid=param_grid, cv=10)
print(datetime.datetime.now())
grid_result = grid.fit(X_train, y_train)
import gc; gc.collect()
print(datetime.datetime.now())

print("Best Hyper Parameters:\n",grid_result.best_params_)
y_pred=grid_result.predict(X_test)
y_pred_train = grid_result.predict(X_train)

y_train1 = label_binarize(y_train, classes = [0,1,2,3,4])
y_pred_train1 = label_binarize(y_pred_train, classes = [0,1,2,3,4])
y_pred1 = label_binarize(y_pred, classes = [0,1,2,3,4])
y_test1 = label_binarize(y_test, classes = [0,1,2,3,4])

auc_r2_rmse(y_train1,y_pred_train1,y_test1,y_pred1,"MLP")

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_y_1 = LabelEncoder()
y_train3 = labelencoder_y_1.fit_transform(y_train)

classifier = OneVsRestClassifier(KerasClassifier(build_fn=baseline_model, epochs=100, batch_size=32, verbose=0))
classifier.fit(X_train, y_train3)
y_score = classifier.predict_proba(X_test)
y_score_train = classifier.predict_proba(X_train)

plot_roc_auc(y_score,y_test1,'mlp_auc_roc.png', 'MLP (test)')
plot_roc_auc(y_score_train,y_train1,'mlp_train_auc_roc.png', 'MLP (train)')


from ann_visualizer.visualize import ann_viz;
ann_viz(model, title="My first neural network")


#############################################################################################
#############################################################################################


#############################################################################################
#############################################################################################
############# K Nearest neighbours ##########################################################
#############################################################################################
#############################################################################################
from sklearn.neighbors import KNeighborsClassifier  

mms = MinMaxScaler()
mms.fit(data_X)
data_normal = mms.transform(data_X)
data_normal = pd.DataFrame(data_normal, index = data_X.index, columns= data_X.columns)  

X_train, X_test, y_train, y_test = train_test_split_for_oneclass(data_normal, data_Y)

params = {'n_neighbors': list(range(1,100))}

modelknn = KNeighborsClassifier(n_neighbors=5)  
modelknn = GridSearchCV(modelknn, param_grid=params, n_jobs=-1, cv=10)
modelknn.fit(X_train, y_train)  
print("Best Hyper Parameters:\n",modelknn.best_params_)

y_pred = modelknn.predict(X_test)  
y_pred_train = modelknn.predict(X_train)
auc_r2_rmse(y_train,y_pred_train,y_test,y_pred,"KNN")

classifier_nn = OneVsRestClassifier(modelknn.best_estimator_)
y_score = classifier_nn.fit(X_train, y_train1).predict_proba(X_test)
y_score_train = classifier_nn.predict_proba(X_train)
plot_roc_auc(y_score,y_test,'knn_auc_roc.png', 'KNN (test)')
plot_roc_auc(y_score_train,y_train,'knn_train_auc_roc.png', 'KNN (train)')

#############################################################################################
#############################################################################################





################### Neural Net
from sklearn.neural_network import MLPClassifier

model_Mlp = MLPClassifier(solver='sgd', alpha=1e-5,hidden_layer_sizes=(7), random_state=random_state)

parameters = {'solver': ['sgd'],
              'activation' : ['logistic', 'tanh', 'relu'],
              'max_iter': [200,300,500,1000,1500,2000,2500], 
              'learning_rate': ['constant','invscaling']}
model_Mlp = GridSearchCV(model_Mlp, param_grid=parameters, n_jobs=-1)
model_Mlp.fit(data_Xtrain_transformed,y_train)
#The best hyper parameters set
print("Best Hyper Parameters:\n",model_Mlp.best_params_)
y_pred=model_Mlp.predict(data_Xtest_transformed)
y_pred_train = model_Mlp.predict(X_train)

print("NN TRAIN")
print("Accuracy:",metrics.accuracy_score(y_train, y_pred_train))
print('R2:', r2_score(y_train, y_pred_train))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_train, y_pred_train)))  

# Model Accuracy, how often is the classifier correct?
print("NN TEST")
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print('R2:', r2_score(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))  

#########################################
#########################################
#########################################
##################################################################################
#########################################
#########################################
### test data
filename_test = 'test.xlsx'
sheetname_test = 'test_data'

xl = pd.ExcelFile(filename_test)
data_test = xl.parse(sheetname_test)
data_test.head()

data_X_test = data_test.drop(['orgid', 'lastaccesscount', 'sinceyear'
                           ],
                        axis = 1)


y_pred1=model_DT.predict(data_X_test)

y_pred1_df = pd.DataFrame(y_pred1, columns= ['0','1','2','3','4'])  

data_test_combined = pd.concat([data_X_test.reset_index(drop=True), y_pred1_df], axis=1)

data_test_combined.to_csv("dataTestWithLabel.csv", sep=',', encoding='utf-8')

##[ 'asset_modified','frequency(3mth)', 'Recency(days)' ,'inVOL','Billed']

data_test_X = data_test.drop([ 'orgid', 'lastaccesscount','sinceyear'],
                        axis = 1)

test_y_pred= rfc.predict(data_test_X)

data_test_combined = data_test
data_test_combined['label'] = test_y_pred

data_test_combined.to_csv("dataTestWithLabel.csv", sep=',', encoding='utf-8')


################ R part

from sklearn import tree
model = tree.DecisionTreeClassifier(criterion='gini') # for classification, here you can change the algorithm as gini or entropy (information gain) by default it is gini  
model.fit(X_train,y_train)
model.score(X_train,y_train)


##plt.scatter(data_transformed['Billed'], data_transformed['last6month'], c=labels, alpha=0.75, edgecolor='k')
#for idx, centroid in enumerate(centroids):
#    plt.scatter(*centroid, c=idx)
    
#plt.xlim(0, 80)
#plt.ylim(0, 80)
#plt.show()


### PCA

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(data_transformed)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])

kmeans = KMeans(n_clusters=3)
kmodel = kmeans.fit(principalDf)
labels = kmeans.predict(principalDf)
centroids = kmeans.cluster_centers_

fig = plt.figure(figsize=(5, 5))

colors = map(lambda x: colmap[x+1], labels)

plt.scatter(principalDf['principal component 1'], principalDf['principal component 2'], c=labels, cmap = 'rainbow')

### combine label with dataframe
data_combined = datax
data_combined['label'] = labels

##### dendogram
from scipy.cluster.hierarchy import dendrogram, linkage  

pdf = matplotlib.backends.backend_pdf.PdfPages("dendogram.pdf")

for i in range(0,len(data_cluster.columns)):
    for j in range(i+1, len(data_cluster.columns)):
        fig = plt.figure() 
        den_data = data_cluster.iloc[:, [j,i]].values  

        dend = dendrogram(linkage(den_data, method='ward')) 
        plt.title("Dendogram")
        plt.xlabel(data_cluster.columns[j])
        plt.ylabel(data_cluster.columns[i])
        plt.show()
        pdf.savefig(fig)
pdf.close()


##### Hierarchical clustering

from sklearn.cluster import AgglomerativeClustering

cluster = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='complete')  
cluster.fit_predict(data_cluster) 

pdf = matplotlib.backends.backend_pdf.PdfPages("Hierarchical.pdf")

for i in range(0,len(data_cluster.columns)):
    for j in range(i+1, len(data_cluster.columns)):
        fig = plt.figure() 
        plt.scatter(data_cluster[data_cluster.columns[i]], data_cluster[data_cluster.columns[j]], c=cluster.labels_,cmap = 'rainbow')
        plt.title("Hierarchical clustering")
        plt.xlabel(data_cluster.columns[i])
        plt.ylabel(data_cluster.columns[j])
        plt.show()
        pdf.savefig(fig)
pdf.close()