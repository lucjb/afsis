import pandas as pd
import numpy as np
from sklearn import svm, cross_validation, preprocessing, metrics
from sklearn.cross_validation import KFold
from sklearn.ensemble import BaggingRegressor, GradientBoostingRegressor
from sklearn.grid_search import GridSearchCV
from sklearn import linear_model, neighbors
import sys
from sklearn.feature_selection import SelectFpr, f_regression, SelectKBest
import matplotlib.pyplot as pl
from sklearn import datasets, cluster
from sklearn import random_projection
from sklearn.gaussian_process import GaussianProcess
def score(a, b):
	n = b.shape[0] 
	return np.mean(np.sqrt(sum((a-b)**2)/n))

def scores(a, b):
	n = b.shape[0] 
	return np.sqrt(sum((a-b)**2)/n)

def mse(estimator, X, y):
	v = score(estimator.predict(X), y)
	print v	
	return v

train = pd.read_csv('training.csv')
test = pd.read_csv('sorted_test.csv')
print train.describe()
labels = train[['Ca','P','pH','SOC','Sand']].values


train.ix[train['Depth']=='Topsoil', 'Depth2']=1.
train.ix[train['Depth']=='Subsoil', 'Depth2']=0.
test.ix[test['Depth']=='Topsoil', 'Depth2']=1.
test.ix[test['Depth']=='Subsoil', 'Depth2']=0.
train.drop(['Ca', 'P', 'pH', 'SOC', 'Sand', 'PIDN', 'Depth'], axis=1, inplace=True)
test.drop(['PIDN', 'Depth'], axis=1, inplace=True)

co2wl=['m2379.76','m2377.83','m2375.9','m2373.97','m2372.04','m2370.11','m2368.18','m2366.26','m2364.33','m2362.4','m2360.47','m2358.54','m2356.61','m2354.68','m2352.76']
train.drop(co2wl, axis=1, inplace=True)
test.drop(co2wl, axis=1, inplace=True)
#train.drop('Depth2', axis=1, inplace=True)
#test.drop('Depth2', axis=1, inplace=True)

																										
xtrain, xtest = train.as_matrix(), test.as_matrix()

examples = xtrain.shape[0]
all = np.concatenate((xtrain, xtest))
all = preprocessing.scale(all)

#from sklearn.random_projection import johnson_lindenstrauss_min_dim
#k = johnson_lindenstrauss_min_dim(n_samples=examples, eps=0.999999)
#print "k: " + str(k)
#transformer = random_projection.GaussianRandomProjection(eps=0.999999)
#transformer = random_projection.SparseRandomProjection(eps=0.5)
#all = transformer.fit_transform(all)
#print all.shape


import scipy.signal
p = xtrain.shape[1]
for i in range(0, all.shape[0]):
 all[i][range(0,3578)] = scipy.signal.savgol_filter(all[i][range(0,3578)], 51, 2)
# pl.plot(all[i])
# pl.plot(smoothed)
# pl.show()

from sklearn.decomposition import PCA
pca = PCA(n_components=750)
all = pca.fit_transform(all)


#agglo = cluster.FeatureAgglomeration(n_clusters=1250)
#all = np.c_[all[:,3578:], agglo.fit_transform(all[:,:3578])]


xtrain,xtest = all[0:examples], all[examples:]



eps = 1e-4
tol=1e-6
v=0
models =[svm.SVR(C=512, gamma=2**-14, verbose=v, kernel="rbf", tol=tol, epsilon=eps), svm.SVR(C=1, gamma=2**-7, verbose=v, kernel="rbf", tol=tol, epsilon=eps), svm.SVR(C=64, gamma=2**-14, verbose=v, kernel="rbf", tol=tol, epsilon=eps), svm.SVR(C=256, gamma=2**-14, verbose=v, kernel="rbf", tol=tol, epsilon=eps), svm.SVR(C=32, gamma=2**-14, verbose=v, kernel="rbf", tol=tol, epsilon=eps)]


#for i in range(0,5):
#	models[i]=linear_model.RidgeCV()
#	models[i]=neighbors.KNeighborsRegressor(1, weights='uniform') 
#	models[i]= GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=1, random_state=0, loss='ls')
#	models[i]= GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=1, random_state=0, loss='ls')
#	models[i] = GaussianProcess()

fs=[]
#or i in range(0,2):	
#	a=range(0,xtrain.shape[1])	
#	fs.append(a)

for i in range(0,5):	
	selector = SelectKBest(f_regression,10)
	a=selector.fit(xtrain, labels[:,i]).get_support()	
	a=range(0,xtrain.shape[1])	
	fs.append(a)
	print len(selector.get_support(True))



grid=[{'C':[1,2,4,8,16,32,64,128,256,512,1024],'gamma':[2**i for i in range(-14,3,1) ]}]
#grid=[{'C':[4,8,16,32,64,128]}]
#grid=[{'base_estimator':[svm.SVR(C=2**v, gamma=2**-14) for v in range(2,8)]}]
'''
for i in range(1,2):
	ensemble = BaggingRegressor(base_estimator=svm.SVR(C=2**v, gamma=2**-14),max_samples=0.8, max_features=1.0)
	clf = GridSearchCV(svm.SVR(verbose=0), grid, cv=10, loss_func=score)
	clf.fit(xtrain, labels[:,i])
	print "=========" + str(i) + "================"	
	print(clf.best_estimator_)
	models[i]=clf.best_estimator_
	for params, mean_score, gscores in clf.grid_scores_:
		print("%0.3f (+/-%0.03f) for %r" % (mean_score, gscores.std() / 2, params))
	print "---------------------------"
'''



folds = 10
kf = KFold(xtrain.shape[0], folds)
fold_scores = np.zeros((folds, 5))
mean_scores = []
k=0


for train, test in kf:
    n = xtrain[test].shape[0]
    cv_preds = np.zeros((n, 5))
    for i in range(5):
        sup_vec = models[i]
        ensemble = sup_vec#BaggingRegressor(base_estimator=sup_vec, max_samples=0.7, max_features=0.7, n_estimators=10)
        fs_train = xtrain[train][:,fs[i]]
	fs_test = xtrain[test][:,fs[i]]
        ensemble.fit(fs_train, labels[train][:,i])
	cv_preds[:,i]=ensemble.predict(fs_test).astype(float)
    fold_scores[k]=scores(labels[test], cv_preds)	    
    mean_score = np.mean(fold_scores[k])
    mean_scores.append(mean_score) 
    print fold_scores[k]
    print mean_score 
    k+=1
print fold_scores.T
print "Cross Validation score: "
print np.mean(fold_scores, axis=0)
print np.std(fold_scores, axis=0)
print np.mean(mean_scores)

n = xtrain.shape[0]
preds = np.zeros((xtest.shape[0], 5))
train_preds = np.zeros((n, 5))
for i in range(5):
    sup_vec = models[i]
    ensemble = sup_vec#BaggingRegressor(base_estimator=sup_vec, max_samples=0.7, max_features=0.7, n_estimators=10)
    fs_train = xtrain[:,fs[i]]
    fs_test = xtest[:,fs[i]]
    ensemble.fit(fs_train, labels[:,i])
    preds[:,i] = ensemble.predict(fs_test).astype(float)
    train_preds[:,i]=ensemble.predict(fs_train).astype(float)

print "Training score: " + str(score(train_preds, labels)) 

sample = pd.read_csv('sample_submission.csv')
sample['Ca'] = preds[:,0]
sample['P'] = preds[:,1]
sample['pH'] = preds[:,2]
sample['SOC'] = preds[:,3]
sample['Sand'] = preds[:,4]

sample.to_csv('cronopio_submission.csv', index = False)

