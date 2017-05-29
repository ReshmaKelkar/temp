import pandas as pd
import numpy as np
import xgboost as xgb
import csv
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import log_loss
from scipy.sparse import hstack, vstack
from scipy.sparse import *

print('Load data...')
train = pd.read_csv("input/train.csv")
target = train['target']
train = train.drop(['ID','target'],axis=1) 
test = pd.read_csv("input/test.csv")
ids = test['ID'].values
test = test.drop(['ID'],axis=1)


### Check equality...

print 'Checking equality...'

for (train_name, train_series), (test_name, test_series) in zip(train.iteritems(),test.iteritems()):
	#~ print "processing", train_name
	if train_series.dtype == 'O':
		f1=np.sort(train_series.unique())
		f2=np.sort(test_series.unique())	
		if (np.array_equal(f1,f2)):
			test[test_name]=test_series
		else:
			missing_from_1 = set(f2)-set(f1)
			missing_from_2 = set(f1)-set(f2)
			if (len(missing_from_1) !=0):
				print "missing from test", len(missing_from_1),  train_name
				test_series_mat=test_series.as_matrix()
				for val in missing_from_1:
					test_series_mat[np.where(test_series_mat == val)]= np.nan
					test[test_name]=test_series_mat
			else:
				test[test_name]=test_series
			if (len(missing_from_2) !=0):
				print "missing from train", len(missing_from_2),  train_name
				train_series_mat=train_series.as_matrix()
				for val1 in missing_from_2:
					train_series_mat[np.where(train_series_mat == val1)]= np.nan
					train[train_name]=train_series_mat
			else:
				train[train_name]=train_series


print "Done with equality check..."
num_train = len(train)
all_data = pd.concat((train, test), axis=0, ignore_index=True)

feature_enc=all_data.select_dtypes(include=[object]) ## Only cat columns
feature_notenc=all_data.select_dtypes(exclude=[object]) ## Non cat colums

#~ header=[feature_enc.columns+feature_notenc.columns]

print "Fresh Features...."
print "shape of cat data",feature_enc.shape
print "shape of num data",feature_notenc.shape

print('Clearing...')
lbl = preprocessing.LabelEncoder()

#~ nan_enc={}
nan_cnt={}

for (fdata_name, fdata_series) in feature_enc.iteritems():
	na_cnt=fdata_series.isnull().sum()
	nan_cnt[fdata_name]= na_cnt
	fdata_series.fillna(-100,inplace=True)
	feature_enc[fdata_name], tmp_indexer = pd.factorize(feature_enc[fdata_name])

## Instead of deleting feature don't encode them 
for (data_name, data_series) in feature_notenc.iteritems():
	na_cnt=data_series.isnull().sum()
	nan_cnt[data_name]= na_cnt
	data_series.fillna(-100,inplace=True)
            
print "Process feature....."
print "shape of cat data",feature_enc.shape
print "shape of num data",feature_notenc.shape

all_data_concat=pd.concat((feature_enc, feature_notenc), axis=1)
header=all_data_concat.columns.values

print "processing correlations cat data...."

all_arr=[]
for (fdata_name1, fdata_series1) in all_data_concat.iteritems():
	fill_arr=[]
	fill_arr.append(fdata_name1)
	na_ratio=float(nan_cnt[fdata_name1])/float(all_data_concat.shape[0])
	#~ if (na_ratio<=0):
		#~ nc_arr=['NC'] *all_data_concat.shape[1]
		#~ fill_arr.append(nc_arr)
	#~ else:
	for (fdata_name2, fdata_series2) in all_data_concat.iteritems():
		corr=np.corrcoef(fdata_series1, fdata_series2)
		fill_arr.append(corr[0][1])
	all_arr.append(fill_arr)
	
all_arr_flat=np.array(all_arr)
np.savetxt('corr_mat_noNC.csv', all_arr_flat, delimiter=',', fmt ='%s')
#~ submit_flat=np.vstack(np.array(all_arr))	
