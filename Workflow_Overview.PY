import pandas as pd
import numpy as np
import matplotlib as plt
import scipy as sc
from sklearn import datasets, linear_model
from sklearn.datasets import load_diabetes, fetch_openml
from sklearn.pipeline import make_pipeline
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder

#Split train and test data
store=pd.read_csv("Store.csv",index_col=0,na_values=["??","nan"])
store1=store.copy(deep=True)

#See data
store.head()
desc=store.describe(include='all')
store.info()


#Handle Missing Values:
    # Imputation(Filling Data):
        # SimpleImputer - Use mean/mode or other statistic to fill data
        # IterativeImputer - Intelligent Imputation from complete data
        # KNNImputer - Filling values from nearest datapoints

store._get_numeric_data().columns
# store._get_categorical_data().columns

# Find Correlation and Map Individually
store1.drop(columns=["Assortment","StoreType","PromoInterval"],inplace=True)
corret=store1.corr(method='pearson').round(2)
sns.heatmap(data=corret)

# Perform OHE on test and training data individually
storecat=store[["Assortment","StoreType","PromoInterval"]]
encoder=OneHotEncoder()
new_ohdf=encoder.fit_transform(sparse=False)
new_ohdf.index=storecat.index

# Perform Correlation and select features

#CONSIDER REMOVING:
    # Features with same values majority of time
    # Features with large no of missing values
    #Categotical Data with categories not occuring significantly
    #Feature Having high correlation with other feature
    #Feature having low correlation with target feature
    #Feature having high co relation if visually do not correlate

# Train Model
# Fill Test Data with train data statistic
# Test Model




    
