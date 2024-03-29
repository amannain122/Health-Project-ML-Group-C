# -*- coding: utf-8 -*-
"""Machine_Learning_Assignment.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1TsoSqNhyA-TAq1FB1r_M_WXwLDHLa_mG
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

rawHealthCareDF = pd.read_csv('/content/HealthCareDataSet-Sheet1.csv',index_col=0)
X_train, X_test=train_test_split(rawHealthCareDF,test_size=0.2,random_state=24)
X_train.to_csv("healthCare_train_dataset")
X_test.to_csv("healthCare_test_dataset")

healthCareDF=pd.read_csv("/content/healthCare_train_dataset",index_col=0)

healthCareDF.head(5)

healthCareDF.tail(5)

healthCareDF.shape

healthCareDF.describe(include='all')

healthCareDF.count()

healthCareDF.isnull().sum()

healthCareDF.dropna(how='all', axis=1, inplace=True)

mean_RMCOunt=healthCareDF["ReadmissionCount"].mean().round()
mean_SecDiagno=healthCareDF["SecondaryDiagnosis"].mean().round()
healthCareDF["ReadmissionCount"].fillna(mean_RMCOunt,inplace=True)
healthCareDF["SecondaryDiagnosis"].fillna(mean_SecDiagno,inplace=True)

healthCareDF["SubstanceAbuseHistory"].fillna('Unknown',inplace=True)
healthCareDF.dropna(axis=0,inplace=True)
healthCareDF.isnull().sum()

healthCareDF.info()

numerical_data_hcare=healthCareDF[['LengthOfStay', 'ReadmissionCount', 'FacilityId', 'BMI', 'ABG', 'Pulse',
       'SecondaryDiagnosis']]
correl=numerical_data_hcare.corr(method="pearson")

sns.heatmap(correl,annot=True)

healthCareDF['Gender'].replace("M",0,inplace=True)
healthCareDF['Gender'].replace("F",1,inplace=True)
healthCareDF['PyschologicalAilments'].replace(False,0,inplace=True)
healthCareDF['PyschologicalAilments'].replace(True,1,inplace=True)
healthCareDF.SubstanceAbuseHistory.describe(include='all')
healthCareDF['SubstanceAbuseHistory'].unique()

numerical_data_hcare=healthCareDF[['LengthOfStay', 'ReadmissionCount', 'Gender', 'FacilityId',
       'PyschologicalAilments', 'BMI', 'ABG', 'Pulse',
       'SecondaryDiagnosis']]
correl=numerical_data_hcare.corr(method="pearson").round(2)

sns.heatmap(correl,annot=True)

encoder=OneHotEncoder()
categ_subs_abuse=encoder.fit_transform(healthCareDF[["SubstanceAbuseHistory"]])
newcateg= pd.DataFrame(categ_subs_abuse.toarray(), columns=encoder.get_feature_names_out(['SubstanceAbuseHistory']))
newcateg.index=healthCareDF.index
newcateg.head()

healthCareDF=healthCareDF.join(newcateg)

actual_correl=healthCareDF.corr(method='spearman').round(2)
plt.figure(figsize=(8,8))
sns.heatmap(actual_correl,annot=True,square=True)
plt.show()

def show_histogram(data_frame_cols):
  plt.figure(figsize=(6,6))
  sns.displot(data_frame_cols,bins='auto')
  plt.show()

for column in healthCareDF.columns:
  show_histogram(healthCareDF[column])

def show_scatterplot(data_frame_cols):
  plt.figure(figsize = (6,6))
  sns.scatterplot(data= healthCareDF, x= data_frame_cols, y = healthCareDF.LengthOfStay)
  plt.show()

#features
features = ['ReadmissionCount', 'Gender', 'ABG' ,'SecondaryDiagnosis', 'Pulse','BMI']
target_value = 'LengthOfStay'
for column in features:
  show_scatterplot(healthCareDF[column])

def show_scatterplot(data_frame_cols):
  plt.figure(figsize = (6,6))
  sns.scatterplot(data= healthCareDF, x= data_frame_cols, y = healthCareDF.LengthOfStay)
  plt.show()

features_data_frame = healthCareDF[features]
features_data_frame = features_data_frame.join(healthCareDF.LengthOfStay)
features_data_frame.to_csv('features_extracted.csv', index=False)

