import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score,mean_squared_error

import sklearn.metrics as metrics
import seaborn as sns

# Import tools needed for visualization
from sklearn.tree import export_graphviz
from sklearn.tree import DecisionTreeClassifier
import pydot
import pydotplus
from IPython.display import Image 

def item_user(file, normalize= True)
  df_x= df_dum.drop('FFH_88', axis = 1)
  df_FFH_y = np.array(df_dum['FFH_88'])

  # Saving feature names for later use
  feature_list = list(df_x.columns)

  X = data.iloc[:,3:14]  #independent columns
  y = data.iloc[:,-1]    #target column i.e price range

  #get correlations of each features in dataset
  corrmat = df.corr()
  top_corr_features = corrmat.index
  plt.figure(figsize=(20,20))
  #plot heat map
  g=sns.heatmap(df[top_corr_features].corr(),annot=True,cmap="RdYlGn",linewidths=0.1,annot_kws={"size": 20})
  
  return corrmat



def data_split(path, save=True)

  # Split the data into training and testing sets
  train_data, test_data, train_FFH_88, test_FFH_88 = train_test_split(df_x, df_FFH_y, test_size = 0.2, random_state = 42)

  #test_FFE_88.describe()
  train_data.describe()
  
  returen train_FFH_88
  
  
# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 100, random_state = 42)

# Train the model on training data
rf.fit(train_data, train_FFH_88);

# Use the forest's predict method on the test data
predictions = rf.predict(test_data)

# Calculate the MSE
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(test_FFH_88, predictions)))


# Get numerical feature importances
importances = list(rf.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];





