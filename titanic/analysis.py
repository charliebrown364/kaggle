import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import sys

# Read in the data
df = pd.read_csv('/home/runner/kaggle/titanic/dataset.csv')

# Filter to only the column's we're interested in
keep_cols = ['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin', 'Embarked']
df = df[keep_cols]

# Process The Columns:

# Sex
def convert_sex_to_int(sex):
    if sex == 'male':
        return 0
    elif sex == 'female':
        return 1

df['Sex'] = df['Sex'].apply(convert_sex_to_int)

# Age

age_nan = df['Age'].apply(lambda entry: np.isnan(entry))
age_not_nan = df['Age'].apply(lambda entry: not np.isnan(entry))
df['Age'][age_nan] = df['Age'][age_not_nan].mean()

# SibSp

def indicator_greater_than_zero(x):
    if x > 0:
        return 1
    else:
        return 0
  
df['SibSp>0'] = df['SibSp'].apply(indicator_greater_than_zero) 
df['Parch>0'] = df['Parch'].apply(indicator_greater_than_zero) 

# CabinType

df['Cabin'] = df['Cabin'].fillna('None')

def get_cabin_type(cabin):
    if cabin != 'None':
        return cabin[0]
    else:
        return cabin

df['CabinType'] = df['Cabin'].apply(get_cabin_type)

for cabin_type in df['CabinType'].unique():
    dummy_var_name = 'CabinType={}'.format(cabin_type)
    dummy_var_val = df['CabinType'].apply(lambda entry: int(entry == cabin_type)) 
    df[dummy_var_name] = dummy_var_val

# Embarked

df['Embarked'] = df['Embarked'].fillna('None')

for embarked in df['Embarked'].unique():
    dummy_var_name = 'Embarked={}'.format(embarked)
    dummy_var_val = df['Embarked'].apply(lambda entry: int(entry == embarked)) 
    df[dummy_var_name] = dummy_var_val

del df['Embarked']

# setting the dataframe with the right features

features_to_use = ['Sex', 'Pclass', 'Fare', 'Age', 'SibSp', 'SibSp>0', 'Parch>0', 'Embarked=C', 'Embarked=None', 'Embarked=Q', 'Embarked=S', 'CabinType=A', 'CabinType=B', 'CabinType=C', 'CabinType=D', 'CabinType=E', 'CabinType=F', 'CabinType=G', 'CabinType=None', 'CabinType=T']
columns_needed = ['Survived'] + features_to_use
df = df[columns_needed]

# split into traiding/testing dataframe

df_train = df[:500]
df_test = df[500:]

arr_train = np.array(df_train)
arr_test = np.array(df_test)

y_train = arr_train[:,0]
y_test = arr_test[:,0]

X_train = arr_train[:,1:]
X_test = arr_test[:,1:]

regressor = LinearRegression()
regressor.fit(X_train, y_train)

def convert_regressor_output_to_survival_value(output):
    if output < 0.5:
        return 0
    else:
        return 1

y_train_predictions = [convert_regressor_output_to_survival_value(output) for output in regressor.predict(X_train)]
y_test_predictions = [convert_regressor_output_to_survival_value(output) for output in regressor.predict(X_test)]


def get_accuracy(predictions, actual):
    
    num_correct = 0
    num_incorrect = 0
    
    for i in range(len(predictions)):
        if predictions[i] == actual[i]:
            num_correct += 1
        else:
            num_incorrect += 1
    
    return num_correct / (num_correct + num_incorrect)

# print
print("\nfeatures:", features_to_use)
print("\ntraining accuracy", round(get_accuracy(y_train_predictions, y_train), 4))
print("testing accuracy", round(get_accuracy(y_test_predictions, y_test), 4))
print("\ncoefficients:", [round(num, 4) for num in regressor.coef_])