import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

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

x_train = arr_train[:,1:]
x_test = arr_test[:,1:]

regressor = LogisticRegression(max_iter=392)
regressor.fit(x_train, y_train)

y_test_predictions = regressor.predict(x_train)
y_train_predictions = regressor.predict(x_test)

y_train_predictions = [round(output) for output in regressor.predict(x_train)]
y_test_predictions = [round(output) for output in regressor.predict(x_test)]

def get_accuracy(predictions, actual):
    correct = ['' for i in range(len(predictions)) if predictions[i] == actual[i]]
    return len(correct) / len(predictions)

# print
print("\nfeatures:", features_to_use)
print("\ntraining accuracy", round(get_accuracy(y_train_predictions, y_train), 4))
print("testing accuracy", round(get_accuracy(y_test_predictions, y_test), 4))

columns_featured = df_train.columns[1:]
coefficients_featured = regressor.coef_[0]
constant_dict = {'Constant':regressor.intercept_[0]}
coefficients = {columns_featured[n]:coefficients_featured[n] for n in range(len(columns_featured))}
constant_dict.update(coefficients)

print("\ncoefficients:", {key:round(value, 4) for key, value in constant_dict.items()})
# for key, value in {key:round(value, 4) for key, value in constant_dict.items()}.items():
#     print(key, value)

"""

'Constant': 1.894,
'Sex': 2.5874,
'Pclass': -0.6511,
'Fare': -0.0001,
'Age': -0.0398,
'SibSp': -0.545,
'SibSp>0': 0.4958,
'Parch>0': 0.0499,
'Embarked=C': -0.2078,
'Embarked=None': 0.0867,
'Embarked=Q': 0.479,
'Embarked=S': -0.3519,
'CabinType=A': -0.0498,
'CabinType=B': 0.0732,
'CabinType=C': -0.2125,
'CabinType=D': 0.7214,
'CabinType=E': 0.4258,
'CabinType=F': 0.6531,
'CabinType=G': -0.7694,
'CabinType=None': -0.5863,
'CabinType=T': -0.2496

"""