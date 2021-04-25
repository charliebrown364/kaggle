import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

# Read in the data
df = pd.read_csv('/home/runner/kaggle/data&libraries/dataset.csv')

# Filter to only the column's we're interested in
keep_cols = ['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin', 'Embarked']
df = df[keep_cols]

# process the columns

def process_df(df):

    df['Sex'] = df['Sex'].apply(lambda sex: 0 if sex == 'male' else 1) # sex

    age_nan = df['Age'].apply(lambda entry: np.isnan(entry)) # age
    age_not_nan = df['Age'].apply(lambda entry: not np.isnan(entry))
    df.loc[age_nan, ['Age']] = df['Age'][age_not_nan].mean()
        
    df['SibSp>0'] = df['SibSp'].apply(lambda x: 1 if x > 0 else 0) # sib
    df['Parch>0'] = df['Parch'].apply(lambda x: 1 if x > 0 else 0) # parch

    df['CabinType'] = df['Cabin'].fillna('None').apply(lambda cabin: cabin[0] if cabin != 'None' else cabin) # cabinType

    for cabin_type in df['CabinType'].unique():
        dummy_var_name = 'CabinType={}'.format(cabin_type)
        df[dummy_var_name] = df['CabinType'].apply(lambda entry: int(entry == cabin_type)) 

    df['Embarked'] = df['Embarked'].fillna('None') # embarked

    for embarked in df['Embarked'].unique():
        dummy_var_name = 'Embarked={}'.format(embarked)
        dummy_var_val = df['Embarked'].apply(lambda entry: int(entry == embarked)) 
        df[dummy_var_name] = dummy_var_val

    del df['Embarked']

process_df(df)

# setting the dataframe with the right features

features_to_use = ['Sex', 'Pclass', 'Fare', 'Age', 'SibSp', 'SibSp>0', 'Parch>0', 'Embarked=C', 'Embarked=None', 'Embarked=Q', 'Embarked=S', 'CabinType=A', 'CabinType=B', 'CabinType=C', 'CabinType=D', 'CabinType=E', 'CabinType=F', 'CabinType=G', 'CabinType=None', 'CabinType=T']
columns_needed = ['Survived'] + features_to_use
df = df[columns_needed]

# creating interaction terms

redundant_terms = [['SibSp', 'SibSp>0'], ['Embarked=C', 'Embarked=None', 'Embarked=Q', 'Embarked=S'], ['CabinType=A', 'CabinType=B', 'CabinType=C', 'CabinType=D', 'CabinType=E', 'CabinType=F', 'CabinType=G', 'CabinType=None', 'CabinType=T']]

for term_1 in features_to_use:
    for term_2 in features_to_use:

        if features_to_use.index(term_1) < features_to_use.index(term_2):

            interaction = term_1 + ' * ' + term_2
            redundant_list = [(term_1 in elem_list and term_2 in elem_list) for elem_list in redundant_terms]

            if not any(redundant_list):
                df[interaction] = df[term_1] * df[term_2]

# computing accuracy

def get_accuracy(predictions, actual):
    correct = ['' for i in range(len(predictions)) if predictions[i] == actual[i]]
    return len(correct) / len(predictions)

features = []
final_testing_accuracy = 0

feature_list = list(df.columns)[1:]
print("\nfeature_list:", feature_list)

while True:
    
    testing_accuracy = 0
    training_accuracy = 0
    best_temp_feature = feature_list[0]
    
    for temp_feature in feature_list:
      
        if temp_feature in features:
            continue

        df_features = ['Survived'] + features + [temp_feature]
        training_df = df[:500][df_features]
        testing_df = df[500:][df_features]

        training_array = np.array(training_df)
        testing_array = np.array(testing_df)

        x_train = training_array[:,1:]
        x_test = testing_array[:,1:]

        y_train = training_array[:,0]
        y_test = testing_array[:,0]

        regressor = LogisticRegression(max_iter=1000)
        regressor.fit(x_train, y_train)

        y_train_predictions = [int(output) for output in regressor.predict(x_train)]
        y_test_predictions = [int(output) for output in regressor.predict(x_test)]

        temp_training_accuracy = get_accuracy(y_train_predictions, y_train)
        temp_testing_accuracy = get_accuracy(y_test_predictions, y_test)

        if temp_testing_accuracy > testing_accuracy:
                
            testing_accuracy = temp_testing_accuracy
            training_accuracy = temp_training_accuracy
            best_temp_feature = temp_feature
    
    if testing_accuracy <= final_testing_accuracy:
        break
    
    final_testing_accuracy = testing_accuracy
    final_training_accuracy = training_accuracy
    features.append(best_temp_feature)

    print('')
    print(features)
    print(final_training_accuracy)
    print(final_testing_accuracy)