import pandas as pd
import numpy as np
from warnings import simplefilter
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)

if True:

    # turning df into what we want

    # Read in the data
    df = pd.read_csv('/home/runner/kaggle/data&libraries/dataset.csv')

    # Filter to only the column's we're interested in
    keep_cols = ['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin', 'Embarked']
    df = df[keep_cols]

    # process the columns

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
        df[dummy_var_name] = df['Embarked'].apply(lambda entry: int(entry == embarked)) 

    del df['Embarked']

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

def get_accuracy(predictions, actual):
    correct = ['' for i in range(len(predictions)) if predictions[i] == actual[i]]
    return len(correct) / len(predictions)

def get_accuracies(input_df, features=list(df.columns)):
    
    df_train = input_df[:500][features]
    df_test = input_df[500:][features]

    arr_train = np.array(df_train)
    arr_test = np.array(df_test)

    y_train = arr_train[:,0]
    y_test = arr_test[:,0]

    x_train = arr_train[:,1:]
    x_test = arr_test[:,1:]

    regressor = LogisticRegression(max_iter=100, random_state=0)
    regressor.fit(x_train, y_train)

    y_train_predictions = [round(output) for output in regressor.predict(x_train)]
    y_test_predictions = [round(output) for output in regressor.predict(x_test)]

    return [get_accuracy(y_train_predictions, y_train), get_accuracy(y_test_predictions, y_test)]

all_features = list(df.columns)[1:]
removed_indices = []
baseline_testing_accuracy = get_accuracies(df)[1]

print("\nall features:", all_features)
print("\ntraining:", get_accuracies(df)[0]) # 0.822
print("baseline_testing_accuracy:", get_accuracies(df)[1]) # 0.7877237851662404

for i in range(167):

    current_features = ['Survived'] + [all_features[j] for j in range(167) if j not in removed_indices and i != j]
    training_accuracy = get_accuracies(df, current_features)[0]
    testing_accuracy = get_accuracies(df, current_features)[1]

    status = "kept"
    if testing_accuracy >= baseline_testing_accuracy:
        status = "removed"
        removed_indices.append(i)
        baseline_testing_accuracy = testing_accuracy

    print("\ncandidate for removal:", all_features[i], "(index {})".format(i))
    print("training:", round(training_accuracy, 6))
    print("testing:", round(testing_accuracy, 6))
    print(status)
    print("baseline testing accuracy:", round(baseline_testing_accuracy, 6))
    print("removed indices:", removed_indices)