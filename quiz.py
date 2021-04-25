import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('/home/runner/kaggle/data&libraries/students_performance.csv')

# Create dummy variables for test preparation course and parental level of education. Then, fit a linear regression to all the data except for the last 3 rows, and use it to predict the math scores in the last 3 rows of the data. What scores do you get?

make_dummy_variables_for = ["test preparation course", "parental level of education"]
new_columns = ["math score"] + make_dummy_variables_for
df = df[new_columns]

for key in make_dummy_variables_for:
    unique_values = df[key].unique()
    for value in unique_values:
        dummy_key = '{}={}'.format(key, value)
        df[dummy_key] = df[key].apply(lambda x: int(x == value))
    del df[key]

all_data_array = np.array(df)
train_array = all_data_array[:-3, :]
test_array = all_data_array[-3:, :]

X_train = train_array[:, 1:]
X_test = test_array[:, 1:]

y_train = train_array[:, 0]
y_test = test_array[:, 0]

regressor = LogisticRegression()
regressor.fit(X_train, y_train)
predictions = regressor.predict(X_test)

print(y_test)
print(predictions)

# I didn't do the last part of #2 because I didn't know how to start, and I ran out of time finishing the other problems.