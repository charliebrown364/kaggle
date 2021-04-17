import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('/home/runner/kaggle/students_performance.csv')

# What are the math scores in the last 3 rows of the data?

# print("part a:", df["math score"][-3:])

# What is the average math score across all students?

def avg(input_list):
    return sum(input_list) / len(input_list)

print("part b:", avg(df["math score"]))

# What were the average math scores for students who did vs didn't complete the test preparation course?

df1 = df.copy()

good_math_scores = []
bad_math_scores = []

for i in range(len(list(df["math score"]))):
    
    math = df["math score"][i]
    test_prep = df["test preparation course"][i]

    if test_prep == 'completed':
        good_math_scores.append(math)
    elif test_prep == 'none':
        bad_math_scores.append(math)

print("part c (completed):", avg(good_math_scores))
print("part c (didn't complete):", avg(bad_math_scores))

# How many categories of parental level of education are there?

num_categories = len(set(df['parental level of education']))
print("part d:", num_categories)

# Create dummy variables for test preparation course and parental level of education. Then, fit a linear regression to all the data except for the last 3 rows, and use it to predict the math scores in the last 3 rows of the data. What scores do you get?