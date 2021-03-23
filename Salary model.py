import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle

data = pd.read_csv('Salary_Data.csv')
data

plt.scatter(data.YearsExperience, data.Salary)
plt.show()

# Split data into train and validation
train = data[:int(0.7*len(data))]
valid = data[int(0.7*len(data)):]

# Assigning feature vectors to target and explanatory variable
Y_train = train.Salary
X_train = train.YearsExperience
X_train = sm.add_constant(X_train)

Y_valid = valid.Salary
X_valid = valid.YearsExperience
X_valid = sm.add_constant(X_valid)

# Building a Simple Linear Regression model
model = sm.OLS(Y_train, X_train)
results = model.fit()
results.params

# Predict with validation data
predicted = results.predict(X_valid)

# Model performance
print(f'The MAE is {mean_absolute_error(Y_valid, predicted)}')
print(f'The RMSE is {np.sqrt(mean_squared_error(Y_valid, predicted))}')

# Visualizing predictions and actual values
plt.scatter(Y_valid, predicted)
plt.show()

# Visualizing the model
plt.scatter(X_valid.YearsExperience, Y_valid)
plt.plot(X_valid.YearsExperience, predicted, color='orange')
plt.show()

with open('model.pkl', 'wb') as f:
    pickle.dump(results, f)
    print('Pickling completed')