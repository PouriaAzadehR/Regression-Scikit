import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from utils import *

font = {'family': 'serif',
        'weight': 'normal',
        'size': 10
        }

##loading data and preparing matrixes
#data = np.loadtxt('./data/data_w3_ex2.csv', delimiter=',')
data = np.loadtxt('./data2.csv', delimiter=',')
x = data[:,:-1]
y = data[:,-1]
y1 = np.expand_dims(y, axis=1)

##generating train, cross validation and test sets
x_train, x_, y_train, y_  = train_test_split(x, y1, test_size=0.40, random_state=1) # 0.4: 40% remained, 60% used
x_cv, x_test, y_cv, y_test = train_test_split(x_, y_, test_size=0.50, random_state=1)
del x_, y_

## Initialize lists containing the Mean Square Errors, models, and scalers
errors = []
train_errors = []
cv_errors = []
models = []
scalers = []

## Loop over different polynomials to add extra features for a more accurate fit. This might cause overfitting.
num = 5
for degree in range(1,num):

    ## Add polynomial features to the training set (data engineering)
    poly = PolynomialFeatures(degree, include_bias=False)
    X_train_mapped = poly.fit_transform(x_train)
    ## Scale the training set (normalizing)
    scaler = StandardScaler()
    X_train_mapped_scaled = scaler.fit_transform(X_train_mapped)
    #mean = scaler.mean_
    #standard_deviation = scaler.scale_
    scalers.append(scaler)

    ## Create and train the model
    model = LogisticRegression()
    model.fit(X_train_mapped_scaled, y_train)
    models.append(model)

    ## Compute the training error
    yhat = model.predict(X_train_mapped_scaled)
    train_error = 1-np.mean(yhat == y_train.T[0])
    train_errors.append(train_error)

    ## Add polynomial features and scale the cross validation set
    X_cv_mapped = poly.fit_transform(x_cv)
    X_cv_mapped_scaled = scaler.transform(X_cv_mapped)

    ## Compute the cross validation error
    yhat = model.predict(X_cv_mapped_scaled)
    cv_error = 1-np.mean(yhat == y_cv.T[0])
    cv_errors.append(cv_error)

    errors.append([cv_error, train_error])
    #print(model.coef_[0],model.intercept_)

# Print the result
for model_num in range(len(train_errors)):
    print(
        f"Model {model_num+1}: Training Set Classification Error: {train_errors[model_num]*100:.2f}%, " +
        f"CV Set Classification Error: {cv_errors[model_num]*100:.2f}%"
        )

## best fit
min_error = errors[0]
min_index = 0
for i in range(len(errors)):
    if errors[i][0] < min_error[0]:
        min_error = errors[i]
        min_index = i
    elif errors[i][0] == min_error[0]:
        if errors[i][1] < min_error[1]:
            min_error = errors[i]
            min_index = i

best_fit = min_index + 1
print("\n" + f"Selected Model: {best_fit}th degree")

## Add polynomial features to the test set using best_fit and Scale it
poly = PolynomialFeatures(best_fit, include_bias=False)
X_test_mapped = poly.fit_transform(x_test)
X_test_mapped_scaled = scalers[best_fit-1].transform(X_test_mapped)

# Compute the test error
yhat = models[best_fit-1].predict(X_test_mapped_scaled)
test_error = np.mean(yhat != y_test.T[0])
print(f"Training ERROR: {train_errors[best_fit-1]*100:.2f}%")
print(f"Cross Validation ERROR: {cv_errors[best_fit-1]*100:.2f}%")
print(f"Test ERROR: {test_error*100:.2f}% \n")

##plot the data and best_fit
poly = PolynomialFeatures(best_fit, include_bias=False)
X_mapped = poly.fit_transform(x)
X_mapped_scaled = scalers[best_fit-1].transform(X_mapped)
yhat = models[best_fit-1].predict(X_mapped_scaled)
print('Overal ERROR of the Selected Model: %0.2f'%((1-np.mean(yhat == y)) * 100) + "% \n")

## final plot using best_fit and the original data
def plot_output(X,y,scalers,models,best_model):

    num_fig = len(models)
    num_row = num_fig//2+num_fig%2
    num_column = 2
    Fig, ax = plt.subplots(num_row, num_column,figsize=(5*num_row, 5*num_column),sharey=True,sharex=True)
    j = 0
    for i in range(len(models)):
        poly = PolynomialFeatures(i+1, include_bias=False)
        X_mapped = poly.fit_transform(x)
        X_mapped_scaled = scalers[i].transform(X_mapped)
        yhat = models[i].predict(X_mapped_scaled)
        yhat = np.expand_dims(yhat, axis=1)
        positive1 = y[:,0] == 1
        negative1 = y[:,0] == 0
        positive2 = yhat[:,0] == 1
        negative2 = yhat[:,0] == 0

        ax[i//2,j].plot(X[positive1, 0], X[positive1, 1], markeredgecolor='r', label="type1_actual",linestyle="None",marker="o",markerfacecolor='None')
        ax[i//2,j].plot(X[negative1, 0], X[negative1, 1], markeredgecolor='b', label="type2_actual",linestyle="None",marker="x",markerfacecolor='None')
        ax[i//2,j].plot(X[positive2, 0], X[positive2, 1], markeredgecolor='r', label="type1_predicted",linestyle="None",marker="+",markerfacecolor='None')
        ax[i//2,j].plot(X[negative2, 0], X[negative2, 1], markeredgecolor='b', label="type2_predicted",linestyle="None",marker="D",markerfacecolor='None')
        ax[i//2,j].legend(loc='upper left',prop = font)
        if i+1 == best_model:
            ax[i//2,j].set_title(f'model {i+1}' + ': The best model', fontdict=font)
        else:
            ax[i//2,j].set_title(f'model {i+1}', fontdict=font)
        if i%2 == 0:
            ax[i//2,j].set_ylabel('y', fontdict=font)
        if i == len(models) - 1 or i == len(models) - 2:
            ax[i//2,j].set_xlabel('x', fontdict=font)
        j=(i+1)%2
    plt.show()

plot_output(x,y1,scalers,models,best_fit)
