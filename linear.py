import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def plot_train_cv_test(x_train, y_train, x_cv, y_cv, x_test, y_test, title):
    plt.scatter(x_train, y_train, marker='x', c='r', label='training');
    plt.scatter(x_cv, y_cv, marker='o', c='b', label='cross validation');
    plt.scatter(x_test, y_test, marker='^', c='g', label='test');
    plt.title("input vs. target")
    plt.xlabel("x");
    plt.ylabel("y");
    plt.legend()
    plt.show()

##loading data and preparing matrixes
data = np.loadtxt('./data1.csv', delimiter=',')
x = data[:,0]
y = data[:,1]
x = np.expand_dims(x, axis=1) # vector to matrix
y = np.expand_dims(y, axis=1)

##generating train, cross validation and test sets
x_train, x_, y_train, y_  = train_test_split(x, y, train_size=0.60, test_size=0.40, random_state=42)
x_cv, x_test, y_cv, y_test = train_test_split(x_, y_, train_size=0.50, test_size=0.50, random_state=42)
del x_, y_

# #plot the loaded and prepared data
# utils.plot_dataset(x=x, y=y, title="input vs. target")
# utils.plot_train_cv_test(x_train, y_train, x_cv, y_cv, x_test, y_test, title="input vs. target")


## Initialize lists containing the Mean Square Errors, models, and scalers
train_mses = []
cv_mses = []
models = []
scalers = []

## Loop over different polynomials to add extra features for a more accurate fit. This might cause overfitting.
num = 11
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
    model = LinearRegression()
    model.fit(X_train_mapped_scaled, y_train )
    models.append(model)

    ## Compute the training MSE
    yhat = model.predict(X_train_mapped_scaled)
    train_mse = mean_squared_error(y_train, yhat) / 2
    train_mses.append(train_mse)

    ## Add polynomial features and scale the cross validation set
    X_cv_mapped = poly.fit_transform(x_cv)
    X_cv_mapped_scaled = scaler.transform(X_cv_mapped)

    ## Compute the cross validation MSE
    yhat = model.predict(X_cv_mapped_scaled)
    cv_mse = mean_squared_error(y_cv, yhat) / 2
    cv_mses.append(cv_mse)

## Plot the results
#degrees=range(1,num)
#utils.plot_train_cv_mses(degrees, train_mses, cv_mses, title="degree of polynomial vs. train and CV MSEs")

## best fit
best_fit = np.argmin(cv_mses) + 1

## Add polynomial features to the test set using best_fit and Scale it
poly = PolynomialFeatures(best_fit, include_bias=False)
X_test_mapped = poly.fit_transform(x_test)
X_test_mapped_scaled = scalers[best_fit-1].transform(X_test_mapped)

# Compute the test MSE
yhat = models[best_fit-1].predict(X_test_mapped_scaled)
test_mse = mean_squared_error(y_test, yhat) / 2

print(f"Training MSE: {train_mses[best_fit-1]}")
print(f"Cross Validation MSE: {cv_mses[best_fit-1]}")
print(f"Test MSE: {test_mse}")

##plot the data and best_fit
poly = PolynomialFeatures(best_fit, include_bias=False)
X_mapped = poly.fit_transform(x)
X_mapped_scaled = scalers[best_fit-1].transform(X_mapped)
yhat = models[best_fit-1].predict(X_mapped_scaled)

## final plot using best_fit and the original data
xy_plot = np.concatenate((x,yhat),axis=-1)
xy_plot= np.sort(xy_plot,axis=0)
plt.plot(xy_plot[:,0],xy_plot[:,1],label=f"best fit using poly degree = {best_fit}")
plot_train_cv_test(x_train, y_train, x_cv, y_cv, x_test, y_test, title="input vs. target")
