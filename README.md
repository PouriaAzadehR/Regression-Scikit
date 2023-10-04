# Regression-Scikit
---

# Polynomial Regression with Train, Cross-Validation, and Test Sets

This Python code demonstrates Polynomial Regression using Scikit-Learn for a dataset. The goal is to predict a target variable based on an input variable. It's organized into three primary sections: data preparation, model training, and evaluation.

## Key Concepts

### Data Preparation

1. **Data Loading**: The code loads data from a CSV file named `data1.csv`, where each row represents an example with an input and a target value.

2. **Data Splitting**: The data is divided into three sets: `train`, `cross-validation`, and `test`. The `train` set is used for training the models, the `cross-validation` set for tuning hyperparameters, and the `test` set for evaluating the final model.

3. **Feature Engineering**: Polynomial features are added to the input variable up to degree 10. This creates a set of features that capture higher-order relationships between the input and target variable.

### Model Training

1. **Model Selection**: Polynomial regression models are trained for different degrees of polynomial features, from 1 to 10. Higher degrees can lead to overfitting, so the goal is to find the degree that performs best on the cross-validation set.

2. **Scaling Features**: Standard scaling is applied to the features to ensure they have a mean of 0 and a standard deviation of 1. This step is crucial for polynomial regression to work effectively.

3. **Training the Models**: Linear regression models are trained for each degree of polynomial features using the scaled training data.

### Model Evaluation

1. **MSE Calculation**: The Mean Squared Error (MSE) is computed for each trained model on the training, cross-validation, and test sets. The MSE measures the goodness of fit of the model to the data.

2. **Best Model Selection**: The model with the lowest MSE on the cross-validation set is chosen as the best model for making predictions.

3. **Final Testing**: The best model is evaluated on the test set to estimate its performance on unseen data.

## Getting Started

To use this code:

1. Ensure you have Python and the required libraries installed (NumPy, Matplotlib, Scikit-Learn).

2. Prepare your dataset as a CSV file named `data1.csv`, with two columns: input and target values.

3. Run the code, and it will automatically split the data into train, cross-validation, and test sets, perform polynomial feature engineering, train multiple models, and evaluate their performance.

4. The code will display the training, cross-validation, and test MSEs for each degree of polynomial features. The best-performing model and its MSE on the test set will be reported.

## Customize and Extend

Feel free to customize this code for your specific dataset or experiment with different degrees of polynomial features. You can also explore other evaluation metrics or visualization techniques to gain more insights into the model's performance.

## License

This project is licensed under the MIT License.

---

This README provides an overview of the code's functionality, focusing on data preparation, polynomial regression, and model evaluation with an emphasis on train, cross-validation, and test sets.
