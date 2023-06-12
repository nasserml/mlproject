""" This code provides utility functions for saving and loading objects, as well as evaluating models using grid search. Here's a summary of each function:

save_object(file_path, obj): This function takes a file path and an object as input and saves the object to the specified file path using pickle. It creates the necessary directories if they don't exist. If any exception occurs during the process, a CustomException is raised.

evaluate_models(X_train, y_train, X_test, y_test, models, param): This function takes the training and testing data (X_train, y_train, X_test, y_test), a dictionary of models (models), and a dictionary of model parameters (param) as input. It evaluates each model using grid search cross-validation. It iterates over the models and parameters, performs grid search with cross-validation, sets the best parameters found for the model, and fits the model on the training data. It then calculates the R-squared scores for both the training and testing data and stores the scores in a dictionary. Finally, it returns the dictionary containing the model names and their corresponding test scores. If any exception occurs during the process, a CustomException is raised.

load_object(file_path): This function takes a file path as input and loads the object stored in the file using pickle. It returns the loaded object. If any exception occurs during the process, a CustomException is raised.

Overall, these utility functions provide convenient methods for saving and loading objects and performing model evaluation using grid search. They handle exceptions and provide a consistent way to handle errors throughout the codebase. """

import os 
import sys

import numpy as np
import pandas as pd 

import dill
import pickle

from src.exception import CustomException    
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train,X_test,y_test,models,param):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            ####
            para=param[list(models.keys())[i]]

            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            #model.fit(X_train, y_train)  # Train model

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)

            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)
    
""" 
import os: Imports the module for interacting with the operating system, providing functions for file and directory operations.
import sys: Imports the module that provides access to some variables used or maintained by the interpreter and to functions that interact with the interpreter.
import numpy as np: Imports the module for numerical computing with arrays and mathematical functions. It is commonly aliased as np for convenience.
import pandas as pd: Imports the module for data manipulation and analysis. It is commonly aliased as pd for convenience.
import dill: Imports the module for extended pickling and unpickling of Python objects.
import pickle: Imports the module for serializing and deserializing Python objects.
from src.exception import CustomException: Imports the CustomException class from the src.exception module.
def save_object(file_path, obj):: Defines a function named save_object that takes a file path and an object as input.
try:: Starts a try block to handle potential exceptions.
dir_path = os.path.dirname(file_path): Retrieves the directory path from the given file path.
os.makedirs(dir_path, exist_ok=True): Creates the directory specified by dir_path if it doesn't exist. The exist_ok=True parameter ensures that the function doesn't raise an exception if the directory already exists.
with open(file_path, "wb") as file_obj:: Opens the file specified by file_path in binary write mode and assigns it to the variable file_obj. The with statement ensures that the file is properly closed after the block is executed.
dill.dump(obj, file_obj): Serializes the obj using dill and writes it to the file specified by file_obj.
except Exception as e:: Catches any exception that occurs within the try block and assigns it to the variable e.
raise CustomException(e, sys): Raises a CustomException with the caught exception e and the sys module as arguments.
The evaluate_models function follows a similar structure. It takes training and testing data, a dictionary of models, and a dictionary of model parameters as input. It performs grid search cross-validation on each model, sets the best parameters, fits the model on the training data, and calculates the R-squared scores for both the training and testing data. The scores are stored in a dictionary and returned as the result.

These functions provide utility for saving objects, evaluating models, and handling exceptions in the process. They ensure that the objects are saved correctly, models are evaluated with the best parameters, and any exceptions are properly handled and raised as CustomException instances.
"""
"""  the evaluate_models function:

def evaluate_models(X_train, y_train, X_test, y_test, models, param):: Defines the function evaluate_models with parameters X_train, y_train, X_test, y_test, models, and param.

report = {}: Initializes an empty dictionary called report to store the evaluation results of the models.

for i in range(len(list(models))):: Iterates over the indices of the models dictionary.

model = list(models.values())[i]: Retrieves the model at the current index using list(models.values()) and indexing with i. This line extracts the model object from the dictionary.

para = param[list(models.keys())[i]]: Retrieves the parameter grid for the current model by accessing the corresponding key-value pair from the param dictionary.

gs = GridSearchCV(model, para, cv=3): Creates an instance of GridSearchCV with the current model and parameter grid. The cv parameter is set to 3, indicating 3-fold cross-validation.

gs.fit(X_train, y_train): Fits the grid search object gs to the training data X_train and y_train, performing the grid search to find the best hyperparameters for the model.

model.set_params(**gs.best_params_): Updates the model's parameters with the best parameters found during grid search. The best_params_ attribute of gs contains the optimal hyperparameters.

model.fit(X_train, y_train): Fits the model to the training data with the updated parameters. This step trains the model on the training set.

y_train_pred = model.predict(X_train): Predicts the target variable y_train using the trained model on the training data X_train.

y_test_pred = model.predict(X_test): Predicts the target variable y_test using the trained model on the test data X_test.

train_model_score = r2_score(y_train, y_train_pred): Calculates the R-squared score between the actual and predicted values for the training data.

test_model_score = r2_score(y_test, y_test_pred): Calculates the R-squared score between the actual and predicted values for the test data.

report[list(models.keys())[i]] = test_model_score: Stores the test R-squared score of the current model in the report dictionary. The key is the name of the model retrieved using list(models.keys())[i], and the value is test_model_score.

return report: Returns the report dictionary containing the test R-squared scores for each model.

30-34. except Exception as e: raise CustomException(e, sys): Catches any exception that occurs during the evaluation process and raises a custom exception CustomException with the original exception and sys information.

In summary, the evaluate_models function takes the training and test data, a dictionary of models, and a dictionary of corresponding parameter grids. It performs grid search with cross-validation to find the best hyperparameters for each model, trains the models with the optimal parameters, and evaluates their performance by calculating the R-squared scores on both the training and test data. The results are stored in a dictionary and returned. """