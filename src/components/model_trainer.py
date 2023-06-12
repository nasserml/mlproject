
""" The code imports necessary libraries, modules, and classes such as os, sys, dataclass, and various regression models from the catboost, sklearn, and xgboost libraries.
It imports custom classes and functions from the project's source files.
It defines a data class ModelTrainerConfig with a configuration attribute trained_model_file_path that specifies the file path for saving the trained model.
It defines a class ModelTrainer that handles model training and evaluation.
The __init__ method initializes an instance of the ModelTrainer class and sets the model_trainer_config attribute to an instance of the ModelTrainerConfig class.
The initiate_model_trainer method takes the transformed training and test arrays as input. It splits the arrays into input features and target variables.
It defines a dictionary models that contains various regression models with their corresponding class instances.
It defines a dictionary params that specifies hyperparameter search spaces for each model.
The method calls the evaluate_models function with the input data and the models and params dictionaries to perform model evaluation and selection.
It retrieves the best model based on the highest evaluation score from the evaluation results.
If the best model's score is below a threshold (0.6 in this case), it raises a custom exception indicating that no best model was found.
It saves the best model using the save_object function.
The method predicts the target variable for the test data using the best model.
It calculates the R-squared score between the predicted and actual target values.
Finally, the method returns the R-squared score as the evaluation result.
Overall, this code defines a model trainer class that performs model training, evaluation, and selection using various regression models. The best model is saved, and its performance is evaluated using the R-squared score. """
import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models
""" 
    import os: Imports the os module, which provides a way to use operating system dependent functionality.
import sys: Imports the sys module, which provides access to some variables used or maintained by the interpreter and to functions that interact with the interpreter.
from dataclasses import dataclass: Imports the dataclass decorator from the dataclasses module, which simplifies the creation of classes that are primarily used to store data.
from catboost import CatBoostRegressor: Imports the CatBoostRegressor class from the catboost library, which is a gradient boosting library that provides an implementation of gradient boosting algorithms.
from sklearn.ensemble import (AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor): Imports the AdaBoostRegressor, GradientBoostingRegressor, and RandomForestRegressor classes from the sklearn.ensemble module, which provides ensemble-based machine learning algorithms.
from sklearn.linear_model import LinearRegression: Imports the LinearRegression class from the sklearn.linear_model module, which provides linear regression models.
from sklearn.metrics import r2_score: Imports the r2_score function from the sklearn.metrics module, which computes the coefficient of determination R^2 of a prediction.
from sklearn.neighbors import KNeighborsRegressor: Imports the KNeighborsRegressor class from the sklearn.neighbors module, which provides k-nearest neighbors regression.
from sklearn.tree import DecisionTreeRegressor: Imports the DecisionTreeRegressor class from the sklearn.tree module, which provides decision tree-based regression models.
from xgboost import XGBRegressor: Imports the XGBRegressor class from the xgboost library, which is an optimized gradient boosting library that provides an implementation of gradient boosting algorithms.
from src.exception import CustomException: Imports the CustomException class from the src.exception module, which is a custom exception class used in the project.
from src.logger import logging: Imports the logging object from the src.logger module, which is used for logging messages.
from src.utils import save_object, evaluate_models: Imports the save_object and evaluate_models functions from the src.utils module, which are utility functions used in the project.
"""

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join('artifacts','model.pkl')
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        
    
    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split atraining and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }
            params={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoosting Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }
            
            model_report:dict = evaluate_models(X_train=X_train, y_train=y_train,X_test= X_test,y_test=y_test,
                                               models=models, param=params)
            
            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]
            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")
            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)

            r2_square = r2_score(y_test, predicted)
            return r2_square
            



            
        except Exception as e:
            raise CustomException(e,sys)
        
""" 
@dataclass: This is a decorator that enables the automatic generation of special methods, such as __init__, __repr__, etc., for a class. In this case, it is used to define the ModelTrainerConfig class as a data class.

class ModelTrainerConfig:: Defines the ModelTrainerConfig class, which stores configuration settings for the ModelTrainer class.

trained_model_file_path = os.path.join("artifacts","model.pkl"): Sets the default value for the trained_model_file_path attribute of ModelTrainerConfig as the path to save the trained model file.

class ModelTrainer:: Defines the ModelTrainer class.

def __init__(self):: Initializes an instance of the ModelTrainer class.

self.model_trainer_config = ModelTrainerConfig(): Creates an instance of ModelTrainerConfig and assigns it to the model_trainer_config attribute of the ModelTrainer instance.

def initiate_model_trainer(self, train_array, test_array):: Defines the initiate_model_trainer method that takes train_array and test_array as input.

X_train, y_train, X_test, y_test = ...: Unpacks the train_array and test_array into X_train, y_train, X_test, and y_test variables. This assumes that the arrays are structured in a way that separates features from target values.

17-41. models = {...}, params = {...}: Defines dictionaries models and params that specify different models and their corresponding hyperparameter grids. These dictionaries are used for model evaluation.

model_report: dict = evaluate_models(...): Calls the evaluate_models function with the input data and the defined models and parameters. The function returns a dictionary containing the evaluation scores for each model.

best_model_score = max(sorted(model_report.values())): Finds the maximum evaluation score from the model_report dictionary.

best_model_name = list(model_report.keys())[...]: Retrieves the name of the best model based on the maximum score from model_report.

best_model = models[best_model_name]: Retrieves the best model object based on the name obtained from best_model_name.

55-56. if best_model_score < 0.6: ...: Checks if the best model's score is below a threshold (0.6 in this case). If it is, raises a CustomException indicating that no best model was found.

save_object(file_path=self.model_trainer_config.trained_model_file_path, obj=best_model): Saves the best model object to a file using the save_object function.

predicted = best_model.predict(X_test): Uses the best model to make predictions on the test data.

r2_square = r2_score(y_test, predicted): Calculates the R-squared score between the actual target values y_test and the predicted values predicted.

return r2_square: Returns the R-squared score.

69-72. except Exception as e: raise CustomException(e, sys): Catches any exception that occurs during the process and raises a CustomException with the original exception and sys information.

In summary, the ModelTrainer class has a method called initiate_model_trainer that takes training and test data. It evaluates different models on the provided data using the evaluate_models function. It selects the best-performing model based on the evaluation scores and saves it as a serialized object using the save_object function. If the best model's score is below a certain threshold, it raises a CustomException. Finally, it uses the best model to make predictions on the test data and calculates the R-squared score as the evaluation metric. The R-squared score is then returned as the result of the initiate_model_trainer method.

The code relies on external functions and classes such as GridSearchCV, r2_score, and CustomException, which are assumed to be imported from other modules.










"""