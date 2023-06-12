""" This code defines two classes: PredictPipeline and CustomData.

The PredictPipeline class handles the prediction pipeline. It has the following methods:

__init__: Initializes the class (empty implementation).
predict: Performs the prediction using a saved model and preprocessor. It takes features as input, which should be a Pandas DataFrame. It loads the model and preprocessor from the specified file paths, scales the input features using the preprocessor, and then makes predictions using the model. The predicted values are returned.
The CustomData class represents the custom data input for prediction. It has the following methods:

__init__: Initializes the class with the provided input values: gender, race_ethnicity, parental_level_of_education, lunch, test_preparation_course, reading_score, and writing_score.
get_data_as_data_frame: Converts the custom data input into a Pandas DataFrame. It creates a dictionary with the input values as keys and their corresponding values in a list format. Then, it creates a DataFrame using this dictionary and returns it.
Additionally, the code imports necessary modules and defines an exception handling mechanism using the CustomException class.

Overall, this code provides a prediction pipeline and a way to convert custom data inputs into a DataFrame for prediction using the pipeline. """

import sys 
import pandas as pd 
from src.exception import CustomException

from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass
    def predict(self,features):
        try:
            model_path='artifacts\model.pkl'
            preprocessor_path='artifacts\preprocessor.pkl'
            model = load_object(file_path= model_path)
            preprocessor = load_object(file_path= preprocessor_path)
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e, sys)
            
    
    
class CustomData:
    def __init__(self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education,
        lunch: str,
        test_preparation_course: str,
        reading_score: int,
        writing_score: int):
        
        self.gender = gender

        self.race_ethnicity = race_ethnicity

        self.parental_level_of_education = parental_level_of_education

        self.lunch = lunch

        self.test_preparation_course = test_preparation_course

        self.reading_score = reading_score

        self.writing_score = writing_score
        
    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
        

""" The import statements import necessary modules: sys for system-related functionality, pandas as pd for data manipulation, CustomException from src.exception for handling custom exceptions, and load_object from src.utils for loading objects.
This defines a class PredictPipeline with an empty __init__ method.
The predict method takes features as input and performs the prediction pipeline.
It constructs file paths for the model and preprocessor using os.path.join.
It loads the model and preprocessor using the load_object function.
It scales the input features using the preprocessor.
It makes predictions using the model.
The predicted values are returned.
If any exception occurs, it raises a CustomException with the caught exception and the sys module.
This defines a class CustomData that represents custom data inputs for prediction.
It has an __init__ method that initializes the class with the provided input values.
The get_data_as_data_frame method converts the custom data input into a Pandas DataFrame.
It creates a dictionary custom_data_input_dict with the input values as keys and their corresponding values in a list format.
It creates a DataFrame using this dictionary and returns it.
If any exception occurs, it raises a CustomException with the caught exception and the sys module.
Overall, this code defines two classes: PredictPipeline for the prediction pipeline and CustomData for representing custom data inputs. The code provides methods for prediction and data conversion, along with exception handling using the CustomException class. """