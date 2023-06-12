
""" This code can be summarized as follows:

The code imports necessary libraries and modules such as sys, dataclass, numpy, pandas, and various classes and functions from the sklearn library.
It imports custom classes and functions from the project's source files.
It defines a data class DataTransformationConfig with a configuration attribute preprocessor_obj_file_path that specifies the file path for saving a preprocessor object.
It defines a class DataTransformation that handles data transformation operations.
The __init__ method initializes an instance of the DataTransformation class and sets the data_transformation_config attribute to an instance of the DataTransformationConfig class.
The get_data_transformer_object method defines a data transformation pipeline using ColumnTransformer from sklearn. It creates separate pipelines for numerical and categorical columns, performs imputation, one-hot encoding, and scaling operations.
The method returns the constructed ColumnTransformer object.
The initiate_data_transformation method takes the paths to training and test data as input. It reads the CSV files using pd.read_csv and obtains a preprocessing object by calling the get_data_transformer_object method.
It specifies the target column name and selects the input features and target features from the data frames.
The method applies the preprocessing object to transform the input features for both the training and test data.
It combines the transformed input features with the target features into NumPy arrays.
The method saves the preprocessing object using the save_object function.
Finally, the method returns the transformed training and test arrays along with the file path of the saved preprocessing object. """


import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
import os 

from src.utils import save_object

'''
import sys: Imports the sys module, which provides access to system-specific parameters and functions.
from dataclasses import dataclass: Imports the dataclass decorator from the dataclasses module, which simplifies the creation of classes with automatically generated special methods.
import numpy as np: Imports the numpy library and assigns it the alias np.
import pandas as pd: Imports the pandas library and assigns it the alias pd.
from sklearn.compose import ColumnTransformer: Imports the ColumnTransformer class from the sklearn.compose module, which allows applying different transformers to different columns of an array or dataframe.
from sklearn.impute import SimpleImputer: Imports the SimpleImputer class from the sklearn.impute module, which provides strategies for imputing missing values.
from sklearn.pipeline import Pipeline: Imports the Pipeline class from the sklearn.pipeline module, which allows chaining multiple transformers and an estimator into a single object.
from sklearn.preprocessing import OneHotEncoder, StandardScaler: Imports the OneHotEncoder and StandardScaler classes from the sklearn.preprocessing module, which are used for one-hot encoding and standardization, respectively.
from src.exception import CustomException: Imports the CustomException class from the src.exception module, which is a custom exception class.
from src.logger import logging: Imports the logging object from the src.logger module, which provides a logging interface.
import os: Imports the os module, which provides a way to interact with the operating system.
from src.utils import save_object: Imports the save_object function from the src.utils module, which is used to save Python objects to a file.
'''

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts', 'preprocessor.pkl')
    '''
    @dataclass: A decorator that automatically generates special methods for the class based on the defined attributes. In this case, it creates the __init__ method for DataTransformationConfig class.
class DataTransformationConfig: Defines a class representing the configuration for data transformation.
preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl'): Sets the default value for the preprocessor_obj_file_path attribute of DataTransformationConfig class as the file path 'artifacts/preprocessor.pkl' using the os.path.join function.
    '''
    
class DataTransformation:
    def  __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        
        """ 
        class DataTransformation: Defines a class representing the data transformation component.
def __init__(self): Defines the constructor method for the DataTransformation class.
self.data_transformation_config = DataTransformationConfig(): Initializes an instance of DataTransformationConfig class and assigns it to the data_transformation_config attribute of the DataTransformation instance.
        """
        
    def get_data_transformer_object(self):
        
        """ Defines a method named get_data_transformer_object inside the DataTransformation class. """
        
        '''
        this function is responsible for data transformation
        '''
        try:
            
             numerical_columns = ["writing_score", "reading_score"]
             categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]
             num_pipeline = Pipeline(
                 
                 steps=[
                 ("imputer", SimpleImputer(strategy="median")),
                 ("scaler", StandardScaler())
                 ]    
             )
             cat_pipeline=Pipeline(
                 steps=[
                     ('imputer', SimpleImputer(strategy='most_frequent')),
                     ('one_hot_encoder', OneHotEncoder()),
                     ('scaler',StandardScaler(with_mean=False))
                 ]
                     
             )
             logging.info(f'Numerical columns: {numerical_columns}')
             logging.info(f'Categorical columns: {categorical_columns}')
             preprocessor = ColumnTransformer(
                 [
                     ('num_pipeline', num_pipeline, numerical_columns),
                     ('cat_pipeline', cat_pipeline,categorical_columns)
                 ]
             )
             return preprocessor
             
            
        except Exception as e:
            raise CustomException(e,sys)
        """ numerical_columns = ["writing_score", "reading_score"]: Defines a list of column names that represent numerical columns in the dataset.
categorical_columns = [...]: Defines a list of column names that represent categorical columns in the dataset.
num_pipeline = Pipeline(steps=[...]): Creates a pipeline for numerical columns. It consists of two steps: imputer and scaler. The imputer step replaces missing values with the median, and the scaler step standardizes the data.
cat_pipeline = Pipeline(steps=[...]): Creates a pipeline for categorical columns. It consists of three steps: imputer, one_hot_encoder, and scaler. The imputer step replaces missing values with the most frequent value, the one_hot_encoder step performs one-hot encoding, and the scaler step standardizes the data without centering.
preprocessor = ColumnTransformer([...]): Creates a ColumnTransformer object that applies the specified pipelines to the corresponding columns. The num_pipeline is applied to numerical_columns, and the cat_pipeline is applied to categorical_columns.
logging.info(f'Numerical columns: {numerical_columns}'): Logs the numerical column names.
logging.info(f'Categorical columns: {categorical_columns}'): Logs the categorical column names.
return preprocessor: Returns the preprocessor object. Catches any exception that occurs in the try block and raises a CustomException with the captured exception and the sys module. """
            
    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj=self.get_data_transformer_object()

            target_column_name="math_score"
            numerical_columns = ["writing_score", "reading_score"]

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)
        """ Defines a method named initiate_data_transformation inside the DataTransformation class that takes train_path and test_path as parameters. 
        
         train_df = pd.read_csv(train_path): Reads the training data from the CSV file specified by train_path using pd.read_csv.
test_df = pd.read_csv(test_path): Reads the testing data from the CSV file specified by test_path using pd.read_csv.
logging.info('Read train and test data completed'): Logs a message indicating that the train and test data have been successfully read.
`logging.info('Obtaining preprocessing object """
            
            
        
        """ preprocessing_obj = self.get_data_transformer_object(): Calls the get_data_transformer_object() method to obtain the preprocessor object for data transformation.
target_column_name = 'math_score': Specifies the name of the target column in the dataset.
input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1): Creates a dataframe input_feature_train_df by dropping the target column from the training dataframe.
target_feature_train_df = train_df[target_column_name]: Creates a dataframe target_feature_train_df containing only the target column from the training dataframe.
input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1): Creates a dataframe input_feature_test_df by dropping the target column from the testing dataframe.
target_feature_test_df = test_df[target_column_name]: Creates a dataframe target_feature_test_df containing only the target column from the testing dataframe.
input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df): Applies the preprocessor object to the training input features dataframe and obtains the transformed array.
input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df): Applies the preprocessor object to the testing input features dataframe and obtains the transformed array.
train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]: Combines the transformed training input features array with the target feature array using np.c_ (column-wise concatenation).
test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]: Combines the transformed testing input features array with the target feature array using np.c_.
save_object(file_path=self.data_transformation_config.preprocessor_obj_file_path, obj=preprocessing_obj): Saves the preprocessing object to the file specified by preprocessor_obj_file_path.
Returns a tuple (train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path) containing the transformed training array, transformed testing array, and the file path of the saved preprocessing object.Catches any exception that occurs in the try block and raises a CustomException with the captured exception and the sys module."""