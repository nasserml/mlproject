""" 
This code can be summarized as follows:

The code imports necessary libraries, modules, and classes such as os, sys, CustomException, logging, pandas, train_test_split, and dataclass.
It imports classes and configurations from other source files in the project.
It defines a data class DataIngestionConfig with configuration attributes for file paths of train data, test data, and raw data.
It defines a class DataIngestion that handles the data ingestion process.
The __init__ method initializes an instance of the DataIngestion class and sets the ingestion_config attribute to an instance of the DataIngestionConfig class.
The initiate_data_ingestion method reads a dataset from a CSV file and saves it as a dataframe. It creates directories for saving train, test, and raw data files if they don't exist.
It splits the dataset into train and test sets using train_test_split.
It saves the train and test sets as CSV files using the specified file paths.
Finally, the method returns the file paths of the train and test data.
In the main block, an instance of DataIngestion class is created and the initiate_data_ingestion method is called to obtain the train and test data file paths.
An instance of DataTransformation class is created, and the initiate_data_transformation method is called to perform data transformation on the train and test data.
An instance of ModelTrainer class is created, and the initiate_model_trainer method is called to train and evaluate models on the transformed data.
The R-squared score of the best model is printed.
Overall, this code implements the data ingestion, transformation, and model training process. It reads a dataset, splits it into train and test sets, saves them as CSV files, performs data transformation on the train and test data, trains multiple regression models, selects the best model based on evaluation results, and prints the R-squared score of the best model.
"""

import os 
import sys 
from src.exception import CustomException
from src.logger import logging

'''
These lines import the necessary modules for the code, including os for operating system-related functionalities, sys for system-specific parameters and functions, CustomException from src.exception for handling custom exceptions, and logging from src.logger for logging messages.
'''

import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer
'''
These lines import additional modules: pandas as pd for data manipulation and analysis, train_test_split from sklearn.model_selection for splitting the dataset into training and test sets, and dataclass for creating a class with automatically generated special methods.
'''

@dataclass
class DataIngestionConfig:
    train_data_path:str = os.path.join('artifacts', 'train.csv')
    test_data_path:str = os.path.join('artifacts', 'test.csv')
    raw_data_path:str = os.path.join('artifacts', 'data.csv')

'''
This code defines a dataclass DataIngestionConfig that stores the file paths for the training data, test data, and raw data. The dataclass decorator automatically generates special methods for the class.
'''
    

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df = pd.read_csv('notebook\data\stud.csv')
            logging.info('Read the dataset as dataframe')
            
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            
            logging.info('train test split iinitiated')
            train_set, test_set = train_test_split(df,test_size=0.2, random_state=42)
            
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            
            logging.info("Ingestion of the data is completed")
            
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
            
        except Exception as e:
            raise CustomException(e,sys)

'''
This code defines the initiate_data_ingestion method within the DataIngestion class. This method performs the data ingestion process, including reading the dataset from the CSV file, creating the necessary directories, saving the raw data as a CSV file, splitting the data into training and test sets, and saving them as separate CSV files. It logs the progress using the logging module. If an exception occurs during the process, it raises a CustomException with the exception details.
'''
    

if __name__ == "__main__":
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()

    data_transformation=DataTransformation()
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)

    modeltrainer=ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr,test_arr))
    
'''
This code checks if the script is being executed directly (not imported as a module) and creates an instance of DataIngestion. It then calls the initiate_data_ingestion method to start the data ingestion process
'''