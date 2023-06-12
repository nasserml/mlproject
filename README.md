## End to End Machine Learning Project
# Student Performance Analysis

This repository contains code and data for analyzing student performance. The project aims to predict student performance based on various features and provide insights through data analysis.

## Project Structure

The project structure is organized as follows:

- `.ebextensions`: Contains configuration files for AWS Elastic Beanstalk deployment.
  - `python.config`: Configuration file for Python environment setup.

- `artifacts`: Directory for storing model artifacts and preprocessed data.
  - `model.pkl`: Serialized machine learning model.
  - `preprocessor.pkl`: Serialized data preprocessor.

- `data`: Directory for storing raw and processed data.
  - `data.csv`: Raw data in CSV format.
  - `train.csv`: Training dataset.
  - `test.csv`: Testing dataset.
  - `stud.csv`: Additional student data.

- `notebook`: Directory for Jupyter notebooks related to the project.
  - `EDA STUDENT PERFORMANCE.ipynb`: Exploratory data analysis notebook.
  - `MODEL TRAINING.ipynb`: Model training and evaluation notebook.

- `src`: Source code directory.
  - `components`: Contains reusable components and utilities.
    - `__init__.py`: Initialization file for the components package.
    - `data_ingestion.py`: Module for data ingestion functions.
    - `data_transformation.py`: Module for data transformation functions.
    - `model_trainer.py`: Module for training the machine learning model.

  - `pipeline`: Contains pipeline modules for data processing and prediction.
    - `__init__.py`: Initialization file for the pipeline package.
    - `predict_pipeline.py`: Module for prediction pipeline.
    - `train_pipeline.py`: Module for training pipeline.

  - `exception.py`: Custom exception classes for error handling.
  - `logger.py`: Logging utility functions.
  - `utils.py`: Utility functions used throughout the project.

- `templates`: Directory for HTML templates used in the Flask web application.
  - `home.html`: Template for the home page.
  - `index.html`: Template for the index page.

- `.gitignore`: Specifies files and directories to be ignored by Git.
- `README.md`: This file, providing an overview of the project structure.
- `app.py`: Main Flask application file for serving the web application.
- `application.py`: Wrapper file for running the Flask application.
- `requirements.txt`: List of Python dependencies for the project.
- `setup.py`: Setup script for the project.

## Usage

1. Install the required Python packages by running `pip install -r requirements.txt`.
2. Run the Flask application by executing `python application.py`.
3. Access the web application by visiting `http://localhost:5000` in your web browser.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
