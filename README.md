# Student Performance Analysis

This repository contains code and data for analyzing student performance. The project aims to predict student performance based on various features and provide insights through data analysis.

## Project Structure

The project structure is organized as follows:

- [.ebextensions](.ebextensions): Contains configuration files for AWS Elastic Beanstalk deployment.
  - [python.config](.ebextensions/python.config): Configuration file for Python environment setup.

- [artifacts](artifacts): Directory for storing model artifacts and preprocessed data.
  - [model.pkl](artifacts/model.pkl): Serialized machine learning model.
  - [preprocessor.pkl](artifacts/preprocessor.pkl): Serialized data preprocessor.

- [data](data): Directory for storing raw and processed data.
  - [data.csv](data/data.csv): Raw data in CSV format.
  - [train.csv](data/train.csv): Training dataset.
  - [test.csv](data/test.csv): Testing dataset.
  - [stud.csv](data/stud.csv): Additional student data.

- [notebook](notebook): Directory for Jupyter notebooks related to the project.
  - [EDA STUDENT PERFORMANCE.ipynb](notebook/EDA%20STUDENT%20PERFORMANCE.ipynb): Exploratory data analysis notebook.
  - [MODEL TRAINING.ipynb](notebook/MODEL%20TRAINING.ipynb): Model training and evaluation notebook.

- [src](src): Source code directory.
  - [components](src/components): Contains reusable components and utilities.
    - [__init__.py](src/components/__init__.py): Initialization file for the components package.
    - [data_ingestion.py](src/components/data_ingestion.py): Module for data ingestion functions.
    - [data_transformation.py](src/components/data_transformation.py): Module for data transformation functions.
    - [model_trainer.py](src/components/model_trainer.py): Module for training the machine learning model.

  - [pipeline](src/pipeline): Contains pipeline modules for data processing and prediction.
    - [__init__.py](src/pipeline/__init__.py): Initialization file for the pipeline package.
    - [predict_pipeline.py](src/pipeline/predict_pipeline.py): Module for prediction pipeline.
    - [train_pipeline.py](src/pipeline/train_pipeline.py): Module for training pipeline.

  - [exception.py](src/exception.py): Custom exception classes for error handling.
  - [logger.py](src/logger.py): Logging utility functions.
  - [utils.py](src/utils.py): Utility functions used throughout the project.

- [templates](templates): Directory for HTML templates used in the Flask web application.
  - [home.html](templates/home.html): Template for the home page.
  - [index.html](templates/index.html): Template for the index page.

- [.gitignore](.gitignore): Specifies files and directories to be ignored by Git.
- [README.md](README.md): This file, providing an overview of the project structure.
- [app.py](app.py): Main Flask application file for serving the web application.
- [application.py](application.py): Wrapper file for running the Flask application.
- [requirements.txt](requirements.txt): List of Python dependencies for the project.
- [setup.py](setup.py): Setup script for the project.

## Usage

1. Install the required Python packages by running `pip install -r requirements.txt`.
2. Run the Flask application by executing `python application.py`.
3. Access the web application by visiting [http://localhost:5000](http://localhost:5000) in your web browser.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
