""" 
The code you provided is a Flask web application that uses a predictive pipeline to make predictions based on user input. Here's a summary of how the code works:

The code imports necessary dependencies including Flask, request, render_template, numpy, pandas, StandardScaler, CustomData, and PredictPipeline.
An instance of the Flask application is created and assigned to the variable application. Then, app is assigned the value of application.
The code defines a route for the home page (/) using the @app.route decorator. When a user accesses the home page, the index() function is executed, which renders the index.html template.
Another route is defined for the /predictdata URL, which accepts both GET and POST requests. If the request method is GET, the predict_datapoint() function renders the home.html template.
If the request method is POST, the function retrieves the form data submitted by the user. It creates a CustomData object using the form data, which represents a custom data point for prediction.
The CustomData object is converted into a pandas DataFrame, and some print statements are used for debugging purposes.
An instance of the PredictPipeline class is created, and the predict() method is called with the DataFrame as an argument to obtain the prediction results.
The prediction results are then rendered back to the home.html template as the results variable.
The Flask application is run with the app.run() method, making it accessible on the host IP address 0.0.0.0.
Overall, this code sets up a Flask web application that allows users to input data and obtain predictions using a predictive pipeline. It uses HTML templates for rendering the web pages and interacts with the CustomData and PredictPipeline classes to handle the data and prediction processes, respectively.
"""

from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
application=Flask(__name__)
app = application
 
 #route for home page
 
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('writing_score')),
            writing_score=float(request.form.get('reading_score'))
        )
        pred_df=data.get_data_as_data_frame()
        print(pred_df)
        
        predict_pipeline=PredictPipeline() 
        results=predict_pipeline.predict(pred_df)
        return render_template('home.html', results=results[0])
    
if __name__=="__main__":
    app.run(host="0.0.0.0", debug=True)        
    
    
    
""" from flask import Flask, request, render_template: Imports the necessary modules from the Flask framework to create a web application, handle HTTP requests, and render HTML templates.

import numpy as np: Imports the NumPy library for numerical computations.

import pandas as pd: Imports the pandas library for data manipulation and analysis.

from sklearn.preprocessing import StandardScaler: Imports the StandardScaler class from the scikit-learn library, which is used for standardizing data.

from src.pipeline.predict_pipeline import CustomData, PredictPipeline: Imports the CustomData and PredictPipeline classes from the predict_pipeline module within the src.pipeline package.

application = Flask(__name__): Creates a Flask application instance with the name __name__, which represents the current module.

app = application: Assigns the Flask application instance to the variable app for convenience.

@app.route('/'): Defines a route for the home page (/). When a user accesses this URL, the function immediately below will be executed.

def index():: Defines the index function, which will be executed when the home page is accessed.

return render_template('index.html'): Returns the rendered HTML template called index.html, which will be displayed to the user when accessing the home page.

@app.route('/predictdata', methods=['GET', 'POST']): Defines a route for the /predictdata URL. This route accepts both GET and POST requests.

def predict_datapoint():: Defines the predict_datapoint function, which will be executed when the /predictdata URL is accessed.

if request.method == 'GET':: Checks if the request method is GET.

return render_template('home.html'): If the request method is GET, the function returns the rendered HTML template called home.html.

else:: If the request method is not GET (i.e., it's a POST request), the code inside this block will be executed.

data = CustomData(...): Creates an instance of the CustomData class, passing in the form data submitted by the user as arguments. This represents a custom data point for prediction.

pred_df = data.get_data_as_data_frame(): Converts the CustomData object into a pandas DataFrame by calling its get_data_as_data_frame() method.

print(pred_df): Prints the DataFrame containing the user's data for debugging purposes.

print("Before Prediction"): Prints a debugging message.

predict_pipeline = PredictPipeline(): Creates an instance of the PredictPipeline class.

print("Mid Prediction"): Prints a debugging message.

results = predict_pipeline.predict(pred_df): Calls the predict() method of the PredictPipeline instance to obtain the prediction results for the user's data.

print("after Prediction"): Prints a debugging message.

return render_template('home.html', results=results[0]): Returns the rendered HTML template called home.html, passing the prediction results as the results variable to be displayed in the template.

if __name__ == "__main__":: Checks if the current module is the main module (i.e., not imported as a module).

app.run(host="0.0.0.0"): Starts the Flask application, making it accessible on the host IP address 0.0.0.0.

Overall, this code sets up a Flask web application with two routes: the home page (/) and the /predictdata route.

The home page route (/) is associated with the index() function, which renders and returns the index.html template.

The /predictdata route is associated with the predict_datapoint() function. If the request method is GET, it renders and returns the home.html template. If the method is POST, it retrieves the form data submitted by the user, creates a CustomData object with the provided data, converts it to a pandas DataFrame, and passes it to the PredictPipeline class to obtain prediction results. The results are then passed to the home.html template for display.

The if __name__ == "__main__": block ensures that the Flask application is only run when the script is executed directly (not imported as a module). The application runs on the host IP address 0.0.0.0.

Overall, this code sets up a web application using Flask, defines routes for different URLs, handles form submissions, and performs predictions using a custom data pipeline. """