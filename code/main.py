# ************************************************************************88


# Import packages
from dash import Dash, html, callback, Output, Input, State, dcc
import math
import numpy as np
import pandas as pd
import dash_bootstrap_components as dbc
import pickle


#Import csv file
df = pd.read_csv(r'./models/A_3_predict.csv')
# df = pd.read_csv('C:\\Users\\st123\\OneDrive\\Документы\\Projects\\bootcamp\\source_code\\Dash_3\\code\\models\\A_3_predict.csv')
df.drop('Unnamed: 0', axis=1, inplace=True)
# Initialize the app - incorporate a Dash Bootstrap theme
external_stylesheets = [dbc.themes.JOURNAL]
# app = Dash(__name__, suppress_callback_exceptions=True)
app = Dash(__name__, external_stylesheets=external_stylesheets)

# Create app layout
app.layout = html.Div([
        html.H1("CAR SALES PREDICTION"),
        html.Br(),
        html.H3("Assignment3: car prices website."),
        html.Br(),
        html.H6("car price can be predicted by putting numbers in parameters. Car price prediction depends on three features, including:"),
        html.H6("Engine capacity, max power, Age of the car. Firstly, you have to fill at least one input, and then click submit to get result."),
        html.H6("Submit button just below input feilds. Please make sure that filled number are not negative."),
        html.H6("Please note, if no values are inserted for the parameters then average number of this parameter will be taken for calculation."),       
        html.Br(),
        html.H4("Definition"),
        html.Br(),
        html.H6("Engine: Nominal volume of a car engine in liters"),
        html.H6("Max Power: The ratio of max power measuring in bhp"),
        html.H6("Car Age: Total years of car in use since produced year"),
        html.Br(),
        html.Div(["Engine capacity",dbc.Input(id = "engine", type = 'number', min = 0, placeholder="please insert"),
        dbc.FormText("Please do not put nagative numbers.",color="secondary"), html.Br()]),
        html.Div(["Max Power", dbc.Input(id = "max_power", min = 0, type = 'number', placeholder ="please insert"),
        dbc.FormText("Please do not put nagative numbers.",color="secondary"), html.Br()]),
        html.Div(["Car Age", dbc.Input(id = "age", type = 'number', min = 0, placeholder="please insert"),                  
        dbc.FormText("Please do not put nagative numbers.",color="secondary"), html.Br()]),
        dbc.Button(id="submit", children="submit", color="success", className="me-1"),
        html.Div(id="output", children = '')
])
# Callback input and output
@callback(
    Output(component_id = "output", component_property = "children"),
    State(component_id = "engine", component_property = "value"),
    State(component_id = "max_power", component_property = "value"),
    State(component_id = "age", component_property = "value"),
    Input(component_id = "submit", component_property = "n_clicks"),
    prevent_initial_call=True,
    allow_duplicate=True
)

# Function for finding estimated car price
def prediction (engine, max_power, age, submit):
    if engine == None:
        engine = df["engine"].median() # Fill in maximum power if dosen't been inserted
    if max_power == None:
        max_power = df["max_power"].mean() # Fill in max_power if dosen't been inserted
    if age == None:
        age = df["age"].mean() # Fill in years driven if doesn't been inserted    
    # model = pickle.load(open("C:\Users\st123\OneDrive\Документы\Projects\bootcamp\source_code\Dash_3\code\models\Best_model_Prediction.pkl", 'rb')) # Import model
    model = pickle.load(open(r'./models/Best_model_Prediction.pkl', 'rb')) # Import model
    sample = np.array([[engine, max_power, age]]) 
    result = model.predict(sample) #Predict price
    return f"The predicted car price belongs to CLASS -> {int(result[0])}"
# if __name__ == '__main__':
    # app.run(host = '0.0.0.0', port = '80',debug=True)
    # app.run(debug=True)
    # app.run(debug=True, port="8080")
    
if __name__ == '__main__':
    # Before we run the app, download model from mlflow server
    from utils import load_mlflow
    load_mlflow(stage="Production")
    app.run(debug=True)


# ***********************************************************************************
# # Import packages
# from dash import Dash, html, Output, Input, dcc
# import dash_bootstrap_components as dbc

# # Initialize the app - incorporate a Dash Bootstrap theme
# external_stylesheets = [dbc.themes.CERULEAN]
# app = Dash(__name__, use_pages=True, external_stylesheets=[dbc.themes.BOOTSTRAP])
# from pages.home import *

# # Navigation Bar
# navbar = dbc.NavbarSimple(
#     children=[
#         dbc.NavItem(dbc.NavLink("Home", href="/")),
#         dbc.NavItem(dbc.NavLink("Model 1", href="/model1")),
#     ],
#     brand="ML2023 Dash Example A2 TUTORIAL",
#     brand_href="/",
#     color="primary",
#     dark=True,
# )


# app.layout = html.Div([
#     navbar,
#     dash.page_container
# ])

# # Run the app
# if __name__ == '__main__':
#     # Before we run the app, download model from mlflow server
#     from utils import load_mlflow
#     load_mlflow(stage="Production")
#     app.run(debug=True)

