# Bangalore Home Price Prediction

## Project Overview

This project aims to predict home prices in Bangalore based on various input features such as square footage, number of bedrooms (BHK), number of bathrooms, and location. The prediction model is implemented using a machine learning algorithm, and the application is built using Flask for the backend and HTML, CSS, and JavaScript for the frontend.

## Project Structure

The project is structured into the following components:

1. **Machine Learning (ml) Code:**
   - The code in `ml_code.py` handles data cleaning, preprocessing, outlier removal, and model training using linear regression. It also includes functions for data visualization and model evaluation.

2. **Web Application (frontend):**
   - HTML (`index.html`): The main HTML file for the user interface.
   - CSS (`app.css`): Stylesheet for styling the user interface.
   - JavaScript (`app.js`): Frontend logic for handling user input and making requests to the backend.

3. **Backend (backend):**
   - Flask (`app.py`): The Flask application that serves the machine learning model via REST API.
   - Util (`util.py`): Utility functions for loading the model and providing location information.
   - Artifacts Folder (`artifacts`): Contains the saved model (`house_price_model.pickle`) and column information (`columns.json`).

## Running the Application

1. **Machine Learning Model:**
   - Run `ml_code.py` to preprocess the data, train the model, and save the artifacts.

2. **Flask Backend:**
   - Run `app.py` to start the Flask application. This will serve the machine learning model on `http://127.0.0.1:5000/`.

3. **Frontend:**
   - Open `index.html` in a web browser. The user interface allows users to input details and get an estimated home price.

## REST API Endpoints

1. `/get_location` (GET): Returns a list of available locations.
2. `/predict_home_price` (POST): Accepts parameters (total square footage, location, BHK, and bathrooms) and returns the estimated home price.
