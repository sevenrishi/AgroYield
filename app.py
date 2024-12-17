import streamlit as st
import numpy as np
import pickle
import sklearn

# Print scikit-learn version
st.write(f"Scikit-learn version: {sklearn.__version__}")

# Load models
dtr = pickle.load(open('dtr.pkl', 'rb'))
preprocessor = pickle.load(open('preprocesser.pkl', 'rb'))

# Streamlit app
st.title("Crop Yield Prediction App")
st.write("Enter the details below to predict the crop yield:")

# Input fields
year = st.text_input("Year")
average_rainfall = st.text_input("Average Rainfall (mm per year)")
pesticides_tonnes = st.text_input("Pesticides Used (tonnes)")
avg_temp = st.text_input("Average Temperature (°C)")
area = st.text_input("Area")
item = st.text_input("Crop Item")

# Prediction
if st.button("Predict"):
    try:
        # Prepare features for the model
        features = np.array([[year, average_rainfall, pesticides_tonnes, avg_temp, area, item]], dtype=object)
        transformed_features = preprocessor.transform(features)
        prediction = dtr.predict(transformed_features).reshape(1, -1)

        # Display the prediction
        st.success(f"Predicted Crop Yield: {prediction[0][0]}")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
