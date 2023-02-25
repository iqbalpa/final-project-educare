import streamlit as st
import pandas as pd
import joblib

# Title
st.header("Paris Housing Price Prediction")

squareMeters = st.number_input("Enter squareMeters")

preds = ['squareMeters']

# If button is pressed
if st.button("Submit"):

    # Unpickle classifier
    clf = joblib.load("model.pkl")

    # Store inputs into dataframe
    X = pd.DataFrame([[squareMeters]], columns=preds)

    # Get prediction
    prediction = clf.predict(X)[0]

    # Output prediction
    st.text(f"This prediction price is a {prediction}")
