import streamlit as st
import pickle
import numpy as np

# Load the trained model
with open('breastcancer.pkl', 'rb') as file:
    model = pickle.load(file)
   
y_pred = model.predict([[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30] ])
print(y_pred)

# Streamlit app
st.title("Breast Cancer Detection")

# Input fields
radius_mean = st.number_input("Radius Mean", step=0.01)
texture_mean = st.number_input("Texture Mean", step=0.01)
perimeter_mean = st.number_input("Perimeter Mean", step=0.01)
area_mean = st.number_input("Area Mean", step=0.01)
smoothness_mean = st.number_input("Smoothness Mean", step=0.01)
compactness_mean = st.number_input("Compactness Mean", step=0.01)
concavity_mean = st.number_input("Concavity Mean", step=0.01)
concave_points_mean = st.number_input("Concave Points Mean", step=0.01)
symmetry_mean = st.number_input("Symmetry Mean", step=0.01)
fractal_dimension_mean = st.number_input("Fractal Dimension Mean", step=0.01)
radius_se = st.number_input("Radius SE", step=0.01)
texture_se = st.number_input("Texture SE", step=0.01)
perimeter_se = st.number_input("Perimeter SE", step=0.01)
area_se = st.number_input("Area SE", step=0.01)
smoothness_se = st.number_input("Smoothness SE", step=0.01)
compactness_se = st.number_input("Compactness SE", step=0.01)
concavity_se = st.number_input("Concavity SE", step=0.01)
concave_points_se = st.number_input("Concave Points SE", step=0.01)
symmetry_se = st.number_input("Symmetry SE", step=0.01)
fractal_dimension_se = st.number_input("Fractal Dimension SE", step=0.01)
radius_worst = st.number_input("Radius Worst", step=0.01)
texture_worst = st.number_input("Texture Worst", step=0.01)
perimeter_worst = st.number_input("Perimeter Worst", step=0.01)
area_worst = st.number_input("Area Worst", step=0.01)
smoothness_worst = st.number_input("Smoothness Worst", step=0.01)
compactness_worst = st.number_input("Compactness Worst", step=0.01)
concavity_worst = st.number_input("Concavity Worst", step=0.01)
concave_points_worst = st.number_input("Concave Points Worst", step=0.01)
symmetry_worst = st.number_input("Symmetry Worst", step=0.01)
fractal_dimension_worst = st.number_input("Fractal Dimension Worst", step=0.01)

# Make prediction
if st.button("Predict"):
    input_data = np.array([[
        radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean,
        compactness_mean, concavity_mean, concave_points_mean, symmetry_mean,
        fractal_dimension_mean, radius_se, texture_se, perimeter_se, area_se,
        smoothness_se, compactness_se, concavity_se, concave_points_se,
        symmetry_se, fractal_dimension_se, radius_worst, texture_worst,
        perimeter_worst, area_worst, smoothness_worst, compactness_worst,
        concavity_worst, concave_points_worst, symmetry_worst,
        fractal_dimension_worst
    ]])
    prediction = model.predict(input_data)
    if prediction[0] == 0:
        st.success("The tumor is benign.")
    else:
        st.error("The tumor is malignant.")