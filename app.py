import pandas as pd
import numpy as np
import joblib
import streamlit as st

@st.cache_resource
def load_model():
    return joblib.load('xgb_model.pkl')

model = load_model() 

st.title("Real Estate Rent Prediction")
st.write("Enter the property details below to get the estimated rent.")

with st.form("user_input_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        BHK = st.number_input("BHK", 1, 10)
        Size = st.number_input("Size (sqft)", 100, 10000)
        Bathroom = st.number_input("Bathrooms", 1, 10)
        Floor_Number = st.number_input("Floor Number", 0, 50)
        Total_Floors = st.number_input("Total Floors", 1, 50)
    
    with col2:
        Area_Type = st.selectbox("Area Type", ["Super Area", "Carpet Area", "Built Area"])
        City = st.selectbox("City", ["Kolkata", "Mumbai", "Bangalore", "Delhi", "Chennai", "Hyderabad"])
        Furnishing_Status = st.selectbox("Furnishing", ["Furnished", "Unfurnished", "Semi-furnished"])
        Tenant_Preferred = st.selectbox("Tenant Type", ["Family", "Bachelor", "Bachelors/Family"])
        Point_of_Contact = st.selectbox("Contact Via", ["Owner", "Agent", "Contact Builder"])

    submit = st.form_submit_button("Predict Rent")

def predict_price(input_data):
    
    input_df = pd.DataFrame([input_data])

    prediction = model.predict(input_df)

    return np.expm1(prediction[0])

if submit:
    input_data = {
        'BHK': BHK, 'Size': Size, 'Bathroom': Bathroom,
        'Floor_Number': Floor_Number, 'Total_Floors': Total_Floors,
        'Area Type': Area_Type, 'City': City, 'Furnishing Status': Furnishing_Status,
        'Tenant Preferred': Tenant_Preferred, 'Point of Contact': Point_of_Contact,
        'Price_per_sqft': Size / Size, 
        'BHK_per_Bathroom': BHK / Bathroom
    }

    predicted_rent = predict_price(input_data)
    st.success(f"Predicted Monthly Rent: ₹{predicted_rent:,.2f}")
