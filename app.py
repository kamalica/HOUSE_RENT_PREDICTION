import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import streamlit as st
import matplotlib.pyplot as plt

# Load Dataset
df = pd.read_csv('House_Rent_Dataset.csv')
df = df.drop(['Posted On', 'Area Locality'], axis=1)

# Process Floor Column
def process_floor(floor_str):
    try:
        if 'out of' in floor_str:
            current, total = floor_str.split(' out of ')
            return int(current), int(total)
        elif 'basement' in floor_str.lower():
            return -1, np.nan
        elif 'ground' in floor_str.lower():
            return 0, np.nan
        else:
            return np.nan, np.nan
    except:
        return np.nan, np.nan

df[['Floor_Number', 'Total_Floors']] = df['Floor'].apply(lambda x: pd.Series(process_floor(x)))
df = df.dropna(subset=['Floor_Number', 'Total_Floors']).drop('Floor', axis=1)

# Feature Engineering
df['Price_per_sqft'] = df['Rent'] / df['Size']
df['BHK_per_Bathroom'] = df['BHK'] / df['Bathroom']

# Feature Selection
categorical_features = ['Area Type', 'City', 'Furnishing Status', 'Tenant Preferred', 'Point of Contact']
numerical_features = ['BHK', 'Size', 'Bathroom', 'Floor_Number', 'Total_Floors', 'Price_per_sqft', 'BHK_per_Bathroom']

# Preprocessing Pipeline
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

# Model Setup
xgb_model = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6, random_state=42)

xgb_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', xgb_model)
])

# Data Preparation
df['Rent'] = np.log1p(df['Rent'])
X = df.drop('Rent', axis=1)
y = df['Rent']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
xgb_pipeline.fit(X_train, y_train)
joblib.dump(xgb_pipeline, 'xgb_model.pkl')

# Streamlit App
st.title("Real Estate Price Prediction")
st.write("Enter the property details:")

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

def predict_price(BHK, Size, Bathroom, Floor_Number, Total_Floors,
                  Area_Type, City, Furnishing_Status, Tenant_Preferred, Point_of_Contact):
    model = joblib.load('xgb_model.pkl')
    data = pd.DataFrame({
        'BHK': [BHK], 'Size': [Size], 'Bathroom': [Bathroom],
        'Floor_Number': [Floor_Number], 'Total_Floors': [Total_Floors],
        'Area Type': [Area_Type], 'City': [City], 'Furnishing Status': [Furnishing_Status],
        'Tenant Preferred': [Tenant_Preferred], 'Point of Contact': [Point_of_Contact],
    })
    data['Price_per_sqft'] = data['Size'] / Size  # Prevent missing values
    data['BHK_per_Bathroom'] = data['BHK'] / Bathroom
    prediction = model.predict(data)
    return np.expm1(prediction[0])

if st.button("Predict Rent"):
    predicted = predict_price(BHK, Size, Bathroom, Floor_Number, Total_Floors,
                              Area_Type, City, Furnishing_Status, Tenant_Preferred, Point_of_Contact)
    st.success(f"Predicted Monthly Rent: ₹{predicted:,.2f}")
