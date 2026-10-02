import streamlit as st
import numpy as np
import pandas as pd
import joblib


# Load trained model and preprocessing objects
model = joblib.load("Best_Model.pkl")
scaler = joblib.load("scaler.pkl")
pca = joblib.load("pca.pkl")
encoder = joblib.load("encoder.pkl")


st.title("ML Prediction App")

st.write("Enter the required details below")


# Numerical features
numeric_cols = [
    'Inventory_Turnover',
    'Supplier_Rating',
    'Storage_Capacity',
    'Units_Sold',
    'Monthly_Demand',
    'Seasonal_Demand_Index',
    'Shipping_Cost_USD',
    'Delivery_Time_Days',
    'On_Time_Delivery_Rate',
    'Product_Cost_USD',
    'Selling_Price_USD',
    'Revenue_USD',
    'Profit_USD',
    'Month',
    'Year',
    'Demand_Forecast',
    'Required_Stock',
    'Stock_Gap'
]


# Take numerical inputs
num_values = []

for col in numeric_cols:
    value = st.number_input(col, value=0.0)
    num_values.append(value)


# Categorical columns
categorical_cols = [
    'Product_Category',
    'Brand',
    'Warehouse_Location',
    'Transportation_Mode'
]


# Take categorical inputs
cat_values = []

for i, col in enumerate(categorical_cols):
    options = list(encoder.categories_[i])
    value = st.selectbox(col, options)
    cat_values.append(value)


# Prediction
if st.button("Predict"):

    # Numerical data
    num_data = np.array(num_values).reshape(1, -1)

    # Scale numerical data
    num_scaled = scaler.transform(num_data)

    # Apply PCA
    num_pca = pca.transform(num_scaled)

    # Categorical data
    cat_data = pd.DataFrame(
        [cat_values],
        columns=categorical_cols
    )

    cat_encoded = encoder.transform(cat_data)

    # Combine PCA + encoded categorical features
    final_data = np.hstack((num_pca, cat_encoded))

    # Prediction
    st.write("PCA shape:", num_pca.shape)
    st.write("Encoded shape:", cat_encoded.shape)
    st.write("Final shape:", final_data.shape)
    st.write("Model expects:", model.n_features_in_)

    prediction = model.predict(final_data)

    st.write("Prediction:", prediction[0])

    if prediction[0] == 1:
        st.error("Stockout Risk: YES")
    else:
        st.success("Stockout Risk: NO")