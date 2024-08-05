import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
import plotly.graph_objects as go
import io
from PIL import Image

# Set page config
st.set_page_config(
    page_title="Churn Prediction App",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for a more professional look
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6;
    }
    .sidebar .sidebar-content {
        background: #262730;
    }
    .Widget>label {
        color: #262730;
        font-family: 'Helvetica', sans-serif;
        font-weight: bold;
    }
    .stButton>button {
        color: white;
        background-color: #4F8BF9;
        border-radius: 5px;
        height: 3em;
        width: 100%;
        font-weight: bold;
    }
    .stTextInput>div>div>input {
        color: #4F8BF9;
    }
    .stTitle {
        font-family: 'Helvetica', sans-serif;
        font-weight: bold;
        color: #262730;
    }
    .stSubheader {
        font-family: 'Helvetica', sans-serif;
        font-weight: bold;
        color: #4F8BF9;
    }
</style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    with open('model_TA_naive_bayes.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()

# Payment type mapping
payment_type_mapping = {
    'credit_card': 0,
    'boleto': 1,
    'voucher': 2,
    'debit_card': 3
}


col1, col2 = st.columns([1, 5])

with col1:
    # Load and display the logo
    logo = Image.open('logo_thesis.jpeg')  # Replace with your logo path
    st.image(logo, width=100)

with col2:
    st.title("Churn Prediction App")
st.markdown("""
<div style='background-color: #000000; padding: 10px; border-radius: 5px; margin-bottom: 20px;'>
This app predicts customer churn based on 6 key features. You can either upload a CSV file or input data manually.
</div>
""", unsafe_allow_html=True)
# Sidebar
st.sidebar.header("About")
st.sidebar.info("This web app is part of a machine learning thesis on churn prediction.")
st.sidebar.header("Features")
st.sidebar.markdown("""
- Total Order
- Payment Type
- Recency
- Frequency
- Monetary
- Avg Item Ordered
""")
# Input method selection
input_method = st.radio("Choose input method:", ("Upload CSV", "Manual Input"))
if input_method == "Upload CSV":
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            
            st.subheader("Data Preview")
            st.dataframe(data.head())
            
            st.subheader("Data Info")
            buffer = io.StringIO()
            data.info(buf=buffer)
            s = buffer.getvalue()
            st.text(s)
            
            st.subheader("Unique values in payment_type")
            st.write(data['payment_type'].unique())
            
            # Convert payment_type to numeric
            data['payment_type_numeric'] = data['payment_type'].map(payment_type_mapping)
            
            # Check for any unmapped values
            unmapped = data[data['payment_type_numeric'].isna()]
            if not unmapped.empty:
                st.warning("Some payment types couldn't be mapped:")
                st.write(unmapped['payment_type'].unique())
            
            # Fill any unmapped values with -1
            data['payment_type_numeric'] = data['payment_type_numeric'].fillna(-1).astype(int)
            
            st.subheader("Data after conversion")
            st.dataframe(data.head())
            if st.button("Predict Churn"):
                features = ['total_order', 'payment_type_numeric', 'recency', 'frequency', 'monetary', 'avg_item_ordered']
                
                X = data[features]
                
                st.subheader("Features used for prediction")
                st.dataframe(X.head())
                
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                predictions = model.predict(X_scaled)
                data['Churn Prediction'] = predictions
                data['Churn Prediction'] = data['Churn Prediction'].map({0: 'Not Churn', 1: 'Churn'})
                st.subheader("Prediction Results")
                st.dataframe(data)
                churn_summary = data['Churn Prediction'].value_counts()
                st.subheader("Churn Prediction Summary")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Customers likely to churn", churn_summary.get('Churn', 0))
                with col2:
                    st.metric("Customers likely to stay", churn_summary.get('Not Churn', 0))
                fig = go.Figure(data=[go.Pie(labels=churn_summary.index, values=churn_summary.values, hole=.3)])
                fig.update_layout(title="Churn Prediction Distribution")
                st.plotly_chart(fig)
                csv = data.to_csv(index=False)
                st.download_button(
                    label="Download Results as CSV",
                    data=csv,
                    file_name="churn_predictions.csv",
                    mime="text/csv",
                )
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.write("Please check your CSV file and ensure it contains the required columns.")
    else:
        st.info("Please upload a CSV file to get started.")
else:  # Manual Input
    st.subheader("Manual Data Input")
    
    col1, col2 = st.columns(2)
    with col1:
        total_order = st.number_input("Total Order", min_value=0)
        payment_type = st.selectbox("Payment Type", list(payment_type_mapping.keys()))
        recency = st.number_input("Recency (days)", min_value=0)
    with col2:
        frequency = st.number_input("Frequency", min_value=0)
        monetary = st.number_input("Monetary Value", min_value=0.0)
        avg_item_ordered = st.number_input("Average Items Ordered", min_value=0.0)
    if st.button("Predict Churn"):
        input_data = pd.DataFrame({
            'total_order': [total_order],
            'payment_type_numeric': [payment_type_mapping[payment_type]],
            'recency': [recency],
            'frequency': [frequency],
            'monetary': [monetary],
            'avg_item_ordered': [avg_item_ordered]
        })
        st.subheader("Input Data Preview")
        st.dataframe(input_data)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(input_data)
        prediction = model.predict(X_scaled)
        st.subheader("Prediction Result")
        result = "Churn" if prediction[0] == 1 else "Not Churn"
        st.markdown(f"<div style='background-color: {'#FFA07A' if result == 'Churn' else '#90EE90'}; padding: 20px; border-radius: 5px; text-align: center;'><h2>Churn Prediction: {result}</h2></div>", unsafe_allow_html=True)