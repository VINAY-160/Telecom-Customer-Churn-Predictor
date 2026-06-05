import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# ==========================================
# 1. PAGE CONFIGURATION & STYLING
# ==========================================
st.set_page_config(
    page_title="Telecom Churn AI Predictor",
    page_icon="📶",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished UI
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stAlert { border-radius: 10px; }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. MODEL LOADING (CACHED)
# ==========================================
@st.cache_resource
def load_model_data():
    """Loads the pickled model, feature list, and threshold."""
    model_path = 'best_churn_model.pkl'
    if not os.path.exists(model_path):
        return None
    
    with open(model_path, 'rb') as f:
        data = pickle.load(f)
    return data

model_data = load_model_data()

if model_data is None:
    st.error("⚠️ Model file `best_churn_model.pkl` not found! Please run the Jupyter Notebook first to generate the model.")
    st.stop()

# Extract model artifacts
model = model_data['model']
expected_features = model_data['feature_names']
threshold = model_data.get('threshold', 0.40) # Default to 0.40 if not found

# ==========================================
# 3. SIDEBAR NAVIGATION
# ==========================================
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2830/2830305.png", width=100)
st.sidebar.title("Churn AI Engine")
st.sidebar.markdown("Predict customer retention risk using advanced ensemble machine learning.")

app_mode = st.sidebar.radio("Select Mode:", ["Single Customer Analysis", "Batch Prediction (CSV)"])

st.sidebar.markdown("---")
st.sidebar.info(f"**Active Model:** Stacking Ensemble\n\n**Tuned Threshold:** {threshold:.2f}")

# ==========================================
# 4. MAIN APP LOGIC
# ==========================================

st.title("📶 Telecom Customer Churn Predictor")
st.markdown("Identify high-risk customers before they leave and optimize your retention budget.")

if app_mode == "Single Customer Analysis":
    st.header("👤 Single Customer Analysis")
    st.markdown("Enter customer details below based on the model's top features.")
    
    # Dynamically generate input fields based on the exact features the model was trained on
    # We use columns to make the form look professional
    cols = st.columns(3)
    user_inputs = {}
    
    with st.form("prediction_form"):
        for i, feature in enumerate(expected_features):
            col = cols[i % 3]
            with col:
                # Assuming Tenure and Charges are continuous, and others might be binary (0/1) due to encoding
                if 'tenure' in feature.lower() or 'charge' in feature.lower():
                    user_inputs[feature] = st.number_input(f"{feature}", value=0.0, step=1.0)
                else:
                    # For One-Hot Encoded features, provide a simple 0 or 1 selector
                    user_inputs[feature] = st.selectbox(f"{feature} (0=No, 1=Yes)", [0, 1])
        
        submit_button = st.form_submit_button(label="🔍 Analyze Churn Risk")

    if submit_button:
        # Convert dictionary to DataFrame
        input_df = pd.DataFrame([user_inputs])
        
        # Predict
        churn_prob = model.predict_proba(input_df)[:, 1][0]
        prediction = 1 if churn_prob >= threshold else 0
        
        # Display Results
        st.markdown("---")
        st.subheader("📊 Risk Assessment")
        
        res_col1, res_col2 = st.columns(2)
        
        with res_col1:
            if prediction == 1:
                st.error("🚨 **HIGH RISK: Customer is likely to churn.**")
            else:
                st.success("✅ **SAFE: Customer is likely to stay.**")
                
        with res_col2:
            st.metric(label="Churn Probability", value=f"{churn_prob:.1%}", 
                      delta="Above Threshold" if prediction == 1 else "Below Threshold",
                      delta_color="inverse")
            
        st.progress(float(churn_prob))


elif app_mode == "Batch Prediction (CSV)":
    st.header("📂 Batch Prediction")
    st.markdown("Upload a CSV file containing customer data. **Ensure the columns match the features used during model training.**")
    
    uploaded_file = st.file_uploader("Upload Customer Data (CSV)", type=["csv"])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write(f"Loaded {len(df)} customers.")
            
            # Check if all required features are present
            missing_cols = [col for col in expected_features if col not in df.columns]
            
            if missing_cols:
                st.error(f"⚠️ Missing required columns in CSV: {', '.join(missing_cols)}")
                st.info("Tip: You must upload the preprocessed dataset (after One-Hot Encoding and feature engineering).")
            else:
                # Filter down to just the required features in the correct order
                X_batch = df[expected_features]
                
                with st.spinner('Analyzing customers...'):
                    probs = model.predict_proba(X_batch)[:, 1]
                    preds = (probs >= threshold).astype(int)
                
                # Append results to the original dataframe for download
                results_df = df.copy()
                results_df['Churn_Probability'] = np.round(probs * 100, 2).astype(str) + '%'
                results_df['Risk_Level'] = pd.cut(probs, bins=[0, 0.3, threshold, 0.7, 1.0], 
                                                  labels=['Low', 'Medium', 'High', 'Critical'])
                results_df['Predicted_Churn'] = preds
                
                st.success("✅ Analysis Complete!")
                
                # Show summary metrics
                st.subheader("Insights Overview")
                m1, m2, m3 = st.columns(3)
                m1.metric("Total Customers Evaluated", len(results_df))
                m2.metric("Predicted Churners", preds.sum())
                m3.metric("Overall Churn Rate", f"{(preds.sum()/len(preds)):.1%}")
                
                # Display dataframe
                st.dataframe(results_df[['Risk_Level', 'Churn_Probability', 'Predicted_Churn'] + expected_features[:3]].head(15))
                
                # Download Button
                csv = results_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Complete Risk Report",
                    data=csv,
                    file_name='churn_risk_report.csv',
                    mime='text/csv',
                )
                
        except Exception as e:
            st.error(f"An error occurred while processing the file: {e}")