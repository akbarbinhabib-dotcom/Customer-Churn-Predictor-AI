import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.tree import DecisionTreeClassifier

MODEL_FILE = "churn_model.joblib"

# ---------------- MODEL ----------------
def get_trained_model():
    if os.path.exists(MODEL_FILE):
        return joblib.load(MODEL_FILE)
    else:
        data = {
            'Monthly_Revenue': [200, 50, 300, 40, 500, 30, 250, 45, 600, 20],
            'Customer_Tenure': [12, 2, 24, 1, 36, 1, 15, 2, 48, 1],
            'Complaints': [0, 3, 0, 4, 1, 5, 0, 3, 0, 6],
            'Region_North': [1, 0, 1, 0, 0, 0, 1, 0, 0, 0],
            'Region_South': [0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
            'Region_West': [0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
            'Churn': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        }

        df = pd.DataFrame(data)
        X = df.drop("Churn", axis=1)
        y = df["Churn"]

        model = DecisionTreeClassifier(random_state=42)
        model.fit(X, y)

        joblib.dump(model, MODEL_FILE)
        return model

model = get_trained_model()

# ---------------- UI ----------------
st.set_page_config(page_title="Churn Guard AI", page_icon="🛡️")

st.title("🛡️ Churn Guard AI (Business Dashboard)")
st.write("Identify high-risk customers and take action.")

tab1, tab2 = st.tabs(["Single Prediction", "Bulk Analysis (CSV)"])

# ---------------- SINGLE ----------------
with tab1:
    st.sidebar.header("Customer Profile")

    revenue = st.sidebar.number_input("Monthly Revenue ($)", min_value=0, value=100)
    tenure = st.sidebar.slider("Tenure (Months)", 0, 60, 12)
    complaints = st.sidebar.selectbox("Complaints", [0,1,2,3,4,5,6])
    region = st.sidebar.radio("Region", ["North", "South", "West", "East"])

    r_north = 1 if region == "North" else 0
    r_south = 1 if region == "South" else 0
    r_west = 1 if region == "West" else 0

    if st.button("Analyze Risk", type="primary"):
        input_df = pd.DataFrame([[revenue, tenure, complaints, r_north, r_south, r_west]],
            columns=['Monthly_Revenue','Customer_Tenure','Complaints','Region_North','Region_South','Region_West'])

        pred = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1]

        st.metric("Churn Probability", f"{prob*100:.2f}%")

        if prob > 0.7:
            st.error("HIGH RISK")
            st.warning("Action: Offer discount / retention call")
        elif prob > 0.4:
            st.warning("MEDIUM RISK")
            st.info("Action: Engage with email campaign")
        else:
            st.success("LOW RISK")
            st.info("Action: Upsell premium services")

# ---------------- BULK ----------------
with tab2:
    st.subheader("Upload Customer Data (CSV)")

    file = st.file_uploader("Upload CSV", type=["csv"])

    if file:
        df = pd.read_csv(file)

        try:
            preds = model.predict(df)
            probs = model.predict_proba(df)[:,1]

            df["Churn_Risk"] = probs
            df["Prediction"] = preds

            # Segmentation
            def segment(p):
                if p > 0.7:
                    return "High"
                elif p > 0.4:
                    return "Medium"
                else:
                    return "Low"

            df["Segment"] = df["Churn_Risk"].apply(segment)

            st.dataframe(df)

            st.subheader("Summary")
            st.write(df["Segment"].value_counts())

            st.download_button("Download Results", df.to_csv(index=False), "churn_results.csv")

        except Exception as e:
            st.error("Invalid file format. Please match training features.")
