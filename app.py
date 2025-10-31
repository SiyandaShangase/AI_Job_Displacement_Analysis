import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="AI Job Displacement Dashboard", layout="wide")

st.title("🇿🇦 AI Impact on South African Workforce Dashboard")
st.write("This dashboard analyzes industry vulnerability to AI-driven disruption.")

uploaded = st.file_uploader("Upload your dataset (ai_job_market_insights.csv)", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)

    st.subheader("📊 Raw Dataset Preview")
    st.dataframe(df.head())

    industries = st.multiselect(
        "Filter industries",
        options=df["Industry"].unique(),
        default=df["Industry"].unique()
    )

    filtered_df = df[df["Industry"].isin(industries)]

    st.subheader("📈 AI Adoption vs Automation Risk")
    plt.figure()
    plt.scatter(filtered_df["AI_Adoption_Score"], filtered_df["Automation_Risk_Score"])
    plt.xlabel("AI Adoption Score")
    plt.ylabel("Automation Risk Score")
    plt.title("Industry Automation Exposure")
    st.pyplot(plt)

    st.subheader("📉 Job Growth Distribution by Industry")
    plt.figure()
    filtered_df.boxplot(column="Job_Growth_Numeric", by="Industry", rot=45)
    plt.title("Job Growth by Sector")
    plt.suptitle("")
    st.pyplot(plt)

    df["Industry_Risk_Score"] = (
        filtered_df["AI_Adoption_Score"]*0.4 +
        filtered_df["Automation_Risk_Score"]*0.4 -
        filtered_df["Job_Growth_Numeric"]*0.2
    )

    st.subheader("🔥 Industry Risk Ranking")
    st.dataframe(df[["Industry","Industry_Risk_Score"]].sort_values(by="Industry_Risk_Score", ascending=False))

    rf_df = df.dropna(subset=["AI_Adoption_Score","Automation_Risk_Score","Job_Growth_Numeric","Industry_Risk_Score"])
    X = rf_df[["AI_Adoption_Score","Automation_Risk_Score","Job_Growth_Numeric"]]
    y = (rf_df["Industry_Risk_Score"] > rf_df["Industry_Risk_Score"].median()).astype(int)

    model = RandomForestClassifier(n_estimators=300, random_state=42)
    model.fit(X, y)

    importance = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    st.subheader("📌 Feature Importance")
    plt.figure()
    plt.bar(importance["Feature"], importance["Importance"])
    plt.title("Feature Importance")
    plt.ylabel("Importance")
    st.pyplot(plt)

    st.subheader("📑 Policy Insights")
    st.write("""
### Key Findings
- High automation risk industries face most displacement.
- Job growth reduces disruption risk.
- AI adoption accelerates workforce change.

### Recommendations
✅ Government-funded reskilling  
✅ Digital training for youth & low-skilled workers  
✅ AI ethics & workforce guidelines  
✅ Support for vulnerable industries (Manufacturing, Retail, Transport)  
""")

else:
    st.info("👆 Upload your dataset to begin")
