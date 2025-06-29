# Moneyball AI Scoring App - Tcl-Free Version

import matplotlib
matplotlib.use("Agg")  # ✅ Use headless backend to avoid Tcl/Tk errors

import streamlit as st
import pandas as pd
import numpy as np
import re
import shap
import io
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# --- Page Configuration ---
st.set_page_config(
    page_title="Moneyball AI Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Helper Functions ---
def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    st.components.v1.html(shap_html, height=height)

@st.cache_data
def create_sample_file():
    sample_data = {
        'Company Name': ['Innovate Inc.', 'Data Systems', 'Retail Corp', 'Finance Solutions', 'HealthFirst', 'Legacy Logistics', 'Quantum Computing', 'Global Goods'],
        'Industry': ['Technology', 'Technology', 'E-commerce', 'Finance', 'Healthcare', 'Logistics', 'Technology', 'Retail'],
        'Job Title': ['C-Level', 'Director of Ops', 'Owner', 'VP Finance', 'Manager', 'Analyst', 'Head of Research', 'Owner'],
        'Lead Source': ['Organic', 'Referral', 'Paid', 'Organic', 'Referral', 'Paid', 'Organic', 'Referral'],
        'Employee Count': ['100-500', '501-1000', '1-10', '51-100', '10-50', '1000+', '10-50', '1-10'],
        'Email Address': ['contact@innovate.com', 'info@datasys.com', 'owner@retail.co', 'vp@finance-sol.com', 'manager-health.org', 'test@legacy.com', 'info@quantum.dev', ''],
        'Phone Number': ['1234567890', '2345678901', '3456789012', '4567890123', '5678901234', '1112223333', '', '9998887777']
    }
    df = pd.DataFrame(sample_data)
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Leads')
    return output.getvalue()

def to_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Scored_Leads')
    return output.getvalue()

# --- Validators and AI Scoring ---
def is_valid_email(email):
    if not isinstance(email, str): return False
    return bool(re.match(r"(^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$)", email.strip()))

def is_valid_phone(phone):
    if not isinstance(phone, (str, int, float)): return False
    phone_str = re.sub(r'\D', '', str(phone))
    return 10 <= len(phone_str) <= 15

def calculate_data_quality(row):
    score = 100
    if 'Email Address' in row.index and not is_valid_email(row['Email Address']): score -= 30
    if 'Phone Number' in row.index and not is_valid_phone(row['Phone Number']): score -= 30
    for col in row.index:
        if pd.isna(row[col]) or str(row[col]).strip() == '': score -= 5
    return max(score, 0)

def simulate_ai_scoring_api(lead_data: dict, feature_columns: list):
    base_score = 40.0
    feature_contributions = {feature: 0.0 for feature in feature_columns}
    for feature_name, value in lead_data.items():
        if pd.isna(value): continue
        val_str = str(value).lower()
        contribution = 0.0
        if any(term in val_str for term in ['e-commerce', 'retail', 'technology']): contribution += 20.0
        if 'finance' in val_str: contribution += 15.0
        if any(term in val_str for term in ['director', 'vp', 'c-level', 'head']): contribution += 25.0
        if 'owner' in val_str: contribution += 30.0
        if '1-10' in val_str: contribution -= 15.0
        if 'referral' in val_str: contribution += 15.0
        feature_contributions[feature_name] = contribution / 100.0
    total_contribution_score = sum(val * 100 for val in feature_contributions.values())
    final_score = max(0, min(100, base_score + total_contribution_score))
    return {"score": int(final_score), "base_value": base_score, "shap_values": list(feature_contributions.values())}

@st.cache_data(show_spinner="Scoring leads with demonstration AI...")
def get_scores_from_ai(df, feature_columns):
    api_results = df.apply(lambda row: simulate_ai_scoring_api(row[feature_columns].to_dict(), feature_columns), axis=1)
    results_df = pd.json_normalize(api_results)
    return pd.concat([df.reset_index(drop=True), results_df], axis=1)

# --- Sidebar UI ---
with st.sidebar:
    st.header("📄 Upload Leads")
    uploaded_file = st.file_uploader("Upload a leads file to score", type=['xlsx', 'csv', 'txt'], key="file_uploader")
    st.download_button(label="Download Sample Excel", data=create_sample_file(), file_name="sample_leads_to_score.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    st.markdown("---")
    st.header("⚙️ Configuration")

    st.markdown("#### Priority Thresholds")
    medium_threshold = st.slider("Minimum score for 'Medium' priority", 0, 100, 40)
    high_threshold = st.slider("Minimum score for 'High' priority", 0, 100, 75)
    if medium_threshold >= high_threshold:
        st.warning("Medium threshold should be lower than High threshold.")
    st.info("This app uses a built-in mock AI service for demonstration.")
    st.markdown("#### ⚠️ Data Upload Guidelines")
    st.markdown("- Only .xlsx, .csv, or .txt files\n- First row must be headers\n- No empty columns or rows")

# --- Main Logic ---
st.title("🎯 Moneyball AI Lead Scoring Dashboard")

if uploaded_file:
    try:
        if 'scored_df' not in st.session_state or st.session_state.file_name != uploaded_file.name:
            if uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file)
            else:
                df = pd.read_csv(uploaded_file)
            df.dropna(how='all', axis=0, inplace=True)
            df.dropna(how='all', axis=1, inplace=True)
            st.session_state.original_df = df
            st.session_state.file_name = uploaded_file.name
            if 'scored_df' in st.session_state:
                del st.session_state.scored_df

        all_columns = st.session_state.original_df.columns.tolist()
        with st.sidebar:
            st.markdown("### 🔧 Feature Selection")
            default_features = [col for col in all_columns if 'email' not in col.lower() and 'phone' not in col.lower() and 'name' not in col.lower()]
            selected_features = st.multiselect("Select input features for AI scoring", options=all_columns, default=default_features)

            if st.button("🚀 Run AI Scoring", use_container_width=True):
                if not selected_features:
                    st.warning("Please select at least one feature to run the scoring.")
                else:
                    st.session_state.scored_df = get_scores_from_ai(st.session_state.original_df.copy(), selected_features)
    except Exception as e:
        st.error(f"An error occurred: {e}")
        if 'scored_df' in st.session_state:
            del st.session_state.scored_df
else:
    st.info("👋 Welcome! Please upload a lead file using the sidebar to get started.")

# --- Results Display ---
if 'scored_df' in st.session_state:
    scored_df = st.session_state.scored_df.copy()
    bins = [-1, medium_threshold, high_threshold, 101]
    labels = ["Low", "Medium", "High"]
    scored_df["Priority"] = pd.cut(scored_df["score"], bins=bins, labels=labels, right=False)

    if "Email Address" in scored_df.columns:
        scored_df["Email Valid"] = scored_df["Email Address"].apply(is_valid_email)
    if "Phone Number" in scored_df.columns:
        scored_df["Phone Valid"] = scored_df["Phone Number"].apply(is_valid_phone)
    scored_df["Data Quality"] = scored_df[st.session_state.original_df.columns].apply(calculate_data_quality, axis=1)

    st.success("✅ Scoring complete!")
    st.markdown("### Actionable Lead List")
    priority_filter = st.multiselect("Filter by Priority:", options=labels, default=labels)
    filtered_df = scored_df[scored_df['Priority'].isin(priority_filter)]

    st.download_button(
        label="📥 Download Filtered Leads (Excel)",
        data=to_excel(filtered_df),
        file_name="scored_leads.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    st.data_editor(filtered_df, use_container_width=True, height=350)

    with st.expander("📊 Charts & Insights Dashboard", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🥇 Lead Priority Distribution")
            priority_counts = scored_df['Priority'].value_counts()
            fig_pie = go.Figure(data=[go.Pie(labels=priority_counts.index, values=priority_counts.values, hole=0.5)])
            st.plotly_chart(fig_pie, use_container_width=True)

        with col2:
            st.subheader("🌐 Avg. Score by Lead Source")
            if 'Lead Source' in scored_df.columns:
                avg_scores = scored_df.groupby('Lead Source')['score'].mean().reset_index().sort_values('score', ascending=False)
                fig_avg = px.bar(avg_scores, x='Lead Source', y='score', color='score')
                st.plotly_chart(fig_avg, use_container_width=True)

        st.subheader("🧩 Strategic Lead Segments")
        kmeans = KMeans(n_clusters=4, random_state=42, n_init='auto')
        scored_df['Segment'] = kmeans.fit_predict(scored_df[['score', 'Data Quality']])
        centers = pd.DataFrame(kmeans.cluster_centers_, columns=['score', 'Data Quality'])
        centers['Segment Name'] = centers.apply(lambda row:
            'Prime Opportunities' if row['score'] > 70 and row['Data Quality'] > 70 else
            'High Potential, Needs Cleanup' if row['score'] > 70 else
            'Good Data, Lower Priority' if row['Data Quality'] > 70 else
            'Low Priority & Quality', axis=1)
        segment_map = dict(enumerate(centers['Segment Name']))
        scored_df['Segment Name'] = scored_df['Segment'].map(segment_map)

        fig_segment = px.scatter(
            scored_df, x="Data Quality", y="score", color="Segment Name",
            hover_data=st.session_state.original_df.columns,
            title="Customer Segments by Score vs. Data Quality"
        )
        st.plotly_chart(fig_segment, use_container_width=True)

        st.subheader("🔬 Feature Contribution Analysis")
        st.markdown("#### Overall Feature Importance")
        shap_values_df = pd.DataFrame(scored_df['shap_values'].tolist(), columns=selected_features)
        shap_values_obj = shap.Explanation(
            values=shap_values_df.values,
            base_values=np.array([40.0]*len(scored_df)),
            data=scored_df[selected_features],
            feature_names=selected_features
        )
        fig, ax = plt.subplots()
        shap.summary_plot(shap_values_obj, plot_type='bar', show=False)
        st.pyplot(fig)

        st.markdown("#### Individual Lead Breakdown")
        selected_lead_idx = st.selectbox("Select a lead to analyze:", scored_df.index, format_func=lambda x: f"Lead Index {x}")
        if selected_lead_idx is not None:
            lead = scored_df.loc[selected_lead_idx]
            explanation = shap.Explanation(
                values=np.array(lead['shap_values']),
                base_values=lead['base_value'],
                data=lead[selected_features].values,
                feature_names=selected_features
            )
            p = shap.force_plot(explanation.base_values, explanation.values, explanation.data, feature_names=selected_features)
            st_shap(p, height=400)
