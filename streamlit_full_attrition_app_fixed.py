import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from scipy.stats import ks_2samp
import io
from datetime import datetime

# --- CONFIGURATION FOR MONITORING ---
ALERT_THRESHOLD_ACCURACY = 0.75
ALERT_THRESHOLD_KS = 0.1
ALERT_THRESHOLD_PREDICTION_RATE = 0.2

st.set_page_config(layout="wide")
st.title("📊 Full Employee Attrition Analysis & Model Monitoring Dashboard")
st.markdown("### 📁 Upload your dataset")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

# --- DATA UPLOAD AND EXPLORATION ---
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.success("✅ Data loaded successfully!")

    # --- DATA OVERVIEW ---
    st.subheader("📝 Data Overview")
    st.write(f"**Shape:** {data.shape[0]} rows × {data.shape[1]} columns")
    st.write("**First 5 rows:**")
    st.dataframe(data.head())

    buffer = io.StringIO()
    data.info(buf=buffer)
    s = buffer.getvalue()
    st.text("**Info:**\n" + s)

    st.write("**Missing values per column:**")
    st.write(data.isnull().sum())

    st.write("**Number of duplicate rows:**", data.duplicated().sum())

    st.write("**Summary statistics:**")
    st.write(data.describe(include='all'))

    st.markdown("---")

    # --- DATA PREPROCESSING ---
    if 'Attrition' in data.columns:
        attrition_original = data['Attrition'].copy()
    else:
        st.warning("No 'Attrition' column found in data. Please check your file.")
        st.stop()

    categorical_cols = data.select_dtypes(include=['object', 'category']).columns
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))
        label_encoders[col] = le

    numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if 'Attrition' in numerical_cols:
        numerical_cols.remove('Attrition')
    scaler = MinMaxScaler()
    data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

    if 'Monthly_Income' in data.columns and 'Performance_Rating' in data.columns:
        data['Salary_Performance_Ratio'] = data['Monthly_Income'] / (data['Performance_Rating'] + 1)
        data['Salary_Performance_Ratio'] = scaler.fit_transform(data[['Salary_Performance_Ratio']])

    if 'Years_at_Company' in data.columns:
        max_years = 40
        data['Years_at_Company_Actual'] = data['Years_at_Company'] * max_years
        bins = [0, 3, 6, 11, 21, 41]
        labels = ['0-2', '3-5', '6-10', '11-20', '21+']
        data['Tenure_Group'] = pd.cut(data['Years_at_Company_Actual'], bins=bins, labels=labels, right=False)

    # --- VISUALIZATIONS ---
    st.header("🔍 Data Visualizations")

    # Attrition Count
    st.subheader("Employee Attrition Count")
    fig, ax = plt.subplots()
    sns.countplot(x=attrition_original, ax=ax)
    ax.set_xlabel("Attrition")
    ax.set_ylabel("Count")
    st.pyplot(fig)

    # Attrition by Job Role
    if 'Job_Role' in data.columns:
        st.subheader("Attrition by Job Role")
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.countplot(x=data['Job_Role'], hue=attrition_original, ax=ax)
        if 'Job_Role' in label_encoders:
            ax.set_xticklabels(label_encoders['Job_Role'].classes_, rotation=45)
        ax.set_xlabel("Job Role")
        ax.set_ylabel("Count")
        ax.legend(title="Attrition")
        st.pyplot(fig)

    # Monthly Income vs Attrition (Plotly)
    if 'Monthly_Income' in data.columns:
        st.subheader("Monthly Income vs Attrition")
        fig = px.box(
            pd.DataFrame({'Attrition': attrition_original, 'Monthly_Income': data['Monthly_Income']}),
            x='Attrition', y='Monthly_Income', color='Attrition',
            labels={'Attrition': 'Attrition', 'Monthly_Income': 'Monthly Income'},
            color_discrete_map={'Yes': 'red', 'No': 'green'},
            title='Monthly Income vs Attrition (Interactive)'
        )
        st.plotly_chart(fig, use_container_width=True)

    # Correlation Heatmap
    st.subheader("Correlation Heatmap")
    corr = data.select_dtypes(include=['number']).corr()
    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(corr, cmap='coolwarm', annot=False, ax=ax)
    st.pyplot(fig)

    # Attrition by Tenure Group (Seaborn)
    if 'Tenure_Group' in data.columns:
        st.subheader("Attrition by Tenure Group")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.countplot(x=data['Tenure_Group'], hue=attrition_original, ax=ax)
        ax.set_xlabel("Tenure Group (Years)")
        ax.set_ylabel("Count")
        ax.legend(title="Attrition")
        st.pyplot(fig)

    # Attrition by Tenure Group (Interactive Plotly)
    if 'Tenure_Group' in data.columns:
        st.subheader("Attrition Rate by Tenure Group (Interactive Plotly)")
        if attrition_original.dtype != 'O':
            attrition_plot = attrition_original.map({1: 'Yes', 0: 'No'})
        else:
            attrition_plot = attrition_original

        fig = px.bar(
            data_frame=pd.DataFrame({'Tenure_Group': data['Tenure_Group'], 'Attrition': attrition_plot}),
            x='Tenure_Group',
            color='Attrition',
            title='Attrition Rate by Tenure Group',
            labels={'Tenure_Group': 'Tenure Group', 'Attrition': 'Attrition Status'},
            color_discrete_map={'Yes': 'red', 'No': 'green'},
            barmode='stack'
        )
        fig.update_layout(xaxis_title='Tenure Group', yaxis_title='Count', xaxis_tickangle=-45, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # --- MODEL MONITORING SECTION ---
    st.header("🛠️ Model Monitoring")

    st.sidebar.header("Monitoring Settings")
    days_to_monitor = st.sidebar.slider("Days to monitor", min_value=7, max_value=60, value=30)

    @st.cache_data(ttl=3600)
    def load_recent_predictions(days=30):
        """
        Simulate or load predictions and actuals for the last `days` days.
        Expected columns: ['date', 'employee_id', 'y_true', 'y_pred_proba']
        """
        np.random.seed(42)
        dates = pd.date_range(end=datetime.today(), periods=days)
        data_mon = []
        for date in dates:
            for i in range(100):  # 100 employees per day
                y_true = np.random.binomial(1, 0.1)
                alpha = 2 + (date - dates[0]).days * 0.02
                beta_param = 18 - (date - dates[0]).days * 0.02
                y_pred_proba = np.clip(np.random.beta(alpha, max(beta_param,1)), 0, 1)
                data_mon.append({'date': date, 'employee_id': i, 'y_true': y_true, 'y_pred_proba': y_pred_proba})
        df_mon = pd.DataFrame(data_mon)
        return df_mon

    @st.cache_data(ttl=3600)
    def load_baseline_predictions():
        np.random.seed(0)
        baseline_probs = np.random.beta(2, 18, size=1000)
        return baseline_probs

    def calculate_metrics_over_time(df, baseline_probs):
        metrics = []
        for date, group in df.groupby('date'):
            y_true = group['y_true']
            y_pred_proba = group['y_pred_proba']
            y_pred = (y_pred_proba >= 0.5).astype(int)

            accuracy = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
            try:
                auc = roc_auc_score(y_true, y_pred_proba)
            except ValueError:
                auc = np.nan

            pred_attrition_rate = y_pred.mean()
            ks_stat, _ = ks_2samp(y_pred_proba, baseline_probs)

            metrics.append({
                'date': date,
                'accuracy': accuracy,
                'f1': f1,
                'auc': auc,
                'pred_attrition_rate': pred_attrition_rate,
                'ks_stat': ks_stat
            })
        return pd.DataFrame(metrics)

    # Load monitoring data
    df_predictions = load_recent_predictions(days=days_to_monitor)
    baseline_probs = load_baseline_predictions()
    df_metrics = calculate_metrics_over_time(df_predictions, baseline_probs)

    # Show latest metrics
    latest = df_metrics.iloc[-1]
    st.subheader(f"Latest Metrics for {latest['date'].date()}")
    st.write(f"Accuracy: {latest['accuracy']:.3f}")
    st.write(f"F1 Score: {latest['f1']:.3f}")
    st.write(f"AUC: {latest['auc']:.3f}")
    st.write(f"Predicted Attrition Rate: {latest['pred_attrition_rate']:.3f}")
    st.write(f"KS Statistic: {latest['ks_stat']:.3f}")

    # Alerts
    alerts = []
    if latest['accuracy'] < ALERT_THRESHOLD_ACCURACY:
        alerts.append(f"⚠️ Accuracy dropped below threshold ({ALERT_THRESHOLD_ACCURACY}): {latest['accuracy']:.3f}")
    if latest['ks_stat'] > ALERT_THRESHOLD_KS:
        alerts.append(f"⚠️ Significant shift detected in prediction distribution (KS stat={latest['ks_stat']:.3f})")
    if latest['pred_attrition_rate'] > ALERT_THRESHOLD_PREDICTION_RATE:
        alerts.append(f"⚠️ Predicted attrition rate spike above threshold ({ALERT_THRESHOLD_PREDICTION_RATE}): {latest['pred_attrition_rate']:.3f}")

    if alerts:
        st.error("### Alerts")
        for alert in alerts:
            st.write(alert)
    else:
        st.success("No alerts triggered. Model performance is stable.")

    # Performance plots
    st.header("📊 Model Performance Over Time")
    fig_perf = go.Figure()
    fig_perf.add_trace(go.Scatter(x=df_metrics['date'], y=df_metrics['accuracy'], mode='lines+markers', name='Accuracy'))
    fig_perf.add_trace(go.Scatter(x=df_metrics['date'], y=df_metrics['f1'], mode='lines+markers', name='F1 Score'))
    fig_perf.add_trace(go.Scatter(x=df_metrics['date'], y=df_metrics['auc'], mode='lines+markers', name='AUC'))
    fig_perf.update_layout(title='Model Performance Metrics Over Time', xaxis_title='Date', yaxis_title='Metric Value', yaxis=dict(range=[0,1]), legend=dict(x=0, y=1))
    st.plotly_chart(fig_perf, use_container_width=True)

    st.header("📊 Prediction Distribution & Attrition Rate")
    fig_pred = go.Figure()
    fig_pred.add_trace(go.Scatter(x=df_metrics['date'], y=df_metrics['pred_attrition_rate'], mode='lines+markers', name='Predicted Attrition Rate'))
    fig_pred.add_trace(go.Scatter(x=df_metrics['date'], y=df_metrics['ks_stat'], mode='lines+markers', name='KS Statistic'))
    fig_pred.update_layout(title='Predicted Attrition Rate & KS Statistic Over Time', xaxis_title='Date', yaxis_title='Value')
    st.plotly_chart(fig_pred, use_container_width=True)

else:
    st.info("Please upload a CSV file to start the analysis.")
