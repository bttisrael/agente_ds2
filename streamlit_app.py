import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

st.set_page_config(
    page_title="Supply Chain Late Delivery Risk Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_data():
    try:
        df = pd.read_parquet('df4_predictions.parquet')
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def main():
    st.title("🚚 Supply Chain Late Delivery Risk Predictor")
    st.markdown("**DataCo Global Operations** · Proactive shipment risk management for warehouse routing, carrier selection, and customer communication")
    
    df = load_data()
    
    if df is None:
        st.stop()
    
    required_cols = ['late_delivery_risk', 'prediction', 'prediction_proba']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        st.warning(f"Missing columns: {missing_cols}. Some features may not work.")
    
    tabs = st.tabs(["📊 Overview", "🎯 Actual vs Predicted", "🔍 Explore Predictions", "📈 Feature Insights"])
    
    with tabs[0]:
        render_overview(df)
    
    with tabs[1]:
        render_actual_vs_predicted(df)
    
    with tabs[2]:
        render_explore_predictions(df)
    
    with tabs[3]:
        render_feature_insights()
    
    st.markdown("---")
    st.caption("Powered by Auto Data Scientist v7 · CrewAI + Claude 3.5 Sonnet")

def render_overview(df):
    st.header("📊 Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_records = len(df)
        st.metric("Total Records", f"{total_records:,}")
    
    with col2:
        model_accuracy = 0.9745
        st.metric("Model Accuracy", f"{model_accuracy:.2%}")
    
    with col3:
        if 'prediction' in df.columns:
            most_predicted = df['prediction'].mode()[0] if len(df['prediction'].mode()) > 0 else 'N/A'
            most_predicted_pct = (df['prediction'] == most_predicted).sum() / len(df) * 100
            label = "Late" if most_predicted == 1.0 else "On Time"
            st.metric("Most Predicted Class", label, f"{most_predicted_pct:.1f}%")
        else:
            st.metric("Most Predicted Class", "N/A")
    
    with col4:
        if 'prediction_proba' in df.columns:
            avg_confidence = df['prediction_proba'].mean()
            st.metric("Avg Confidence", f"{avg_confidence:.2%}")
        else:
            st.metric("Avg Confidence", "N/A")
    
    st.markdown("---")
    
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("Prediction Distribution")
        if 'prediction' in df.columns:
            pred_counts = df['prediction'].value_counts().sort_index()
            fig = px.bar(
                x=['On Time', 'Late'],
                y=pred_counts.values,
                labels={'x': 'Prediction', 'y': 'Count'},
                color=['On Time', 'Late'],
                color_discrete_map={'On Time': '#00CC96', 'Late': '#EF553B'}
            )
            fig.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Prediction column not available")
    
    with col_right:
        st.subheader("Confidence Distribution")
        if 'prediction_proba' in df.columns:
            fig = px.histogram(
                df,
                x='prediction_proba',
                nbins=50,
                labels={'prediction_proba': 'Prediction Confidence'},
                color_discrete_sequence=['#636EFA']
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Confidence column not available")
    
    if 'delivery_status' in df.columns:
        st.subheader("Delivery Status Breakdown")
        status_counts = df['delivery_status'].value_counts()
        fig = px.pie(
            values=status_counts.values,
            names=status_counts.index,
            hole=0.4
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

def render_actual_vs_predicted(df):
    st.header("🎯 Actual vs Predicted")
    
    if 'late_delivery_risk' not in df.columns or 'prediction' not in df.columns:
        st.warning("Required columns (late_delivery_risk, prediction) not available")
        return
    
    df_valid = df.dropna(subset=['late_delivery_risk', 'prediction'])
    
    if len(df_valid) == 0:
        st.warning("No valid rows with both actual and predicted values")
        return
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Confusion Matrix")
        
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(df_valid['late_delivery_risk'], df_valid['prediction'])
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['On Time', 'Late'],
            yticklabels=['On Time', 'Late'],
            ax=ax,
            cbar_kws={'label': 'Count'}
        )
        ax.set_xlabel('Predicted', fontsize=12)
        ax.set_ylabel('Actual', fontsize=12)
        ax.set_title('Confusion Matrix', fontsize=14, pad=20)
        
        st.pyplot(fig)
        plt.close()
        
        tn, fp, fn, tp = cm.ravel()
        
        col_a, col_b, col_c, col_d = st.columns(4)
        with col_a:
            st.metric("True Negatives", f"{tn:,}")
        with col_b:
            st.metric("False Positives", f"{fp:,}")
        with col_c:
            st.metric("False Negatives", f"{fn:,}")
        with col_d:
            st.metric("True Positives", f"{tp:,}")
    
    with col2:
        st.subheader("Class Distribution Comparison")
        
        actual_counts = df_valid['late_delivery_risk'].value_counts().sort_index()
        pred_counts = df_valid['prediction'].value_counts().sort_index()
        
        comparison_df = pd.DataFrame({
            'Class': ['On Time', 'Late'] * 2,
            'Count': list(actual_counts.values) + list(pred_counts.values),
            'Type': ['Actual'] * 2 + ['Predicted'] * 2
        })
        
        fig = px.bar(
            comparison_df,
            x='Class',
            y='Count',
            color='Type',
            barmode='group',
            color_discrete_map={'Actual': '#636EFA', 'Predicted': '#EF553B'}
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        accuracy = accuracy_score(df_valid['late_delivery_risk'], df_valid['prediction'])
        precision = precision_score(df_valid['late_delivery_risk'], df_valid['prediction'], zero_division=0)
        recall = recall_score(df_valid['late_delivery_risk'], df_valid['prediction'], zero_division=0)
        f1 = f1_score(df_valid['late_delivery_risk'], df_valid['prediction'], zero_division=0)
        
        st.markdown("### Performance Metrics")
        met_col1, met_col2, met_col3, met_col4 = st.columns(4)
        with met_col1:
            st.metric("Accuracy", f"{accuracy:.2%}")
        with met_col2:
            st.metric("Precision", f"{precision:.2%}")
        with met_col3:
            st.metric("Recall", f"{recall:.2%}")
        with met_col4:
            st.metric("F1 Score", f"{f1:.2%}")

def render_explore_predictions(df):
    st.header("🔍 Explore Predictions")
    
    st.sidebar.markdown("## 🎛️ Filters")
    
    categorical_cols = ['type', 'delivery_status', 'customer_country', 'customer_segment', 'department_name', 'market', 'category_name', 'customer_state']
    categorical_cols = [col for col in categorical_cols if col in df.columns]
    
    numeric_cols = ['days_for_shipping_real', 'days_for_shipment_scheduled', 'benefit_per_order', 'sales_per_customer']
    numeric_cols = [col for col in numeric_cols if col in df.columns]
    
    df_filtered = df.copy()
    
    st.sidebar.markdown("### Categorical Filters")
    for col in categorical_cols[:5]:
        unique_vals = df[col].dropna().unique()
        if len(unique_vals) > 0 and len(unique_vals) <= 100:
            selected = st.sidebar.multiselect(
                f"{col.replace('_', ' ').title()}",
                options=sorted(unique_vals.astype(str)),
                key=f"filter_{col}"
            )
            if selected:
                df_filtered = df_filtered[df_filtered[col].astype(str).isin(selected)]
    
    st.sidebar.markdown("### Numeric Filters")
    for col in numeric_cols[:3]:
        col_min = float(df[col].min())
        col_max = float(df[col].max())
        if col_min != col_max:
            selected_range = st.sidebar.slider(
                f"{col.replace('_', ' ').title()}",
                min_value=col_min,
                max_value=col_max,
                value=(col_min, col_max),
                key=f"slider_{col}"
            )
            df_filtered = df_filtered[
                (df_filtered[col] >= selected_range[0]) & 
                (df_filtered[col] <= selected_range[1])
            ]
    
    if 'prediction' in df_filtered.columns:
        st.sidebar.markdown("### Prediction Filter")
        pred_filter = st.sidebar.radio(
            "Show predictions:",
            options=["All", "Late (1)", "On Time (0)"],
            index=0
        )
        if pred_filter == "Late (1)":
            df_filtered = df_filtered[df_filtered['prediction'] == 1.0]
        elif pred_filter == "On Time (0)":
            df_filtered = df_filtered[df_filtered['prediction'] == 0.0]
    
    st.subheader(f"Filtered Results: {len(df_filtered):,} records")
    
    col_metrics1, col_metrics2, col_metrics3 = st.columns(3)
    
    with col_metrics1:
        if 'prediction' in df_filtered.columns:
            late_pct = (df_filtered['prediction'] == 1.0).sum() / len(df_filtered) * 100 if len(df_filtered) > 0 else 0
            st.metric("Late Predictions", f"{late_pct:.1f}%")
    
    with col_metrics2:
        if 'prediction_proba' in df_filtered.columns:
            avg_conf = df_filtered['prediction_proba'].mean()
            st.metric("Avg Confidence", f"{avg_conf:.2%}")
    
    with col_metrics3:
        if 'benefit_per_order' in df_filtered.columns:
            avg_benefit = df_filtered['benefit_per_order'].mean()
            st.metric("Avg Benefit/Order", f"${avg_benefit:.2f}")
    
    display_cols = []
    priority_cols = ['prediction', 'prediction_proba', 'late_delivery_risk', 'delivery_status', 
                     'days_for_shipping_real', 'customer_city', 'customer_state', 'category_name', 
                     'department_name', 'benefit_per_order']
    
    for col in priority_cols:
        if col in df_filtered.columns:
            display_cols.append(col)
    
    for col in df_filtered.columns:
        if col not in display_cols and col not in ['customer_email', 'customer_password', 'customer_street', 'customer_id']:
            display_cols.append(col)
    
    display_cols = display_cols[:20]
    
    df_display = df_filtered[display_cols].copy()
    
    if 'prediction' in df_display.columns:
        df_display['prediction'] = df_display['prediction'].map({1.0: '🔴 Late', 0.0: '🟢 On Time'})
    
    st.dataframe(df_display, use_container_width=True, height=400)
    
    csv = df_filtered.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Filtered Data as CSV",
        data=csv,
        file_name='filtered_predictions.csv',
        mime='text/csv',
    )
    
    if len(df_filtered) > 0:
        st.markdown("---")
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            if 'customer_state' in df_filtered.columns and 'prediction' in df_filtered.columns:
                st.subheader("Late Predictions by State")
                state_pred = df_filtered[df_filtered['prediction'] == 1.0]['customer_state'].value_counts().head(10)
                fig = px.bar(
                    x=state_pred.values,
                    y=state_pred.index,
                    orientation='h',
                    labels={'x': 'Count', 'y': 'State'},
                    color_discrete_sequence=['#EF553B']
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        with chart_col2:
            if 'category_name' in df_filtered.columns:
                st.subheader("Top Categories")
                cat_counts = df_filtered['category_name'].value_counts().head