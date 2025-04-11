import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Configure Streamlit layout
st.set_page_config(page_title="Student EDA", layout="wide")

# ---------------- ENTRY PAGE ----------------
st.title("📊 Student Academic Dashboard")
st.markdown("Welcome! Upload your dataset and choose an option from the sidebar.")

# ----------- FILE UPLOADER ------------------
uploaded_file = st.file_uploader("Upload your student dataset (.csv)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # ---------------- DATA CLEANING ----------------
    st.session_state['cleaning_log'] = {}

    # 1. Strip column names
    df.columns = df.columns.str.strip()
    st.session_state['cleaning_log']["Column name cleanup"] = "Whitespace stripped from column names"

    # 2. Drop duplicates
    initial_shape = df.shape
    df.drop_duplicates(inplace=True)
    duplicates_dropped = initial_shape[0] - df.shape[0]
    st.session_state['cleaning_log']["Duplicates removed"] = f"{duplicates_dropped} duplicate rows removed"

    # 3. Handle missing values
    missing_info_before = df.isnull().sum()
    total_missing_before = missing_info_before.sum()

    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            df[col].fillna(df[col].median(), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)

    missing_info_after = df.isnull().sum()
    total_missing_after = missing_info_after.sum()
    st.session_state['cleaning_log']["Missing values handled"] = (
        f"{total_missing_before} missing values filled using median/mode"
    )

    # ---------------- SIDEBAR NAVIGATION ----------------
    st.sidebar.title("📌 Navigation")
    option = st.sidebar.radio(
        "Go to",
        ("📋 View Data", "🧹 Data Cleaning Report", "📈 Summary & Correlations", "📊 Visualizations")
    )

    # ---------- OPTION 1: VIEW DATA ----------
    if option == "📋 View Data":
        st.subheader("🔍 Raw Dataset Preview")
        st.dataframe(df)

        st.subheader("🧮 Summary Statistics")
        st.write(df.describe(include='all'))

    # ---------- OPTION 2: DATA CLEANING REPORT ----------
    elif option == "🧹 Data Cleaning Report":
        st.subheader("🧹 Data Cleaning Summary")
        for step, result in st.session_state['cleaning_log'].items():
            st.markdown(f"✅ **{step}**: {result}")

        st.subheader("📉 Missing Values (Before & After Cleaning)")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Before Cleaning**")
            st.dataframe(missing_info_before[missing_info_before > 0])
        with col2:
            st.write("**After Cleaning**")
            st.dataframe(missing_info_after[missing_info_after > 0])

    # ---------- OPTION 3: SUMMARY/CORRELATION ----------
    elif option == "📈 Summary & Correlations":
        st.subheader("📌 Correlation Heatmap (Numerical)")
        numeric_df = df.select_dtypes(include='number')
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

        st.subheader("🧩 Missing Values (After Cleaning)")
        st.write(df.isnull().sum())

    # ---------- OPTION 4: VISUALIZATIONS ----------
    elif option == "📊 Visualizations":
        # Objective 1: Relationship Between Study Habits and Final Grade
        st.header("Study Hours, Assignment Completion, and Exam Score influence Final Grade")

        # Scatter Plots
        st.subheader("📍 Scatter Plots")
        scatter_cols = ['Study_Hours_per_Week', 'Assignment_Completion_Rate (%)', 'Exam_Score (%)']
        for col in scatter_cols:
            fig, ax = plt.subplots()
            sns.scatterplot(data=df, x=col, y='Final_Grade', hue='Gender', palette='cool', ax=ax)
            ax.set_title(f'{col.replace("_", " ")} vs Final Grade')
            ax.set_xlabel(col.replace("_", " "))
            ax.set_ylabel("Final Grade")
            st.pyplot(fig)

        # Objective 2: Impact of Lifestyle Factors on Performance
        st.header("💤Impact of Lifestyle Factors on Final Grade or Exam Score")

        # Box Plots
        st.subheader("📦 Box Plots: Lifestyle Factors vs Performance")
        lifestyle_cols = ['Sleep_Hours_per_Night', 'Time_Spent_on_Social_Media (hours/week)']
        performance_metric = 'Final_Grade'

        for col in lifestyle_cols:
            fig, ax = plt.subplots()
            sns.boxplot(data=df, x='Gender', y=col, palette='Set2', ax=ax)
            ax.set_title(f'{col} by Gender')
            st.pyplot(fig)

        # Objective 3: Stress Level Analysis
        st.header("😰 Stress Level Analysis")

        st.subheader("📊 Grouped Bar Plot: Average Metrics by Stress Level")

        # Correct column names
        stress_factors = ['Sleep_Hours_per_Night', 'Study_Hours_per_Week', 'Final_Grade']

        # Ensure numeric conversion to avoid aggregation errors
        for col in stress_factors:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Group by stress level and calculate means
        stress_grouped = df.groupby('Self_Reported_Stress_Level')[stress_factors].mean().reset_index()

        # Plotting bar plots
        fig, ax = plt.subplots(figsize=(10, 6))
        stress_grouped.set_index('Self_Reported_Stress_Level').plot(kind='bar', ax=ax)
        ax.set_title("Average Performance & Lifestyle Factors by Stress Level")
        ax.set_ylabel("Average Values")
        ax.set_xlabel("Self-Reported Stress Level")
        ax.legend(title="Metric")
        st.pyplot(fig)

        # Correlation Heatmap
        st.subheader("🔆 Heatmap: Correlation with Stress Level")

        # Encode 'Self_Reported_Stress_Level' if it's categorical
        df['Self_Reported_Stress_Level_Encoded'] = df['Self_Reported_Stress_Level'].astype('category').cat.codes
        corr = df[stress_factors + ['Self_Reported_Stress_Level_Encoded']].corr()

        fig, ax = plt.subplots()
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
        ax.set_title("Correlation Between Stress Level and Other Metrics")
        st.pyplot(fig)
