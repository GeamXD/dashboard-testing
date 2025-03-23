# Import libraries
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio
import streamlit as st
import streamlit_shadcn_ui as ui


# Streamlit page name
st.set_page_config(
    page_title="Heart Disease Dashboard",
    page_icon=":heart:",
    layout="wide",
    # initial_sidebar_state="expanded"
)


# Select template
# pio.templates.default = "plotly_white"

# 'plotly': Default template.
# 'plotly_white': Clean, white background.
# 'ggplot2': Mimics the style of ggplot2 (R).
# 'seaborn': Mimics the style of Seaborn.
# 'simple_white': Minimalist white background.
# 'presentation': 
custom_colors = ["#F17D0A", "#4682B4"]

# Load dataset
@st.cache_data
def load_dataset(filepath: str) -> pd.DataFrame:
    """
    Load dataset from a file
    Params:
        filepath: str - file path
    Returns:
        pd.DataFrame - dataset
    """
    assert filepath.endswith(".csv"), "File must be a CSV file"
    assert isinstance(filepath, str), "File path must be a string"
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

# Clean dataset
@st.cache_data
def clean_dataset(df: pd.DataFrame, duplicates: bool) -> pd.DataFrame:
    """
    Clean dataset
    Params:
        df: pd.DataFrame - dataset
        duplicates: bool - remove duplicates
    Returns:
        pd.DataFrame - cleaned dataset
    """
    assert isinstance(df, pd.DataFrame), "Input must be a pandas DataFrame"
    assert isinstance(duplicates, bool), "Duplicates must be a boolean"

    # Fix resting column
    df = df.rename(columns={'resting bp s': 'resting blood pressure'})
    # Recode sex columns
    df['sex'] = df['sex'].replace({
        1: 'Male',
        0: 'Female'
    })

    # Recode chest pain type columns
    df['chest pain type'] = df['chest pain type'].replace({
        1: 'Typical angina',
        2: 'Atypical angina',
        3: 'Non-anginal pain',
        4: 'Asymptomatic'
    })

    # Recode fasting blood sugar columns
    df['fasting blood sugar'] = df['fasting blood sugar'].replace({
        1: '> 120 mg/dl',
        0: '< 120 mg/dl'
    })

    # Recode resting ecg olumns
    df['resting ecg'] = df['resting ecg'].replace({
        0: 'Normal',
        1: 'ST-T wave abnormality',
        2: 'Left ventricular hypertrophy'
    })

    # Recode exercise angina columns
    df['exercise angina'] = df['exercise angina'].replace({
        1: 'Yes',
        0: 'No'
    })

    # Recode st slope columns
    df['ST slope'] = df['ST slope'].replace({
        1: 'Upsloping',
        2: 'Flat',
        3: 'Downsloping',
        0: 'Flat'
    })


    # Recode target column
    df['target'] = df['target'].replace({
        0: 'No',
        1: 'Yes'
    })
    df = df.rename(columns={'target': 'disease status'})

    # Create age group column
    df['age group'] = pd.cut(df['age'], bins=[0, 30, 40, 50, 60, 70, 80], 
                             labels=['0-29', '30-39', '40-49', '50-59', '60-69', '70-79'])

    # Create cholesterol group column
    df['cholesterol group'] = pd.cut(df['cholesterol'], bins=[0, 150, 200, 240, 300, 1000],
                                 labels=['<150 mg/dL', '150-199 mg/dL', '200-239 mg/dL', '240-299 mg/dL', '≥300 mg/dL'])

    if duplicates:
        df.drop_duplicates(inplace=True)
        return df
    else:
        return df

def plot_hist(df: str, xcol: str, title: str, x_title: str, bins: int = 40) -> None:
    """
    Plots a histogram
    Params:
        df: DataFrame
        xcol: str
        title: str
        x_title: str
    Returns:
        None
    """
    fig  = px.histogram(df, x=xcol, color='disease status', title=title, nbins=bins,
                        color_discrete_sequence=custom_colors)
    fig.update_layout(
        bargap=0.01,
        width=500,
        height=300,
        yaxis_title='Frequency',
        xaxis_title=x_title,
    )

    return fig


def plot_pie_chart(df: pd.DataFrame, xcol: str, title: str) -> None:
    """
    Plots a pie chart
    Params:
        df: DataFrame
        xcol: str
        title: str
    Returns:
        None
    """
    fig = px.pie(data_frame=df, names=xcol, title=title,  hole=0.6, width=300, height=300,
                 color_discrete_sequence=custom_colors)
    fig.update_traces(
    textinfo='label + percent',
    textposition='inside',
    )

    fig.update_layout(
    showlegend=False,
    )

    return fig

def get_metrics_card_info(df: pd.DataFrame) -> int:
    """
    Get info for metrics card
    Params:
        df: DataFrame
    Returns:
        str
    """
    no_participants = str(df.shape[0])
    num_disease = str(df['disease status'].value_counts().values[0])
    max_hr = str(df[df['disease status'] == 'Yes']['max heart rate'].max())
    num_exerc_angina = str(df[df['disease status'] == 'Yes']['exercise angina'].value_counts().iloc[0])
    return no_participants, num_disease, max_hr, num_exerc_angina


def plot_multivar_bar(df: pd.DataFrame, xcol: str, title: str, x_title: str) -> None:
    """
    Plots a bar chart
    Params:
        df: DataFrame
        xcol: str
        title: str
        x_title: str
    Returns:
        None
    """
    # Group by mean
    color = 'disease status'
    df = df.groupby([xcol, color]).size().reset_index(name='count')
    fig = px.bar(data_frame=df, x=xcol, y='count', color=color,
                 title=title, text_auto=True, barmode='group',
                 color_discrete_sequence=custom_colors)

    fig.update_layout(
        width=500,
        height=300,
        xaxis_title=x_title,
    )

    fig.update_traces(
        textposition='outside',
        )

    return fig


def plot_max_hr():
    """
    Plots max heart rate
    """
    max_heart_rate_counts = df['max heart rate'].value_counts().sort_index()
    fig_max_heart_rate = px.bar(
    x=max_heart_rate_counts.index, y=max_heart_rate_counts.values,
    labels={'x': 'Max Heart Rate', 'y': 'Number of Individuals'},
    title="Max Heart Rate",
    color_discrete_sequence=['#F17D0A']
    )
    fig_max_heart_rate.update_layout(
        height=650,
        width=600
    )
    return fig_max_heart_rate



# Define filepath
filepath = "heart_statlog_cleveland_hungary_final.csv"

# Load dataset
df = load_dataset(filepath)

# Clean dataset
df = clean_dataset(df, duplicates=False)

# Get metrics card info
no_participants, num_disease, max_hr, num_exerc_angina = get_metrics_card_info(df)

# Get pie charts
gender_pie = plot_pie_chart(df, 'sex', 'Gender')
disease_status_pie = plot_pie_chart(df, 'disease status', 'Disease Status')
exercise_angina_pie = plot_pie_chart(df, 'exercise angina', 'Exercise Angina')
fbs_pie = plot_pie_chart(df, 'fasting blood sugar', 'Fasting Blood Sugar')


# Get histogram
max_hr_hist = plot_max_hr()
# rest_bp_hist = plot_hist(df, 'resting blood pressure', 'Resting Blood Pressure Distribution', 'Resting Blood Pressure', bins=20)
cholesterol_hist = plot_hist(df, 'cholesterol', 'Cholesterol Distribution', 'Cholesterol', bins=40)
age_hist = plot_hist(df, 'age', 'Age Distribution', 'Age', bins=40)

# Bar Plots
age_group_by_disease_status = plot_multivar_bar(df, 'age group', 'Age Group by Disease Status', 'Age Group')
sex_by_disease_status = plot_multivar_bar(df, 'sex', 'Gender by Disease Status', 'Gender')
chest_pain_by_disease_status = plot_multivar_bar(df, 'chest pain type', 'Chest pain type by Disease Status', 'Chest Pain Type')
resting_ecg_by_disease_status = plot_multivar_bar(df, 'resting ecg', 'Resting ECG by Disease Status', 'Resting ECG')
st_slope_by_diease_status = plot_multivar_bar(df, 'ST slope', 'ST Slope by Disease Status', 'St Slope')
fbs_by_diease_status = plot_multivar_bar(df, 'fasting blood sugar', 'Fasting Blood Sugar by Disease Status', 'Fasting Blood Sugar')
exer_angine_by_disease_status = plot_multivar_bar(df, 'exercise angina', 'Exercise Angina by Disease Status', 'Exercise Angina')
chole_by_disease_status = plot_multivar_bar(df, 'cholesterol group', 'Cholesterol Group by Disease Status', 'Cholesterol Group')


# Dashboard title
st.markdown("<h1 style='text-align: center;'>Heart Disease Prediction Dashboard ♥️</h1>", unsafe_allow_html=True)
st.write("") # White space

# Metrics cards
metric_col = st.columns(4)
with metric_col[0]:
    ui.metric_card(title="Participants", content=no_participants, description="Total", key='card1')
with metric_col[1]:
    ui.metric_card(title="Disease Status", content=num_disease, description="Total", key='card2')
with metric_col[2]:
    ui.metric_card(title="Max Heart Rate", content=max_hr, description="bpm", key='card3')
with metric_col[3]:
    ui.metric_card(title="Exercise Angina", content=num_exerc_angina, description="Total", key='card4')

# First row of visuals
first_col = st.columns(3)
with first_col[0]:
    # Bar chart
    st_slope_con = st.container(border=True)
    st_slope_con.plotly_chart(st_slope_by_diease_status, use_container_width=True)
    # Pie chart
    diesase_stat_con = st.container(border=True)
    diesase_stat_con.plotly_chart(disease_status_pie, use_container_width=True)
with first_col[1]:
    # Histogram
    mx_hr = st.container(border=True)
    mx_hr.plotly_chart(max_hr_hist, use_container_width=True)
with first_col[2]:
    # bar chart
    chst_con = st.container(border=True)
    chst_con.plotly_chart(chest_pain_by_disease_status, use_container_width=True)
    # bar chart
    rest_con = st.container(border=True)
    rest_con.plotly_chart(resting_ecg_by_disease_status, use_container_width=True)

# Second row of visuals
second_col = st.columns(3)
with second_col[0]:
    # Bar chart
    age_con = st.container(border=True)
    age_con.plotly_chart(age_group_by_disease_status, use_container_width=True)
with second_col[1]:
    # bar chart
    fbs_con = st.container(border=True)
    fbs_con.plotly_chart(fbs_by_diease_status, use_container_width=True)
with second_col[2]:
    # bar chart
    chol_con = st.container(border=True)
    chol_con.plotly_chart(chole_by_disease_status, use_container_width=True)


# Final Row
final_row = st.columns(2)
with final_row[0]:
    # Histogram
    age_h_con = st.container(border=True)
    age_h_con.plotly_chart(age_hist, use_container_width=True)
with final_row[1]:
    # Histogram
    chol_hst_con = st.container(border=True)
    chol_hst_con.plotly_chart(cholesterol_hist, use_container_width=True)
