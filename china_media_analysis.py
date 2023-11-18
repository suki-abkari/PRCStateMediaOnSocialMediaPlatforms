import streamlit as st
import pandas as pd
import plotly.express as px
from textblob import TextBlob
import re
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

st.set_page_config(
    page_title="Ex-stream-ly Cool App",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load data with caching disabled
@st.cache_data
def load_data(file_path, sheet_name):
    return pd.read_excel(file_path, sheet_name=sheet_name)

# Function to display dataset and summary
def display_dataset_summary(df):
    st.write("Full Dataset")
    st.write(df.head(758))  # Displaying the first 10 rows for brevity

# Objective 1
# Function to perform data visualization and analysis
def perform_data_visualization(df, selected_tab):
    relevant_columns = []

    if selected_tab == "Followers":
        relevant_columns = ['X (Twitter) Follower #', 'Facebook Follower #', 'Instagram Follower #', 'Threads Follower #']
        title = 'Followers Comparison'
        y_label = 'Followers'
        description = f"""
        The bar charts above compare the {y_label} of PRC State Media on different social media platforms.
        - ***Black bars (Tweetteer - X):*** Represents the Facebook {y_label}. This group focuses on Twitter followers.
        - ***Blue bars (Facebook):*** Represents the Facebook {y_label}. This group focuses on Facebook followers.
        - ***Orange bars (Instagram):*** Represents the Instagram {y_label}. This group focuses on Instagram followers.
        - ***Green bars (Threads):*** Represents the Threads {y_label}. This group focuses on Thread followers.
        """

    elif selected_tab == "Subscribers":
        relevant_columns = ['YouTube Subscriber #', 'TikTok Subscriber #']
        title = 'Subscribers Comparison'
        y_label = 'Subscribers'
        description = f"""
        The bar charts above compare the {y_label} of PRC State Media on different social media platforms.
        - ***Red bars (Youtube):*** Represents the Youtube {y_label}. This group focuses on Youtube subscribers.
        - ***Pink bars (TikTok):*** Represents the TikTok {y_label}. This group focuses on TikTok subscribers.
        """

    available_columns = [col for col in relevant_columns if col in df.columns]

    if available_columns:
        chunk_size = 50
        total_names = len(df['Name (English)'])
        num_chunks = total_names // chunk_size + 1

        for i in range(1, num_chunks + 1):
            tab_name = f"Group {i} - {selected_tab} Analysis"
            with st.expander(tab_name):
                start_idx = (i - 1) * chunk_size
                end_idx = min(i * chunk_size, total_names)
                subset_names = df['Name (English)'].iloc[start_idx:end_idx]

                fig_social_media_presence = px.bar(
                    df[df['Name (English)'].isin(subset_names)],
                    x='Name (English)',
                    y=available_columns,
                    title=title,
                    labels={'value': y_label, 'variable': 'Social Media Platform'},
                    template='plotly',
                    color_discrete_map={'X (Twitter) Follower #': 'black', 'Facebook Follower #': 'blue', 'Instagram Follower #': 'purple', 'Threads Follower #': 'green',
                                        'YouTube Subscriber #': 'red', 'TikTok Subscriber #': 'violet'}
                )

                fig_social_media_presence.update_layout(barmode='stack', height=800)
                fig_social_media_presence.update_xaxes(tickangle=45)

                # Display analysis and description before the chart
                st.markdown(description)
                st.plotly_chart(fig_social_media_presence, use_container_width=True, className="scrolling-container")

        # Description and Analysis (outside the tabs)
        st.subheader(f"{selected_tab} Analysis and Insights")
    else:
        st.warning(f"The selected tab '{selected_tab}' doesn't have any relevant data columns.")

# Load your dataset
file_path = 'data/CANIS_PRC_state_media_on_social_media_platforms-2023-11-03.xlsx'
dataset_sheet_name = 'FULL'
df_dataset = load_data(file_path, sheet_name=dataset_sheet_name)

# Objective 2
# Function to perform sentiment analysis on social media engagement
def analyze_sentiment_textblob(text):
    analysis = TextBlob(str(text))
    return analysis.sentiment.polarity

def perform_sentiment_analysis(df):
    text_columns = ['Name (English)', 'Name (Chinese)', 'Entity owner (English)', 'Entity owner (Chinese)',
                    'Parent entity (English)', 'Parent entity (Chinese)']

    title = 'Sentiment Analysis on Social Media Entities'
    y_label = 'Sentiment Score'

    description = f"""
    The stacked bar chart below showcases sentiment analysis on various text columns of social media entities.
    X-axis (Horizontal Axis): Sentiment
    Negative: Represents the negative sentiment scores.
    Neutral: Represents the neutral sentiment scores.
    Positive: Represents the positive sentiment scores.

    Y-axis (Vertical Axis): Count
    Represents the count of occurrences for each sentiment category.
    The height of each bar indicates how many text entries fall into the respective sentiment category.

    Stacked Bar Chart:
    Each bar is divided into three segments (Negative, Neutral, Positive),
    representing the distribution of sentiment within the selected text column.
    The color of each segment corresponds to the sentiment category (e.g., red for negative, grey for neutral, green for positive).

    Chart Title:
    Indicates which text column is being analyzed for sentiment (e.g., 'Name (English) Sentiment Analysis Stacked Bar Chart').

    Descriptive Text:
    Provides additional information about the analysis, the sentiment calculation method (TextBlob),
    and the score interpretation (ranging from -1 to 1, where 0 is neutral).

    Summary Statistics:
    Mean Sentiment: The average sentiment score across all text entries in the selected column.
    A higher mean indicates a more positive sentiment, while a lower mean indicates a more negative sentiment.
    Standard Deviation: Measures the amount of variation or dispersion of sentiment scores.
    A higher standard deviation suggests more variability in sentiment.

    Overall Purpose:
    The chart and accompanying information aim to visualize and analyze the sentiment distribution in different text columns
    of social media entities. This helps in understanding the emotional context and tone of social media engagement for each specified text column.
    """
    print("Columns in DataFrame:", df.columns)

    st.caption(description)

    available_columns = [col for col in text_columns if col in df.columns]

    if available_columns:
        selected_tab = st.selectbox("Select Tab", available_columns)

        sentiment_scores = df[selected_tab].apply(analyze_sentiment_textblob)
        sentiment_labels = ['Negative', 'Neutral', 'Positive']
        sentiment_counts = [sum(sentiment_scores < 0), sum(sentiment_scores == 0), sum(sentiment_scores > 0)]

        fig_stacked_bar = px.bar(
            x=sentiment_labels,
            y=sentiment_counts,
            title=f'{selected_tab} Sentiment Analysis Stacked Bar Chart',
            labels={'x': 'Sentiment', 'y': 'Count'},
            color=sentiment_labels,
            template='plotly',
        )

        st.subheader(f"{selected_tab} Sentiment Analysis Distribution")
        st.plotly_chart(fig_stacked_bar, use_container_width=True)

        st.subheader(f"{selected_tab} Summary Statistics")
        st.write(f"Mean Sentiment: {sentiment_scores.mean():.2f}")
        st.write(f"Standard Deviation: {sentiment_scores.std():.2f}")

    else:
        st.warning("No relevant text columns for sentiment analysis.")

# Set the width of the sidebar using custom CSS
st.markdown(
    """
    <style>
        .reportview-container .sidebar-content {
            width: 300px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("PRC State Media On Social Media Platforms - 2023-11-03")

# Display dataset and summary
display_dataset_summary(df_dataset)

# Set up tabs for objectives
selected_objective = st.selectbox("Select Objective", ["Objective 1: Subscribers Analysis and Insights", "Objective 2: Sentiment Analysis on Social Media Engagement"])

# Define content for each tab
if "Objective 1" in selected_objective:
    st.title("Objective 1: Subscribers/Followers Analysis and Insights")
    st.caption('These visualizations provide a clear comparison of the Followers across various social media platforms, allowing us to identify trends and patterns in the data specific to Followers and Subscribers.')
    st.sidebar.caption('Please select the desired tab to explore the corresponding analysis.')

    selected_tab = st.selectbox("Select Tab", ["Followers", "Subscribers"])
    perform_data_visualization(df_dataset, selected_tab)

elif "Objective 2" in selected_objective:
    st.title("Objective 2: Sentiment Analysis on Social Media Engagement")
    st.caption('The objective of Sentiment Analysis on Social Media Engagement is to gain insights into the emotional tone and sentiment expressed in the textual content associated with social media engagement. This includes analyzing comments, likes, and shares across different social media platforms. By assessing sentiment, we aim to identify patterns and trends in the way audiences interact with and respond to the content presented by PRC State Media.')
    st.sidebar.caption('Please select the desired tab to explore the corresponding analysis.')

    perform_sentiment_analysis(df_dataset)



# Sidebar demonstration section
st.sidebar.title("Demos")
demo_links = {
    "Objective 1": {
        "url": "data/chrome_GpID0Dy4kc.gif",
        "type": "image",
        "caption": "Demonstration of chart usage Analysis and subscriber information",
    },
    "Objective 2": {
        "url": "data/chrome_GpID0Dy4kc.gif",
        "type": "image",
        "caption": "Demonstration of chart usage ...",
    },
}

selected_demo = st.sidebar.selectbox("Select Demo", list(demo_links.keys()))

demo = demo_links[selected_demo]
if demo["type"] == "video":
    st.sidebar.video(demo["url"])
    st.sidebar.caption(demo["caption"])
elif demo["type"] == "image":
    st.sidebar.image(demo["url"], caption=demo["caption"], use_column_width=True)