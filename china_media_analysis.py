import streamlit as st
import pandas as pd
import plotly.express as px
from textblob import TextBlob

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

# Load your dataset
file_path = 'data/CANIS_PRC_state_media_on_social_media_platforms-2023-11-03.xlsx'
dataset_sheet_name = 'FULL'
df_dataset = load_data(file_path, sheet_name=dataset_sheet_name)

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
    else:
        st.warning(f"The selected tab '{selected_tab}' doesn't have any relevant data columns.")

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

# Objective 3
# Function to perform language-based analysis
def perform_language_analysis(df):
    language_column = 'Language'
    social_media_columns = ['X (Twitter) Follower #', 'Facebook Follower #', 'Instagram Follower #', 'Threads Follower #',
                             'YouTube Subscriber #', 'TikTok Subscriber #']

    if language_column in df.columns and any(col in df.columns for col in social_media_columns):
        fig_language = px.bar(
            df,
            x=language_column,
            y=social_media_columns,
            title='Language-based Analysis of Subscribers and Followers',
            labels={'value': 'Counts of Followers/Subscribers', 'variable': 'Social Media Platform'},
            template='plotly',
            color_discrete_map={'X (Twitter) Follower #': 'black', 'Facebook Follower #': 'blue', 'Instagram Follower #': 'purple',
                                'Threads Follower #': 'green', 'YouTube Subscriber #': 'red', 'TikTok Subscriber #': 'violet'},
            barmode='group',  # This makes the bars grouped instead of stacked
        )

        st.title("Objective 3: Language Engagement Analysis")

        # Add a comprehensive caption explaining the objectives and insights of the analysis
        st.caption("""
        This analysis aims to provide insights into language-based engagement on social media platforms for PRC State Media entities.
        The grouped bar chart displays the distribution of subscribers and followers across different social media platforms based on the language
        in which content is presented. Each group of bars represents a language, and within each group, the different colors correspond
        to the counts of subscribers and followers on different social media platforms.

        Key Insights:
        - Identify which languages have the highest and lowest engagement across social media platforms.
        - Understand the distribution of subscribers and followers on specific platforms for each language.
        - Explore potential correlations between language choice and social media engagement.
        """)

        st.plotly_chart(fig_language, use_container_width=True)
    else:
        st.warning("The dataset doesn't contain the necessary columns for language-based analysis.")
        
# Objective 4
# Function to display top N entities
def display_top_entities(df, n=10):
    # Caption explaining chart elements
    st.caption("""
    **Chart Explanation:**
    - **Number of Occurrences:** The horizontal bars in this chart represent the frequency of each media entity's appearance in the dataset. The length of each bar is proportional to the number of occurrences, providing a quantitative measure of the media entities' prominence.
    - **Entity Owner:** The y-axis displays the names of the media entities included in the analysis. Each bar corresponds to a specific media entity, allowing for easy identification.
    - **Color:** The color of each bar is determined by the count of occurrences, creating a color gradient. Darker colors indicate higher frequencies, while lighter colors represent lower frequencies. This color scale provides an additional dimension for understanding the dataset.

    **Context:**
    This analysis employs a horizontal bar chart to visually depict the distribution of media entities based on their occurrences in the dataset. The choice of a horizontal bar chart is suitable for comparing the frequencies of different entities, enabling a quick assessment of the most frequently mentioned media entities.

    **Statistical Significance:**
    The length of the bars serves as a direct representation of the statistical significance of each media entity within the dataset. Entities with longer bars are more frequently mentioned, suggesting a higher level of importance or relevance in the context under study.

    **Color Mapping and Frequency Gradient:**
    The color gradient adds an additional layer of information, allowing for a more nuanced interpretation of the data. Darker colors signify entities with higher frequencies, contributing to a more detailed analysis of the prominence distribution.

    **Insights and Further Analysis:**
    - Identification of Key Players: The chart facilitates the identification of key media entities with the highest occurrence rates.
    - Comparative Analysis: Researchers can compare the prominence of different entities, gaining insights into the overall landscape of media representation.

    This analytical approach provides a quantitative and visual representation of media entity occurrences, aiding researchers in uncovering patterns and making informed interpretations.
    """)
    # Count occurrences of each entity
    entity_counts = df['Entity owner (English)'].value_counts().head(n)
    
    # Plot a different type of chart (e.g., horizontal bar chart)
    fig_entities = px.bar(
        x=entity_counts.values,
        y=entity_counts.index,
        orientation='h',  # Use 'h' for a horizontal bar chart
        labels={'x': 'Number of Occurrences', 'y': 'Entity Owner'},
        title=f'Top {n} Entity Owners',
        color=entity_counts.values,
        color_continuous_scale='Viridis',
    )

    # Display the chart
    st.plotly_chart(fig_entities, use_container_width=True)
# Function to perform platform-specific analysis
def perform_platform_analysis(df):
    
    platform_columns = ['X (Twitter) Follower #', 'Facebook Follower #', 'Instagram Follower #', 'Threads Follower #',
                        'YouTube Subscriber #', 'TikTok Subscriber #']

    if any(col in df.columns for col in platform_columns):
        # Create a bar chart using Plotly Express
        fig_platform = px.bar(
            df,
            x='Name (English)',
            y=platform_columns,
            title='Platform-specific Analysis of Followers and Subscribers',
            labels={'value': 'Counts', 'variable': 'Social Media Platform'},
            template='plotly',
            color_discrete_map={'X (Twitter) Follower #': 'black', 'Facebook Follower #': 'blue', 'Instagram Follower #': 'purple',
                                'Threads Follower #': 'green', 'YouTube Subscriber #': 'red', 'TikTok Subscriber #': 'violet'}
        )

        # Customize the layout of the chart
        fig_platform.update_layout(
            barmode='stack',  # Choose 'stack' or 'group' as per your preference
            xaxis_title='Media Entities',
            yaxis_title='Counts',
            legend_title='Social Media Platforms',
            height=600,
            width=1000,
        )

        # Display insights and explanations above the chart
        st.title("Objective 4: Platform-specific Analysis")
        display_top_entities(df, n=10)
        st.caption("""
        The objective of this analysis is to provide a detailed examination of the followers and subscribers 
        across various social media platforms for PRC State Media entities. The data is presented in the form 
        of a stacked bar chart, where each bar represents a distinct media entity.

        Key Analytical Components:
        - **Data Selection:** The dataset is examined for specific columns related to follower and subscriber counts on various platforms.
        - **Visualization:** The stacked bar chart visually represents the distribution of counts for each platform across media entities.
        - **Color Mapping:** Different colors within each bar indicate counts on different social media platforms.
        
        Insights:
        - Identify the distribution of followers and subscribers across various social media platforms for each media entity.
        - Explore variations in engagement on different platforms.
        - Understand the overall social media presence of each media entity.

        This analysis aids in uncovering patterns, trends, and disparities in the engagement levels of PRC State Media entities on different social media platforms.
        """)
        # Display the chart with the specified width
        st.plotly_chart(fig_platform, use_container_width=True)
    else:
        st.warning("The dataset doesn't contain the necessary columns for platform-specific analysis.")


st.title("PRC State Media On Social Media Platforms - 2023-11-03")

# Display dataset and summary
display_dataset_summary(df_dataset)

# Set up tabs for objectives
selected_objective = st.selectbox("Select Objective", ["Objective 1: Subscribers/Followers Analysis and Insights",
                                                       "Objective 2: Sentiment Analysis on Social Media Engagement",
                                                       "Objective 3: Language Engagement Analysis",
                                                       "Objective 4: Platform-specific Analysis"])

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

elif "Objective 3" in selected_objective:
    perform_language_analysis(df_dataset)

elif "Objective 4" in selected_objective:
    perform_platform_analysis(df_dataset)

# Define documentation for each objective
documentation = {
    "Objective 1: Subscribers/Followers Analysis and Insights": """
        This objective focuses on providing a clear comparison of the Followers across various social media platforms.
        The visualizations allow us to identify trends and patterns in the data specific to Followers and Subscribers.
    """,
    "Objective 2: Sentiment Analysis on Social Media Engagement": """
        The objective of Sentiment Analysis on Social Media Engagement is to gain insights into the emotional tone and sentiment expressed in the textual content associated with social media engagement.
        This includes analyzing comments, likes, and shares across different social media platforms.
        By assessing sentiment, we aim to identify patterns and trends in the way audiences interact with and respond to the content presented by PRC State Media.
    """,
    "Objective 3: Language Engagement Analysis": """
        This analysis provides insights into language-based engagement on social media platforms for PRC State Media entities.
        The chart displays the distribution of subscribers and followers across different social media platforms based on the language in which content is presented.
        Key insights include identifying languages with the highest and lowest engagement and exploring potential correlations between language choice and social media engagement.
    """,
    "Objective 4: Platform-specific Analysis": """
        This objective focuses on platform-specific analysis, providing insights into the distribution of followers and subscribers across different social media platforms.
        The chart displays counts of followers and subscribers for each social media platform, allowing for a platform-specific understanding of engagement.
    """
}

# Sidebar section for objectives documentation
st.sidebar.title("Objectives Documentation")

# Set up tabs for objectives documentation
selected_objective_doc = st.sidebar.selectbox("Select Objective Documentation", list(documentation.keys()))

# Display documentation for the selected objective in the sidebar
st.sidebar.markdown(documentation[selected_objective_doc])