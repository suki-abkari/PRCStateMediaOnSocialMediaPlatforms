import streamlit as st
import pandas as pd
import plotly.express as px
from textblob import TextBlob

st.set_page_config(
    page_title="PRC State Media Analysis",
    page_icon="📊",
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
    st.write(df.head(758)) 

# Objective 1
# Function to perform data visualization and analysis
def perform_data_visualization(df, selected_tab):
    relevant_columns = []
    title = ''
    y_label = ''
    description = ''

    if selected_tab == "Followers":
        relevant_columns = ['X (Twitter) Follower #', 'Facebook Follower #', 'Instagram Follower #', 'Threads Follower #']
        title = 'Followers Comparison'
        y_label = 'Followers'
        description = f"""
        The bar charts above compare the {y_label} of PRC State Media on different social media platforms.
        - **Black bars:** Represents Twitter {y_label}.
        - **Blue bars:** Represents Facebook {y_label}.
        - **Orange bars:** Represents Instagram {y_label}.
        - **Green bars:** Represents Thread {y_label}.
        """

    elif selected_tab == "Subscribers":
        relevant_columns = ['YouTube Subscriber #', 'TikTok Subscriber #']
        title = 'Subscribers Comparison'
        y_label = 'Subscribers'
        description = f"""
        The bar charts above compare the {y_label} of PRC State Media on different social media platforms.
        - **Red bars:** Represents YouTube {y_label}.
        - **Pink bars:** Represents TikTok {y_label}.
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
            barmode='group',
        )

        st.plotly_chart(fig_language, use_container_width=True)
    else:
        st.warning("The dataset doesn't contain the necessary columns for language-based analysis.")
        
# Objective 4
# Function to display top N entities
def display_top_entities(df, n=10):
    # Count occurrences of each entity
    entity_counts = df['Entity owner (English)'].value_counts().head(n)
    
    fig_entities = px.bar(
        x=entity_counts.values,
        y=entity_counts.index,
        orientation='h', #'h' for a horizontal bar chart
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
            title='Platform specific Analysis of Followers and Subscribers',
            labels={'value': 'Counts', 'variable': 'Social Media Platform'},
            template='plotly',
            color_discrete_map={'X (Twitter) Follower #': 'black', 'Facebook Follower #': 'blue', 'Instagram Follower #': 'purple',
                                'Threads Follower #': 'green', 'YouTube Subscriber #': 'red', 'TikTok Subscriber #': 'violet'}
        )
        
        fig_platform.update_layout(
            barmode='stack', 
            xaxis_title='Media Entities',
            yaxis_title='Counts',
            legend_title='Social Media Platforms',
            height=600,
            width=1000,
        )
        display_top_entities(df, n=10)
        st.plotly_chart(fig_platform, use_container_width=True)
    else:
        st.warning("The dataset doesn't contain the necessary columns for platform specific analysis.")


# Brief description about the application and dataset
st.title("Media Insight Explorer: PRC State Media")
# Use the columns layout
col1, col2 = st.columns(2)
# Dataset Information
with col1:
    st.header(""" ***Dataset Information*** """)
    st.container()
    st.markdown("""
    The dataset contains relevant metrics such as follower counts on Twitter, Facebook, Instagram, Threads, YouTube, and TikTok for various PRC State Media entities. It also includes sentiment scores, language-based engagement, and platform-specific metrics.
    
    Feel free to navigate through different objectives using the tabs on the left to gain valuable insights and make data-driven decisions.
    """)
# Application Overview
with col2:
    st.header(""" ***Application Overview*** """)
    st.container()
    st.markdown("""
    Explore insights and analytics related to PRC State Media's engagement on various social media platforms. The analysis covers subscribers/followers, sentiment analysis, language engagement, and platform-specific metrics.
    """)




# Display dataset and summary
display_dataset_summary(df_dataset)

# Set up tabs for objectives
selected_objective = st.selectbox("Select Objective: Please select the target to display the charst", ["Objective 1: Subscribers/Followers Analysis and Insights",
                                                       "Objective 2: Sentiment Analysis on Social Media Engagement",
                                                       "Objective 3: Language Engagement Analysis",
                                                       "Objective 4: Platform specific Analysis"])

# Define content for each tab
if "Objective 1" in selected_objective:
    st.header("Objective 1: Subscribers/Followers Analysis and Insights", divider='violet')

    # Create a container for the left column
    left_column, right_column = st.columns(2)

    # Followers Analysis
    with left_column:
        st.subheader("1. Followers Analysis:")
        st.markdown("""
        - **Scientific Objective:** Quantify and compare the follower counts of PRC State Media entities on different social media platforms, including Twitter, Facebook, Instagram, and Threads.
        - **Representation:** Visualizes the distribution of followers across platforms, highlighting the relative popularity of each platform for state media entities.
        - **Insights:** Identify trends, anomalies, or significant variations in follower counts, offering insights into audience preferences and the impact of state media on different platforms.
        """)

    # Subscribers Analysis
    with right_column:
        st.subheader("2. Subscribers Analysis:")
        st.markdown("""
        - **Scientific Objective:** Analyze and compare the subscriber counts of PRC State Media on YouTube and TikTok.
        - **Representation:** Presents a visual comparison of subscribers on YouTube and TikTok, showcasing the platforms' significance for state media engagement.
        - **Insights:** Uncover patterns in subscriber growth, assess the popularity of state media content on video-sharing platforms, and identify potential correlations with global events or specific content types.
        """)

    # Methodology and Analysis Framework
    st.subheader("Methodology and Analysis Framework")
    st.markdown(""" 1. #### ***Data Collection*** """)
    st.markdown("The analysis is based on a dataset containing relevant metrics such as follower counts on Twitter, Facebook, Instagram, Threads, YouTube, and TikTok for various PRC State Media entities.")
    
    st.markdown(""" 2. #### ***Data Processing and Cleaning*** """)
    st.markdown("Prior to analysis, the dataset undergoes thorough processing and cleaning to ensure accuracy and reliability. Missing or inconsistent data points are addressed to maintain the integrity of the analysis.")
    
    st.markdown(""" 3. #### ***Visualization and Interpretation*** """)
    st.markdown("""
    - **Dynamic Tab Selection:** Users can choose between 'Followers' and 'Subscribers' tabs to focus on specific metrics.
    - **Granular Analysis:** The dataset is divided into manageable chunks, enabling a detailed examination of state media entities and their social media metrics.
    - **Color-Coded Representation:** Visualizations utilize a color-coded scheme for different social media platforms, enhancing clarity and ease of interpretation.
    - **Descriptive Insights:** Informative descriptions accompany visualizations, providing context and guiding users through the analysis.
    """)

    # Limitations and Future Considerations
    st.markdown(""" 3. #### ***Limitations and Future Considerations*** """)
    st.markdown("While this analysis provides valuable insights, it is essential to acknowledge potential limitations, such as data availability and the dynamic nature of social media metrics. Future iterations may involve incorporating additional metrics, sentiment analysis, and user feedback to enhance the depth and accuracy of the analysis.")

    st.markdown("""
    In essence, Objective 1 aims to empower users with a nuanced understanding of PRC State Media's social media presence, facilitating data-driven insights and informed decision-making.
    """)
    
    selected_tab = st.selectbox("Select Tab", ["Followers", "Subscribers"])
    perform_data_visualization(df_dataset, selected_tab)
    st.header("Objective 1: Subscribers/Followers Analysis and Insights", divider='violet')
    # Followers Analysis
    st.subheader("1. Followers Analysis:")
    st.markdown("""
    - **Scientific Objective:** Quantify and compare the follower counts of PRC State Media entities on different social media platforms, including Twitter, Facebook, Instagram, and Threads.
    - **Representation:** Visualizes the distribution of followers across platforms, highlighting the relative popularity of each platform for state media entities.
    - **Insights:** Identify trends, anomalies, or significant variations in follower counts, offering insights into audience preferences and the impact of state media on different platforms.
    """)

    # Subscribers Analysis
    st.subheader("2. Subscribers Analysis:")
    st.markdown("""
    - **Scientific Objective:** Analyze and compare the subscriber counts of PRC State Media on YouTube and TikTok.
    - **Representation:** Presents a visual comparison of subscribers on YouTube and TikTok, showcasing the platforms' significance for state media engagement.
    - **Insights:** Uncover patterns in subscriber growth, assess the popularity of state media content on video-sharing platforms, and identify potential correlations with global events or specific content types.
    """)

    st.subheader("Methodology and Analysis Framework:")
    st.markdown("### Data Collection:")
    st.markdown("The analysis is based on a dataset containing relevant metrics such as follower counts on Twitter, Facebook, Instagram, Threads, YouTube, and TikTok for various PRC State Media entities.")
    
    st.markdown("### Data Processing and Cleaning:")
    st.markdown("Prior to analysis, the dataset undergoes thorough processing and cleaning to ensure accuracy and reliability. Missing or inconsistent data points are addressed to maintain the integrity of the analysis.")
    
    st.markdown("### Visualization and Interpretation:")
    st.markdown("""
    - **Dynamic Tab Selection:** Users can choose between 'Followers' and 'Subscribers' tabs to focus on specific metrics.
    - **Granular Analysis:** The dataset is divided into manageable chunks, enabling a detailed examination of state media entities and their social media metrics.
    - **Color-Coded Representation:** Visualizations utilize a color-coded scheme for different social media platforms, enhancing clarity and ease of interpretation.
    - **Descriptive Insights:** Informative descriptions accompany visualizations, providing context and guiding users through the analysis.
    """)

    st.subheader("Limitations and Future Considerations:")
    st.markdown("While this analysis provides valuable insights, it is essential to acknowledge potential limitations, such as data availability and the dynamic nature of social media metrics. Future iterations may involve incorporating additional metrics, sentiment analysis, and user feedback to enhance the depth and accuracy of the analysis.")

    st.markdown("""
    In essence, Objective 1 aims to empower users with a nuanced understanding of PRC State Media's social media presence, facilitating data-driven insights and informed decision-making.
    """)
    
    selected_tab = st.selectbox("Select Tab", ["Followers", "Subscribers"])
    perform_data_visualization(df_dataset, selected_tab)
    
elif "Objective 2" in selected_objective:
    st.header("Objective 2: Sentiment Analysis on Social Media Engagement", divider='violet')
    st.markdown("""
    The stacked bar chart below showcases sentiment analysis on various text columns of social media entities. X-axis (Horizontal Axis): Sentiment Negative: Represents the negative sentiment scores. Neutral: Represents the neutral sentiment scores. Positive: Represents the positive sentiment scores.
    """)
    st.markdown("""
    The objective of Sentiment Analysis on Social Media Engagement is to gain insights into the emotional tone and sentiment expressed in the textual content associated with social media engagement. This includes analyzing comments, likes, and shares across different social media platforms. By assessing sentiment, we aim to identify patterns and trends in the way audiences interact with and respond to the content presented by PRC State Media.
    """)    
    st.markdown(""" #### ***Visualization and Interpretation*** """)
    st.markdown("""
    - **Y-axis (Vertical Axis):** Count Represents the count of occurrences for each sentiment category. The height of each bar indicates how many text entries fall into the respective sentiment category.            
    - **Stacked Bar Chart:** Each bar is divided into three segments (Negative, Neutral, Positive), representing the distribution of sentiment within the selected text column. The color of each segment corresponds to the sentiment category.
    - **Chart Title:** Indicates which text column is being analyzed for sentiment.
    - **Descriptive Text:** Provides additional information about the analysis, the sentiment calculation method (TextBlob), and the score interpretation (ranging from -1 to 1, where 0 is neutral).
    - **Summary Statistics:** Mean Sentiment: The average sentiment score across all text entries in the selected column. A higher mean indicates a more positive sentiment, while a lower mean indicates a more negative sentiment. Standard Deviation: Measures the amount of variation or dispersion of sentiment scores. A higher standard deviation suggests more variability in sentiment.
    """)    
    st.markdown("""
    The chart and accompanying information aim to visualize and analyze the sentiment distribution in different text columns of social media entities. This helps in understanding the emotional context and tone of social media engagement for each specified text column.
    """)
    perform_sentiment_analysis(df_dataset)

elif "Objective 3" in selected_objective:
    st.header("Objective 3: Language Engagement Analysis", divider='violet')
    st.markdown("""
    This analysis aims to provide insights into language-based engagement on social media platforms for PRC State Media entities. The grouped bar chart displays the distribution of subscribers and followers across different social media platforms based on the language in which content is presented. Each group of bars represents a language, and within each group, the different colors correspond to the counts of subscribers and followers on different social media platforms.
    """)
    st.markdown("""
    The objective of Sentiment Analysis on Social Media Engagement is to gain insights into the emotional tone and sentiment expressed in the textual content associated with social media engagement. This includes analyzing comments, likes, and shares across different social media platforms. By assessing sentiment, we aim to identify patterns and trends in the way audiences interact with and respond to the content presented by PRC State Media.
    """)    
    st.markdown(""" #### ***Key Insights*** """)
    st.markdown("""
    - Identify which languages have the highest and lowest engagement across social media platforms.
    - Understand the distribution of subscribers and followers on specific platforms for each language.
    - Explore potential correlations between language choice and social media engagement.       
    """)    
    st.markdown("""
    The chart and accompanying information aim to visualize and analyze the sentiment distribution in different text columns of social media entities. This helps in understanding the emotional context and tone of social media engagement for each specified text column.
    """)
    perform_language_analysis(df_dataset)

elif "Objective 4" in selected_objective:
    st.header("Objective 4: Platform specific Analysis", divider='violet')
    st.markdown("""
    This analysis aims to provide insights into language-based engagement on social media platforms for PRC State Media entities. The grouped bar chart displays the distribution of subscribers and followers across different social media platforms based on the language in which content is presented. Each group of bars represents a language, and within each group, the different colors correspond to the counts of subscribers and followers on different social media platforms.
    """)
    st.markdown("""
    The objective of Sentiment Analysis on Social Media Engagement is to gain insights into the emotional tone and sentiment expressed in the textual content associated with social media engagement. This includes analyzing comments, likes, and shares across different social media platforms. By assessing sentiment, we aim to identify patterns and trends in the way audiences interact with and respond to the content presented by PRC State Media.
    """)    
    st.markdown(""" #### ***Chart Explanation*** """)
    st.markdown("""
    - Number of Occurrences: The horizontal bars in this chart represent the frequency of each media entity's appearance in the dataset. The length of each bar is proportional to the number of occurrences, providing a quantitative measure of the media entities' prominence.
    - Entity Owner: The y-axis displays the names of the media entities included in the analysis. Each bar corresponds to a specific media entity, allowing for easy identification.
    - Color: The color of each bar is determined by the count of occurrences, creating a color gradient. Darker colors indicate higher frequencies, while lighter colors represent lower frequencies. This color scale provides an additional dimension for understanding the dataset.
    """)   
    
    st.markdown(""" #### ***Context*** """)
    st.markdown("""
    This analysis employs a horizontal bar chart to visually depict the distribution of media entities based on their occurrences in the dataset. The choice of a horizontal bar chart is suitable for comparing the frequencies of different entities, enabling a quick assessment of the most frequently mentioned media entities.
    """) 
   
    st.markdown(""" #### ***Insights and Further Analysis*** """)       
    st.markdown(""" 
    - Identification of Key Players: The chart facilitates the identification of key media entities with the highest occurrence rates.
    - Comparative Analysis: Researchers can compare the prominence of different entities, gaining insights into the overall landscape of media representation.
    """)


    st.markdown("""
    This analytical approach provides a quantitative and visual representation of media entity occurrences, aiding researchers in uncovering patterns and making informed interpretations.
    """)
    perform_platform_analysis(df_dataset)


# Sidebar
st.sidebar.header("Challenge Submission Info")
st.sidebar.markdown("This project analyzes social media data related to PRC State Media entities. It provides insights into followers, subscribers, sentiment analysis, language engagement, and platform-specific analysis.")
st.sidebar.markdown("🔗 [GitHub Repository](https://github.com/your-team-name/your-hackathon-project)")

st.sidebar.header("Challenge Info")
st.sidebar.markdown(""" - ***Challenge name:*** CANIS Data Visualization and Foreign Interference """)
st.sidebar.markdown(""" - ***Challenge Author:*** [CANIS](https://canis-network.ca) """)
st.sidebar.markdown(""" - ***Code Author:*** [suki-abkari](https://github.com/suki-abkari/) """)

