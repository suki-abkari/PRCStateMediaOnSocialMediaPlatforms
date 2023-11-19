# China Media Analysis Project

## Overview

This project analyzes social media data related to PRC State Media entities. It provides insights into followers, subscribers, sentiment analysis, language engagement, and platform-specific analysis.

## Table of Contents

- [Dataset](#dataset)
- [Setup](#setup)
- [Usage](#usage)
- [Objectives](#objectives)
- [Results](#results)
- [Acknowledgments](#acknowledgments)

## Dataset

The dataset used for this analysis includes the following columns:

- Name (English)
- Name (Chinese)
- Region of Focus
- Language
- Entity owner (English)
- Entity owner (Chinese)
- Parent entity (English)
- Parent entity (Chinese)
- X (Twitter) handle
- X (Twitter) URL
- X (Twitter) Follower #
- Facebook page
- Facebook URL
- Facebook Follower #
- Instagram page
- Instagram URL
- Instagram Follower #
- Threads account
- Threads URL
- Threads Follower #
- YouTube account
- YouTube URL
- YouTube Subscriber #
- TikTok account
- TikTok URL
- TikTok Subscriber #

## Setup

1. Clone the repository:

    ```bash
    git clone https://github.com/suki-abkari/PRCStateMediaOnSocialMediaPlatforms.git
    ```

2. Navigate to the project directory:

    ```bash
    cd china_media_analysis
    ```

3. Virtual Environment

    ```bash
    python -m venv venv
    .\venv\Scripts\activate  # On Windows
	source venv/bin/activate  # On macOS/Linux
    ```
     Once activated, your shell prompt should change to indicate that you are working within the virtual environment. 
     Any packages you install or scripts you run will be specific to that environment.

4. Install the required dependencies:

    ```bash
    pip install streamlit pandas plotly textblob scikit-learn matplotlib seaborn nltk spacy
    ```

## Usage

Run the presentation script:

    ```bash
    streamlit run china_media_analysis.py
    ```

## Objectives

The analysis is structured into four main objectives:

1. **Objective 1: Subscribers/Followers Analysis and Insights**
    - Comparison of followers across various social media platforms.

2. **Objective 2: Sentiment Analysis on Social Media Engagement**
    - Analysis of sentiment in textual content associated with social media engagement.

3. **Objective 3: Language Engagement Analysis**
    - Insights into language-based engagement on social media platforms.

4. **Objective 4: Platform-specific Analysis**
    - Detailed examination of followers and subscribers on different social media platforms.

## Results

The results and visualizations are presented in the generated HTML file.

## Acknowledgments

- This project uses Streamlit, Plotly, and other libraries for data analysis and visualization.
- Special thanks to [Your Name] for their contributions.

Feel free to customize this README to provide more specific details about your project, dependencies, and usage instructions.

