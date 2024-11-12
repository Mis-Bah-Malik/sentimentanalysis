import streamlit as st
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import pandas as pd
import plotly.express as px

# Download required NLTK data
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

def analyze_sentiment(text):
    """Analyze sentiment of given text using VADER"""
    sia = SentimentIntensityAnalyzer()
    sentiment_scores = sia.polarity_scores(text)
    return sentiment_scores

def get_sentiment_label(compound_score):
    """Convert compound score to sentiment label"""
    if compound_score >= 0.05:
        return "Positive"
    elif compound_score <= -0.05:
        return "Negative"
    else:
        return "Neutral"

def main():
    # Set page config
    st.set_page_config(
        page_title="Sentiment Analysis Tool",
        page_icon="😊",
        layout="wide"
    )

    # Main title
    st.title("✨ Sentiment Analysis Tool")
    st.markdown("""
    This tool analyzes the sentiment of your text using VADER (Valence Aware Dictionary and sEntiment Reasoner).
    Enter your text below to get started!
    """)

    # Create two columns
    col1, col2 = st.columns([2, 1])

    with col1:
        # Text input
        text_input = st.text_area(
            "Enter your text here:",
            height=200,
            placeholder="Type or paste your text here..."
        )

        # Analysis button
        if st.button("Analyze Sentiment", type="primary"):
            if text_input.strip():
                # Get sentiment scores
                sentiment_scores = analyze_sentiment(text_input)
                
                # Get overall sentiment label
                sentiment_label = get_sentiment_label(sentiment_scores['compound'])

                # Display results
                st.markdown("### Results")
                
                # Create metrics row
                metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                
                with metric_col1:
                    st.metric("Overall Sentiment", sentiment_label)
                with metric_col2:
                    st.metric("Positive Score", f"{sentiment_scores['pos']:.3f}")
                with metric_col3:
                    st.metric("Negative Score", f"{sentiment_scores['neg']:.3f}")
                with metric_col4:
                    st.metric("Neutral Score", f"{sentiment_scores['neu']:.3f}")

                # Create DataFrame for visualization
                scores_df = pd.DataFrame({
                    'Sentiment': ['Positive', 'Neutral', 'Negative'],
                    'Score': [
                        sentiment_scores['pos'],
                        sentiment_scores['neu'],
                        sentiment_scores['neg']
                    ]
                })

                # Create bar chart
                fig = px.bar(
                    scores_df,
                    x='Sentiment',
                    y='Score',
                    color='Sentiment',
                    color_discrete_map={
                        'Positive': '#2ECC71',
                        'Neutral': '#3498DB',
                        'Negative': '#E74C3C'
                    }
                )
                fig.update_layout(
                    title='Sentiment Scores Distribution',
                    showlegend=False,
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)

            else:
                st.warning("Please enter some text to analyze.")

    with col2:
        # Sidebar with information
        st.markdown("### How it works")
        st.markdown("""
        This tool uses VADER (Valence Aware Dictionary and sEntiment Reasoner) to analyze sentiments in text. It considers:
        
        - Punctuation
        - Capitalization
        - Emoticons
        - Common expressions
        
        The scores range from:
        - -1 (most negative)
        - 0 (neutral)
        - +1 (most positive)
        
        ### Tips for best results
        - Use complete sentences
        - Keep original punctuation
        - Include emoticons if relevant
        - Use natural language
        """)

if __name__ == "__main__":
    main()
