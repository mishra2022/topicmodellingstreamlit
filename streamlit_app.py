import streamlit as st
from bertopic import BERTopic
import pandas as pd
import plotly.express as px
def load_data(uploaded_file, nrows, text_column):
    df = pd.read_csv(uploaded_file)
    df = df.head(nrows)
    return df[text_column].tolist()
def run_bertopic(text_data):
    topic_model = BERTopic()
    topics, probs = topic_model.fit_transform(text_data)
    return topic_model, topics, probs
st.title('BERTopic Modeling')
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    nrows = st.number_input('Number of rows to analyze', min_value=10, value=100)
    df = pd.read_csv(uploaded_file)
    column = st.selectbox('Select text column for analysis', df.columns)
    docs = df[column].head(nrows).tolist()
if st.button('Run BERTopic'):
        with st.spinner('Running BERTopic model...'):
            topic_model, topics, _ = run_bertopic(docs)
            st.write(f'Topics found: {topic_model.get_topic_info()}')

            # Visualize topics
            fig = topic_model.visualize_barchart(top_n_topics=5)
            st.plotly_chart(fig)

            # Visualize topic hierarchy
            fig_hierarchy = topic_model.visualize_hierarchy(top_n_topics=5)
            st.plotly_chart(fig_hierarchy)

            #Visualize Topics in a 2-D space
            fig_topics_viz = topic_model.visualize_topics()
            st.plotly_chart(fig_topics_viz)
