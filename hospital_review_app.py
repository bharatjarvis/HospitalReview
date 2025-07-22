#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# hospital_review_app.py

import streamlit as st
import pandas as pd
import numpy as np
import re
import string
import nltk
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from transformers import pipeline
from gensim import corpora, models
from wordcloud import WordCloud
from sentence_transformers import SentenceTransformer, util
from textblob import TextBlob

# Setup
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Title
st.title("🏥 Hospital Review Analysis Dashboard")

# Upload CSV
uploaded_file = st.file_uploader("Upload your hospital review CSV", type=["csv"])
if not uploaded_file:
    st.stop()

df = pd.read_csv(uploaded_file)

# Text Cleaning
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(f"[{string.punctuation}]", '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['clean_review'] = df['review'].apply(clean_text)
st.success("✅ Step 1: Text cleaned")

# Sentiment Analysis
emotion_model = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=False, from_pt=True)

@st.cache_data
def get_emotion(text):
    try:
        return emotion_model(text[:512])[0]['label']
    except:
        polarity = TextBlob(text).sentiment.polarity
        if polarity > 0.1:
            return "joy"
        elif polarity < -0.1:
            return "anger"
        else:
            return "neutral"

df['dominant_emotion'] = df['clean_review'].apply(get_emotion)
st.success("✅ Step 2: Emotion Analysis Done")

# Tokenization
df['tokens_raw'] = df['clean_review'].apply(word_tokenize)
df['tokens'] = df['tokens_raw'].apply(lambda x: [w for w in x if w not in stop_words and len(w) > 2])

# LDA Topic Modeling
dictionary = corpora.Dictionary(df['tokens'])
corpus = [dictionary.doc2bow(text) for text in df['tokens']]
lda_model = models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=7, passes=10, random_state=42)

def get_dominant_topic(bow):
    topics = lda_model.get_document_topics(bow)
    return max(topics, key=lambda x: x[1])[0] if topics else None

df['dominant_topic'] = [get_dominant_topic(bow) for bow in corpus]

# Extract Topic Keywords
topic_keywords = {}
for topic_num, topic_str in lda_model.print_topics():
    keywords = [kw.split("*")[1].replace('"', '').strip() for kw in topic_str.split("+")]
    topic_keywords[topic_num] = keywords

df['topic_keywords'] = df['dominant_topic'].map(lambda t: ", ".join(topic_keywords.get(t, [])))

# Word Clouds
st.subheader("☁️ Topic Word Clouds")
for topic_num, keywords in topic_keywords.items():
    topic_text = " ".join(df[df['dominant_topic'] == topic_num]['clean_review'])
    wordcloud = WordCloud(background_color='white', max_words=100).generate(topic_text)
    st.markdown(f"**Topic {topic_num}:** {', '.join(keywords)}")
    st.image(wordcloud.to_array())

# Semantic Mapping
categories = [
    "staff behavior", "waiting time", "cost", "doctor professionalism", 
    "facility cleanliness", "billing process", "diagnosis accuracy",
    "nursing care", "pharmacy service", "emergency response",
    "room conditions", "insurance process"
]
model = SentenceTransformer('all-MiniLM-L6-v2')
topic_labels = {}
for topic_num, keywords in topic_keywords.items():
    topic_sentence = ", ".join(keywords)
    topic_emb = model.encode(topic_sentence, convert_to_tensor=True)
    category_embs = model.encode(categories, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(topic_emb, category_embs)[0]
    best_match_idx = int(similarities.argmax())
    topic_labels[topic_num] = categories[best_match_idx]

df['semantic_topic'] = df['dominant_topic'].map(topic_labels)

# Polarity & Suggestion Type
df['polarity'] = df['clean_review'].apply(lambda x: TextBlob(x).sentiment.polarity)

def suggestion_type(p):
    if p < -0.1:
        return 'improvement'
    elif p > 0.1:
        return 'exploration'
    else:
        return 'neutral'

df['suggestion_type'] = df['polarity'].apply(suggestion_type)

# Prescriptions
suggestions = {
    "staff behavior": {
        "improvement": "Train staff in soft skills and empathy.",
        "exploration": "Recognize and reward staff praised by patients."
    },
    "waiting time": {
        "improvement": "Optimize scheduling and queue management systems.",
        "exploration": "Explore digital queuing systems or appointment apps."
    },
    "cost": {
        "improvement": "Review pricing transparency and offer affordable packages.",
        "exploration": "Introduce preventive health packages or loyalty discounts."
    },
    "doctor professionalism": {
        "improvement": "Regular peer evaluations and patient feedback.",
        "exploration": "Create a recognition program for highly rated doctors."
    },
    "facility cleanliness": {
        "improvement": "Increase cleaning frequency and audits.",
        "exploration": "Highlight cleanliness ratings in marketing materials."
    },
    "billing process": {
        "improvement": "Simplify and digitize billing with transparency.",
        "exploration": "Introduce real-time billing status updates via app."
    },
    "diagnosis accuracy": {
        "improvement": "Ensure protocol adherence and offer second opinion systems.",
        "exploration": "Pilot AI-assisted diagnostics for complex cases."
    },
    "nursing care": {
        "improvement": "Enhance nurse training and patient care protocols.",
        "exploration": "Enable nurse-patient appreciation boards."
    },
    "pharmacy service": {
        "improvement": "Ensure availability and pricing transparency of medications.",
        "exploration": "Provide digital prescriptions and drug info sheets."
    },
    "emergency response": {
        "improvement": "Improve triage and emergency handling protocols.",
        "exploration": "Implement fast-track systems for frequent emergency cases."
    },
    "room conditions": {
        "improvement": "Upgrade room facilities and maintain hygiene.",
        "exploration": "Offer customizable room environment settings (e.g., lighting, music)."
    },
    "insurance process": {
        "improvement": "Streamline insurance claim process and coordination.",
        "exploration": "Introduce dedicated insurance support desks or digital portals."
    }
}

suggestion_summary = df.groupby(['semantic_topic', 'suggestion_type']).size().unstack(fill_value=0)

# Final Prescription
final_prescriptions = {}
for topic in suggestion_summary.index:
    topic_counts = suggestion_summary.loc[topic]
    selected_type = 'improvement' if topic_counts.get('improvement', 0) >= topic_counts.get('exploration', 0) else 'exploration'
    if topic in suggestions and selected_type in suggestions[topic]:
        final_prescriptions[topic] = {
            'suggestion_type': selected_type,
            'text': suggestions[topic][selected_type]
        }

# Show Prescriptions
st.subheader("🧠 Final Prescriptions")
for topic, info in final_prescriptions.items():
    st.markdown(f"### {topic.title()} ({info['suggestion_type'].capitalize()})")
    st.write(info['text'])

# Visualization
st.subheader("📊 Suggestion Distribution")
st.bar_chart(suggestion_summary)

# Data Preview
st.subheader("🔍 Sample Insights")
st.dataframe(df[['review', 'semantic_topic', 'polarity', 'suggestion_type']].sample(10))

