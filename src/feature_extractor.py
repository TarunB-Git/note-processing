import numpy as np
import re
import spacy
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion

class HandcraftedFeatures(BaseEstimator, TransformerMixin):
    """
    Custom transformer to extract handcrafted features from text using spaCy.
    Features:
    - Question Markers: 1 if starts with WH-word or contains '?'
    - Task Indicators: 1 if first token is a VERB
    - Time/Deadline Markers: 1 if contains time patterns or DATE/TIME entities
    """
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        features = []
        for text in X:
            doc = self.nlp(text)
            # Question Markers
            wh_words = ['who', 'what', 'when', 'where', 'why', 'how']
            starts_with_wh = doc[0].text.lower() in wh_words if len(doc) > 0 else False
            has_question = '?' in text
            question_marker = 1 if starts_with_wh or has_question else 0

            # Task Indicators
            task_indicator = 1 if len(doc) > 0 and doc[0].pos_ == 'VERB' else 0

            # Time/Deadline Markers
            time_patterns = re.findall(r'\b(by \d+|\d+ (am|pm)|tomorrow|at \d+|next week|deadline)\b', text.lower())
            has_date_time_entity = any(ent.label_ in ['DATE', 'TIME'] for ent in doc.ents)
            time_marker = 1 if time_patterns or has_date_time_entity else 0

            features.append([question_marker, task_indicator, time_marker])
        return np.array(features)

def create_pipeline():
    """
    Creates a pipeline combining TF-IDF and handcrafted features.
    """
    tfidf = TfidfVectorizer(ngram_range=(1, 2))
    handcrafted = HandcraftedFeatures()
    union = FeatureUnion([
        ('tfidf', tfidf),
        ('handcrafted', handcrafted)
    ])
    pipeline = Pipeline([
        ('features', union)
    ])
    return pipeline

if __name__ == "__main__":
    pipeline = create_pipeline()
    test_texts = ['Buy milk tomorrow', 'Who is the CEO?', 'Meeting at 5pm']
    features = pipeline.fit_transform(test_texts)
    print(f"Output Feature Matrix Shape: {features.shape}")