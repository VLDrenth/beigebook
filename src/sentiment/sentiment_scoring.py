from huggingface_hub import InferenceClient
from typing import List, Union
from nltk.tokenize import sent_tokenize
import numpy as np
import os

class SentimentScorer:
    def __init__(self, token: str = None):
        if token is None:
            token = os.getenv("HF_TOKEN")  # Store token as environment variable
        
        self.client = InferenceClient(
            model="ProsusAI/finbert",
            token=token
        )
    
    def score(self, text: str = None, batch_size: int = 64) -> float:
        """
        Score text sentiment by splitting into chunks and processing in a single batch.
        
        Args:
            text (str): Input text to analyze
            chunk_size (int): Size of chunks to split text into
                
        Returns:
            float: Average sentiment score between 0 and 1
        """
        if not text:
            raise ValueError("Input text cannot be empty")

        sentences = sent_tokenize(text)
        scores = []
    
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i+batch_size]
            results = self.client.text_classification(batch)
            scores.extend(results)
            
        scores_filtered = [res["score"] if res["label"] == "positive" else -res["score"] 
                  for res in scores if res["label"] in ["positive", "negative"]]

        if not scores_filtered:
            raise ValueError("No scores were found")

        # Calculate average sentiment score
        avg_score = np.mean([res for res in scores_filtered])  
        return avg_score