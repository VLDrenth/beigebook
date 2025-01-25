from huggingface_hub import InferenceClient
from typing import List, Union
import numpy as np
import os

class SentimentScorer:
    def __init__(self, token: str = None):
        if token is None:
            token = os.getenv("HF_TOKEN")  # Store token as environment variable
        
        self.client = InferenceClient(
            model="distilbert-base-uncased-finetuned-sst-2-english",
            token=token
        )
    
    def score(self, text: str = None, chunk_size: int = 512) -> float:
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

        # Split text into chunks of specified size
        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        
        # Process all chunks in a single API call
        results = self.client.text_classification(chunks)
        
        # Calculate average sentiment score
        avg_score = np.mean([
            r["score"] if r["label"] == "POSITIVE" else 1 - r["score"] 
            for r in results
        ])
        
        return avg_score