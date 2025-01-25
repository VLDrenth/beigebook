from abc import ABC, abstractmethod
from typing import List, Dict, Any, Union
import numpy as np
from transformers import pipeline

class SentimentScorer:
    """
    Simple wrapper for HuggingFace models
    """
    def __init__(self, model_name="distilbert-base-uncased-finetuned-sst-2-english"):
        self.classifier = pipeline("sentiment-analysis", model=model_name)
    
    def score(self, text: Union[str, List[str]], chunk_size: int = 256) -> Union[float, List[float]]:
        """Score text sentiment, chunking long inputs"""
        if not text:
            raise ValueError("Input text cannot be empty")
            
        texts = [text] if isinstance(text, str) else text
        results = []
        
        for text in texts:
            # Split into chunks and get average sentiment
            chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
            chunk_results = self.classifier(chunks)
            avg_score = np.mean([r["score"] if r["label"] == "POSITIVE" else 1 - r["score"] 
                            for r in chunk_results])
            results.append(avg_score)
            
        return results[0] if isinstance(text, str) else results