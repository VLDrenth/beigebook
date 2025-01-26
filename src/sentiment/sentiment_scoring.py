import numpy as np
import os
import json

from dataclasses import dataclass
from typing import List
from abc import ABC, abstractmethod
from typing import Optional, List
from huggingface_hub import InferenceClient
from openai import OpenAI
from nltk.tokenize import sent_tokenize


@dataclass
class SentimentScore:
    label: str
    text: str
    chunk_id: int
    score: float

@dataclass
class DocumentSentiment:
    scores: List[SentimentScore]
    date: str
    region: str
    year: str
    month: str
    
    @property
    def median_score(self) -> float:
        # Calculate weighted median based on all scores
        all_scores = []
        for sentiment in self.scores:
            if sentiment.positive > max(sentiment.negative, sentiment.neutral):
                all_scores.append(1)
            elif sentiment.negative > max(sentiment.positive, sentiment.neutral):
                all_scores.append(-1)
        return np.median(all_scores) if all_scores else 0

class BaseSentimentScorer(ABC):
    """Abstract base class for sentiment scoring implementations."""
    
    @abstractmethod
    def score(self, text: str, batch_size: int = 64) -> DocumentSentiment:
        """Score text sentiment and return a normalized score.
        
        Args:
            text: Input text to analyze
            batch_size: Size of batches for processing (if applicable)
                
        Returns:
            float: Sentiment score between -1 and 1
        """
        pass

class HuggingFaceSentimentScorer(BaseSentimentScorer):
    """Sentiment scorer using HuggingFace's FinBERT model."""
    
    def __init__(self, token: Optional[str] = None):
        if token is None:
            token = os.getenv("HF_TOKEN")
            
        self.client = InferenceClient(
            model="https://j1jnt8vxqmcxavjl.us-east-1.aws.endpoints.huggingface.cloud",
            token=token
        )
    
    def score(self, text: str) -> DocumentSentiment:
        if not text:
            raise ValueError("Input text cannot be empty")

        chunks = sent_tokenize(text)
        sentiment_scores = []

        for idx, chunk in enumerate(chunks):
            result = self.client.text_classification(chunk, top_k=1)
            sentiment_scores.append(SentimentScore(
                label=result["label"],
                text=chunk,
                chunk_id=idx,
                score=result["score"]
            ))

        if not sentiment_scores:
            raise ValueError("No scores were found")

        return sentiment_scores

class OpenAISentimentScorer(BaseSentimentScorer):
    """Sentiment scorer using OpenAI's API."""
    
    def __init__(self, token: Optional[str] = None, model: str = "gpt-4-turbo"):
        self.token = token or os.getenv("OPENAI_API_KEY")
        self.model = model
        
    def score(self, text: str, batch_size: int = 64) -> float:
        """The batch_size parameter is included for interface consistency but not used."""
        if not text:
            raise ValueError("Input text cannot be empty")

        client = OpenAI(api_key=self.token)
        
        messages = [
            {"role": "system", "content": """You are a macroeconomic sentiment analyzer. 
            Analyze the sentiment of the given text and return a JSON object with a 
            'score' field containing a float between -1 (most negative) and 1 (most positive).
            A text indicating heavy distress or a big recession would be -1, while a historic boom of economic growth might be 1.
            """},
            {"role": "user", "content": f"Analyze the sentiment of: {text}"}
        ]
        
        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            response_format={"type": "json_object"}
        )
        
        try:
            result = json.loads(response.choices[0].message.content)
            return float(result["score"])
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            raise ValueError(f"Failed to parse LLM response: {e}")

class SentimentScorer:
    """Factory class for creating sentiment scorers with consistent interface."""
    
    def __init__(self, scorer_type: str = "huggingface", token: Optional[str] = None, 
                 model: str = "gpt-4"):
        """Initialize the appropriate sentiment scorer.
        
        Args:
            scorer_type: Type of scorer to use ("huggingface" or "openai")
            token: API token/key for the chosen service
            model: Model identifier (only used for OpenAI)
        """
        if scorer_type == "huggingface":
            self.scorer = HuggingFaceSentimentScorer(token)
        elif scorer_type == "openai":
            self.scorer = OpenAISentimentScorer(token, model)
        else:
            raise ValueError(f"Unknown scorer type: {scorer_type}")
    
    def score(self, text: str, batch_size: int = 64) -> float:
        """Score text sentiment using the configured implementation."""
        return self.scorer.score(text, batch_size)