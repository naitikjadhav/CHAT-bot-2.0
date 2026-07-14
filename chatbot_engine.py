"""
chatbot_engine.py
------------------
Core NLP logic for the College Organization Chatbot.

Approach:
    1. Load intents (patterns + responses) from intents.json
    2. Convert all training patterns into TF-IDF vectors
    3. On a new user message, vectorize it and find the closest
       pattern using cosine similarity
    4. If the best match score is above a confidence threshold,
       return a random response from that intent; otherwise
       fall back to a default "I don't understand" response.

This keeps the project lightweight (no deep learning / training
required) while still demonstrating real NLP concepts, which is
ideal for a college mini-project.
"""

import json
import random
import re
import string
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class CollegeChatbot:
    def __init__(self, intents_path="intents.json", confidence_threshold=0.25):
        self.confidence_threshold = confidence_threshold
        self.intents_path = Path(intents_path)
        self.intents = self._load_intents()

        # Flattened training data: every pattern -> its tag
        self.patterns = []
        self.tags = []
        for intent in self.intents["intents"]:
            for pattern in intent["patterns"]:
                self.patterns.append(self._preprocess(pattern))
                self.tags.append(intent["tag"])

        # Build TF-IDF vectorizer over all known patterns
        self.vectorizer = TfidfVectorizer()
        if self.patterns:
            self.pattern_vectors = self.vectorizer.fit_transform(self.patterns)
        else:
            self.pattern_vectors = None

        # Quick lookup: tag -> list of responses
        self.responses_by_tag = {
            intent["tag"]: intent["responses"] for intent in self.intents["intents"]
        }

    def _load_intents(self):
        with open(self.intents_path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _preprocess(text: str) -> str:
        text = text.lower()
        text = text.translate(str.maketrans("", "", string.punctuation))
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def get_response(self, user_message: str) -> dict:
        """
        Returns a dict: { response, tag, confidence }
        """
        if not user_message.strip():
            return {
                "response": "Please type a question so I can help you.",
                "tag": "empty",
                "confidence": 0.0,
            }

        processed = self._preprocess(user_message)
        user_vector = self.vectorizer.transform([processed])

        similarities = cosine_similarity(user_vector, self.pattern_vectors)[0]
        best_idx = similarities.argmax()
        best_score = similarities[best_idx]
        best_tag = self.tags[best_idx]

        if best_score < self.confidence_threshold:
            best_tag = "fallback"

        response = random.choice(self.responses_by_tag[best_tag])

        return {
            "response": response,
            "tag": best_tag,
            "confidence": round(float(best_score), 3),
        }


if __name__ == "__main__":
    # Simple command-line test loop
    bot = CollegeChatbot()
    print("College Chatbot (type 'quit' to exit)")
    while True:
        msg = input("You: ")
        if msg.lower() in ("quit", "exit"):
            break
        result = bot.get_response(msg)
        print(f"Bot ({result['tag']}, conf={result['confidence']}): {result['response']}")
