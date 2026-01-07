import heapq
import math
import re
from collections import Counter


class BoW_Cosine_Similarity:
    def __init__(self, corpus):
        """
        Given 'corpus' (a list consisting of strings which each represent a document):
        1. construct the vocabulary
        2. represent each document in the corpus as a BoW
        """
        self.corpus = corpus
        self.corpus_BoWs = []
        self.vocabulary = set()

        # 1. construct vocabulary
        for doc in corpus:
            words = self._get_words(doc)
            self.vocabulary.update(words)

        # 2. create BoWs from each document
        self.corpus_BoWs = [self._string_to_BoW(doc) for doc in corpus]

    def _get_words(self, text):
        """
        Given a string 'text':
        1. standardize by converts to lowercase and removing punctuation
        2. split on whitespace
        """
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", "", text)
        return text.split()

    def _string_to_BoW(self, text):
        """
        Converts a text into a BoW.
        Ignores words not in the vocabulary.
        """
        text_words = self._get_words(text)
        text_word_counts = Counter(text_words)
        text_BoW = {
            k: v for k, v in text_word_counts.items() if k in self.vocabulary
        }
        return text_BoW

    def _get_cosine_similarity(self, bow_a, bow_b):
        """
        Calculates the cosine similarity between two BoWs.
        Formula: (A . B) / (||A|| * ||B||)
        """
        dot_product = 0
        for word in bow_a:
            if word in bow_b:
                dot_product += bow_a[word] * bow_b[word]

        magnitude_a = math.sqrt(sum(v**2 for _, v in bow_a.items()))
        magnitude_b = math.sqrt(sum(v**2 for _, v in bow_b.items()))

        if magnitude_a == 0 or magnitude_b == 0:
            return 0.0

        cosine_similarity = dot_product / (magnitude_a * magnitude_b)

        return cosine_similarity

    def search(self, query, k=10):
        """
        Given a query, return the indices of the top k documents.
        """
        query_BoW = self._string_to_BoW(query)

        # Scores is a min heap
        scores = []

        for idx, doc_BoW in enumerate(self.corpus_BoWs):
            score = self._get_cosine_similarity(query_BoW, doc_BoW)
            if len(scores) < k:
                heapq.heappush(scores, (score, idx))
            else:
                heapq.heappushpop(scores, (score, idx))

        return heapq.nlargest(k, scores)


# --- Example Usage ---

corpus = [
    "The quick brown fox jumps over the dog",
    "The quick brown fox is very quick",
    "Dogs are great pets and friends",
    "Cats and dogs are enemies",
]

ranker = BoW_Cosine_Similarity(corpus)

query = "quick brown fox"
results = ranker.search(query)

print(f"Query: '{query}'\n")
print("Score  | idx | Document")
print("-" * 60)
for score, idx in results:
    print(f"{score:.4f} | {idx}   | {corpus[idx]}")
