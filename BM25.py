import heapq
import math
import re
from collections import Counter


class bm25:
    def __init__(self, corpus, b=0.75, k=1.2):
        """
        Given 'corpus' (a list consisting of strings which each represent a document):
        1. construct the vocabulary
        2. represent each document in the corpus as a BoW
        3. for each term calculate the inverse document frequency
        """
        self.b = b
        self.k = k
        self.corpus = corpus
        self.corpus_BoWs = []
        self.corpus_lengths = []
        self.avg_len = 0
        self.vocabulary = set()
        self.inverse_document_frequency = {}

        # 1. construct vocabulary
        for doc in corpus:
            words = self._get_words(doc)
            self.vocabulary.update(words)

        # 2. create BoWs from each document
        self.corpus_BoWs = [self._string_to_BoW(doc) for doc in corpus]

        # 3. calculate length of each document
        self.corpus_lengths = [
            sum(v for _, v in doc_BoW.items()) for doc_BoW in self.corpus_BoWs
        ]

        # 4. calculate average document length
        self.avg_len = sum(self.corpus_lengths) / len(self.corpus_lengths)

        # 5. calculate inverse document frequency
        for term in self.vocabulary:
            self.inverse_document_frequency[term] = (
                self._get_inverse_document_frequency(term)
            )

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

    def _get_inverse_document_frequency(self, term):
        """
        Given a term the inverse document frequency of a corpus is computed as:
            log ( (N - n_term + 0.5) / (n_term + 0.5) + 1)
        where:
        - N = size of the corpus
        - n_term = number of documents in the corpus where the term appears
        """
        N = len(self.corpus)
        n_term = 0
        for doc_BoW in self.corpus_BoWs:
            if term in doc_BoW:
                n_term += 1
        return math.log((N - n_term + 0.5) / (n_term + 0.5) + 1)

    def _get_BM25(self, query_BoW, doc_BoW):
        """
        Given a query and a document both represented as a BoW, returns the BM25 score, computed as:
            BM25()
        """
        score = 0
        for term in query_BoW:
            if term in self.vocabulary and term in doc_BoW:
                numerator = doc_BoW[term] * (self.k + 1)
                denominator = doc_BoW[term] + self.k * (
                    1
                    - self.b
                    + self.b
                    * (sum(v for _, v in doc_BoW.items()) / self.avg_len)
                )
                score += (
                    query_BoW[term]
                    * self.inverse_document_frequency[term]
                    * numerator
                    / denominator
                )
        return score

    def search(self, query, k=10):
        """
        Given a query, return the indices of the top k documents.
        """
        query_BoW = self._string_to_BoW(query)

        # Scores is a min heap
        scores = []

        for idx, doc_BoW in enumerate(self.corpus_BoWs):
            score = self._get_BM25(query_BoW, doc_BoW)
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

ranker = bm25(corpus)

query = "quick brown fox"
results = ranker.search(query)

print(f"Query: '{query}'\n")
print("Score  | idx | Document")
print("-" * 60)
for score, idx in results:
    print(f"{score:.4f} | {idx}   | {corpus[idx]}")
