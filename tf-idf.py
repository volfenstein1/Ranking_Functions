import heapq
import math
import re
from collections import Counter


class tf_idf:
    def __init__(self, corpus):
        """
        Given 'corpus' (a list consisting of strings which each represent a document):
        1. construct the vocabulary
        2. represent each document in the corpus as a BoW
        3. for each term calculate the inverse document frequency
        """
        self.corpus = corpus
        self.corpus_BoWs = []
        self.corpus_lengths = []
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

        # 4. calculate inverse document frequency
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
        print(text)
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

    def _get_term_frequency(self, term, doc_BoW):
        """
        Given a term and a document represented as a BoW, the term frequency is computed as:
            f_{t,d} / sum_{t' in d} f_{t',d}
        where:
        - f_{t,d} is the number of times the term t appears in document d
        - sum_{t' in d} f_{t',d} is the total number of terms in document d

        """
        if term not in doc_BoW:
            return 0
        f_td = doc_BoW[term]
        doc_length = self.corpus_lengths[idx]
        return f_td / doc_length

    def _get_inverse_document_frequency(self, term):
        """
        Given a term the inverse document frequency of a corpus is computed as:
            log(1+N / 1+n_term)
        where:
        - N = size of the corpus
        - n_term = number of documents in the corpus where the term appears
        """
        N = len(self.corpus)
        n_term = 0
        for doc_BoW in self.corpus_BoWs:
            if term in doc_BoW:
                n_term += 1
        return math.log((1 + N) / (1 + n_term))

    def _get_tfidf(self, query_BoW, doc_BoW):
        """
        Given a query and a document both represented as a BoW, returns the tf-idf score, computed as:
            sum_{t in query} tf-idf(t,d)
        where for a single term t:
            tf-idf(t,d) = tf(t,d) * idf(t)
        """
        score = 0
        for term in query_BoW:
            if term in self.vocabulary:
                score += (
                    self._get_term_frequency(term, doc_BoW)
                    * self.inverse_document_frequency[term]
                    * query_BoW[term]
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
            score = self._get_tfidf(query_BoW, doc_BoW)
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

ranker = tf_idf(corpus)

query = "quick brown brown fox"
results = ranker.search(query)

print(f"Query: '{query}'\n")
print("Score  | idx | Document")
print("-" * 60)
for score, idx in results:
    print(f"{score:.4f} | {idx}   | {corpus[idx]}")
