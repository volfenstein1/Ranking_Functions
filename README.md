# Ranking Functions

This repository provides textbook implementations in python of various ranking functions, specifically cosine similarity, tf-idf, and BM25 ranking functions.

Our general setup is as follows.
We are given a corpus $D$ consisting of documents $d \in D$.
For a given query $Q$ consisting of keywords $q_1,\ldots, q_n$ we use a **ranking function** to assign a score to each document.
The documents will be ranked according to this score, with higher score corresponding to a higher document relevance.

## [BoW](https://en.wikipedia.org/wiki/Bag-of-words_model)

Given a text, the **bag of words** representation is a count of the frequency of each word in the text.
For example:

| Text                                                                                          | Bag of Words                                                                                      |
| --------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| "John likes to watch movies. Mary likes movies too. Mary also likes to watch football games." | {"John":1,"likes":3,"to":2,"watch":2,"movies":2,"Mary":2,"too":1,"also":1,"football":1,"games":1} |

Note that this representation does not capture any information involving word order; the bag of words representations of the texts:

- Yesterday, investors were rallying, but today, they are retreating.
- Yesterday, investors were retreating, but today, they are rallying.

are identical.

## [Cosine Similarity](https://en.wikipedia.org/wiki/Cosine_similarity)

Given two texts, consider their representations as bag of words, call them $A$ and $B$. We can consider these bag of words as sparse vectors in a vector space with a basis consisting of all words, and coefficients then given by the word frequencies.

The **cosine similarity** between $A$ and $B$ is the cosine of the angle $\theta$ between these two vectors, and is given by:

$$
 \text{cosine} (\theta) = \frac{A \dot B}{\|A\| \|B\|}.
$$

In this case, the cosine similarity takes values in $[0,1]$; this is because the coefficients of a bag of words are a frequency count, and cannot be negative. Values closer to $1$ indicate a closer match, and values closer to $0$ indicate a worse match.

The file `cosine-similarity.py` includes an implementation of a document ranking system which uses cosine similarity as the ranking function.

## [tf-idf](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)

Given a term $t$ and a document $d$, **tf-idf** is the product of two statistics: term frequency (tf) and inverse document frequency (idf).

The **term frequency** is defined as:

$$
  \text{tf}(t,d) = \frac{f_{t,d}}{ \sum_{t' \in d} f_{t', d} }
$$

where

- $f_(t,d)$ is the number of times term $t$ appears in document $d$,
- the denominator is the total number of terms in the document $d$.

Observe $\text{tf}(t,d) \in [0,1]$; it is $0$ if the term does not appear, and $1$ if the document consists of the term and nothing else.

The **inverse document frequency** is the "logarithmically scaled inverse fraction of the documents which contain the term". It is defined as:

$$
  \text{idf}(t) = \log (\frac{N}{n_t})
$$

where

- $N$ is the total number of documents in the corpus
- $n_t$ is the count of documents $d in D$ where the term appears, i.e., $|{d in D | t in d}|$.

Hence $\tfrac{n_t}{N}$ is the fraction of documents where the term appears; $\tfrac{N}{n_t}$ is the inverse of this fraction.
Observe $\tfrac{n_t}{N} \in [0,1]$; hence $\text{idf}(t) \in [0,\infty]$
To avoid division by zero, it is common to adjust the numerator / denominator as $1+N$ / $1+n_t$, respectively.

Given a query $Q$ consisting of terms $q_1,...,q_n$ and a document $d$ belonging to a corpus $D$, the **tf-idf** score is calculated as:

$$
  \text{tf-idf} (Q, d) = \sum_{q_i \in Q} \text{tf-idf} (q_i, d) =\sum_{q_i \in Q} \text{tf} (q_i, d) \cdot \text{idf}(q_i)
$$

We can summarize its behavior as follows:

- As a term $t$ increases in frequency in the document $d$ we have: $\text{tf}(t,d) \to 1$.
- Likewise as the term $t$ decreases in frequency in the document $d$ we have: $\text{tf}(t,d) \to 0$.
- As a term $t$ increases in frequency in the corpus $D$, we have: $\tfrac{n_t}{N} \to 1^-$ and hence $\text{idf}(t) = \log(\tfrac{N}{n_t}) \to 0^+$.
- As a term $t$ decreases in frequency in the corpus $D$, we have: $\tfrac{n_t}{N} \to 0^+$ and hence $\text{idf}(t) = \log(\tfrac{N}{n_t}) \to \infty$.

The file `tf-idf.py` includes an implementation of a document ranking system which uses tf-idf as the ranking function.

## [BM25](https://en.wikipedia.org/wiki/Okapi_BM25)

The **BM25** score can be viewed as a modified version of tf-idf with different asymptotic behaviors.
Given a query $Q$ consisting of terms $q_1,...,q_n$ and a document $d$ belonging to a corpus $D$, the BM25 score is calculated as:

$$
  \text{BM}25(Q,d) = \sum_{q_i \in Q} \text{idf}(q_i) \frac{f_{q_i,d} \cdot (k + 1) }{ f_{q_i,d} + k (1 - b + b \tfrac{|d|}{\text{avgdl}}) }
$$

where

- $k$ and $b$ are free parameters, often set to $k=1.2$ or $k=2.0$ and $b=0.75$,
- $|d|$ is the length of the document $d$,
- $\text{avgdl}$ is the average length of a document in the corpus,
- $f_{q_i,d}$ is the number of times the term $q_i$ appears in the document $d$,
- $\text{idf}(q_i) = \log ( \frac{N - n_{q_i} + 0.5}{n_{q_i} + 0.5} + 1)$

We can summarize its behavior as follows:

- As a term $t$ increases in frequency in the document $d$ we have: $\tfrac{f_{q_i,d} \cdot (k + 1) }{ f_{q_i,d} + k (1 - b + b \tfrac{|d|}{\text{avgdl}}) } \to k+1$
- From the above, we see how the term $k$ controls the asymptotic behavior as a term increases in frequency in a document.
- Likewise as the term $t$ decreases in frequency in the document $d$ we have: $\tfrac{f_{q_i,d} \cdot (k + 1) }{ f_{q_i,d} + k (1 - b + b \tfrac{|d|}{\text{avgdl}}) } \to 0$
- As the length of $d$ increases we have: $\tfrac{f_{q_i,d} \cdot (k + 1) }{ f_{q_i,d} + k (1 - b + b \tfrac{|d|}{\text{avgdl}}) } \to \tfrac{f_{q_i,d} \cdot (k + 1) }{ f_{q_i,d} + k \cdot \text{large correction} } $
- As the length of $d$ decreases we have: $\tfrac{f_{q_i,d} \cdot (k + 1) }{ f_{q_i,d} + k (1 - b + b \tfrac{|d|}{\text{avgdl}}) } \to \tfrac{f_{q_i,d} \cdot (k + 1) }{ f_{q_i,d} + k (1 - b ) }$
- As a term $q_i$ increases in frequency in the corpus $D$, we have: $\tfrac{n_{q_i} + 0.5}{N - n_{q_i} + 0.5} \to \infty$ and hence $\text{idf}(q_i) \to 0^+$.
- As a term $q_i$ decreases in frequency in the corpus $D$, we have: $\tfrac{n_{q_i} + 0.5}{N - n_{q_i} + 0.5} \to 0$ and hence $\text{idf}(t) \to \infty$.

The file `BM25.py` includes an implementation of a document ranking system which uses BM25 as the ranking function.
