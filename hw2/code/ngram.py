from lm import LangModel
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np


def add_lambda_smoothing(counts: int, total: int, llambda: float, vocab_size: int) -> float:
    num = counts + llambda
    denom = total + llambda * vocab_size
    if num != 0 and denom != 0:
        return np.log(num) - np.log(denom)
    else:
        return -np.inf


class Ngram(LangModel):
    """N-gram Language model implementation."""

    def __init__(self, ngram_size: int, llambda: float = 0, **kwargs):
        super().__init__(**kwargs)

        self.llambda = llambda
        self.ngram_size = ngram_size
        self.counts_totals: Dict[Tuple[str], int] = {}
        self.counts: Dict[Tuple[str], Dict[str, int]] = defaultdict(dict)

        self.unigram_counts: Dict[str, int] = {}
        self.unigram_total: int = 0

    @property
    def name(self):
        return f"{self.ngram_size}-gram"

    def fit_sentence(self, sentence: List[str]):
        for i, word_i in enumerate(sentence):
            # # get context words according to markov assumption
            # # the conditioning words for w_i, are the w_{i-k:i}
            # # (if i < k then 0 else i-k)
            # k_words_bef_i = max(0, i - k)
            # context = sentence[k_words_bef_i:i]
            self.incr_word(sentence[:i], word_i)

    def incr_word(self, context: List[str], word: str):
        """Register occurrence of word with the specified context"""
        context = self.get_context(context)

        # If context does not exist in model, initialize it
        if self.counts[context].get(word, None) is None:
            self.counts[context][word] = 1
        else:
            self.counts[context][word] += 1

        if self.counts_totals.get(context, None) is None:
            self.counts_totals[context] = 1
        else:
            self.counts_totals[context] += 1

        # ---------------------------------------------
        # update unigram counts (necessary for backoff)
        # ---------------------------------------------
        if self.unigram_counts.get(word) is None:
            self.unigram_counts[word] = 1
        else:
            self.unigram_counts[word] += 1
        self.unigram_total += 1


    def get_context(self, context: List[str]):
        """Compute the appropriate context size according to the size of
        the ngram model."""
        if self.ngram_size == 1:
            return tuple([])
        else:
            return tuple(context[-(self.ngram_size - 1):])
            # ^Note: Even if the context is empty, context[-5:] always
            # returns the empty context

    def cond_logprob(self, word: str, context: List[str]) -> float:
        """Computes the natural logarithm of the conditional probability
        of a word, given the context words.
        """
        # Collect the relevant part of the sentence given the ngram model
        context = self.get_context(context)

        logprob = 0
        # --------------------------------------------------------------
        # TODO: finish implementing this part to complete
        # --------------------------------------------------------------
        #  Ngram cond_logprob. To do this you will have to:
        #  * Compute the probability of the word given context for the
        #    current model.
        #    Hint: use `self.counts.get` to obtain the next word
        #          predictions based on `context`)
        #  * For the case where `context` does not exist in the model,
        #    compute the add-lambda smoothing using self.llambda,
        #    self.unigram_counts, and self.unigram_total
        #  * For the case where `context` was seen during training,
        #    compute the probability, p_model(word|context).
        # --------------------------------------------------------------
        p_n: Dict[Tuple[str], int] = {}
        p_uni: Dict[Tuple[str], int] = {}
        if word not in self.vocab:
            if self.counts[context].get(word, None) is None:
                counts = 0
            else:
                counts = self.counts[context][word]
            if self.unigram_counts.get(word) is None:
                uni_counts = 0
            else:
                uni_counts = self.unigram_counts[word]
            if self.counts_totals.get(context, None) is None:
                total = 0
            else:
                total = self.counts_totals[context]
            p_n[word] = add_lambda_smoothing(counts, total, self.llambda, self.vocab_size)
            p_uni[word] = add_lambda_smoothing(uni_counts, self.unigram_total, self.llambda, self.vocab_size)
            p_n[word] = np.exp(p_n[word])
            p_uni[word] = np.exp(p_uni[word])

        for w_i in self.vocab:
            if self.unigram_counts.get(w_i) is None:
                uni_counts = 0
            else:
                uni_counts = self.unigram_counts[w_i]
            if self.counts[context].get(w_i, None) is None:
                counts = 0
            else:
                counts = self.counts[context][w_i]
            if self.counts_totals.get(context, None) is None:
                total = 0
            else:
                total = self.counts_totals[context]
            p_n[w_i] = add_lambda_smoothing(counts, total, self.llambda, self.vocab_size)
            p_uni[w_i] = add_lambda_smoothing(uni_counts, self.unigram_total, self.llambda, self.vocab_size)
            p_n[w_i] = np.exp(p_n[w_i])
            p_uni[w_i] = np.exp(p_uni[w_i])

        if self.counts_totals.get(context, None) is None:  # backoff to unigram if context counts 0
            # normalize probabilities
            x = 1.0 / sum(p_uni.values())
            for w in p_uni:
                p_uni[w] = p_uni[w] * x
            logprob = np.log(p_uni[word])
        else:
            # normalize probabilities
            y = 1.0 / sum(p_n.values())
            for w in p_n:
                p_n[w] = p_n[w] * y
            logprob = np.log(p_n[word])
        # raise NotImplementedError("TO BE IMPLEMENTED BY THE STUDENT")
        # --------------------------------------------------------------
        return logprob
