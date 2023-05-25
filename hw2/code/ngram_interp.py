from lm import LangModel
from ngram import Ngram
from typing import List

import numpy as np


class InterpNgram(LangModel):
    """Interpolated N-gram Language Model with backoff"""

    def __init__(self, ngram_size: int, alpha: float, llambda: float, **kwargs):
        super().__init__(**kwargs)
        assert 0 < alpha < 1
        assert 0 <= llambda
        assert 0 < ngram_size and isinstance(ngram_size, int)

        if ngram_size == 2:
            self.backoff_model = Ngram(1, llambda=llambda, **kwargs)
        else:
            self.backoff_model: InterpNgram = InterpNgram(ngram_size - 1, alpha, llambda=llambda, **kwargs)

        self.alpha = alpha
        self.model = Ngram(ngram_size, llambda=llambda, **kwargs)
        self.ngram_size = ngram_size

    @property
    def name(self):
        return f"interp_{self.ngram_size}-gram"

    def fit_sentence(self, sentence: List[str]):
        for i, word_i in enumerate(sentence):
            self.incr_word(sentence[:i], word_i)

    def incr_word(self, context: List[str], word: str):
        self.model.incr_word(context, word)
        self.backoff_model.incr_word(context, word)

    def cond_logprob(self, word: str, context: List[str]) -> float:
        curr_context = self.model.get_context(context)

        logprob = 0
        # ---------------------------------------------------------------------
        # TODO: finish implementing this part to complete
        # ---------------------------------------------------------------------
        #  Interpolated cond_logprob. To do this you will have to:
        #  * Compute the probability of the word given context for the current
        #    model. (Hint: use `self.model.counts.get` to obtain the next word
        #    predictions based on `context`)
        #  * If the context does not exist in, backoff to `self.backoff_model`.
        #  * If the context exists, compute the next-word probability estimate
        #    using p_{K}(w|context) (self.model) and multiply it by alpha.
        #  * Compute the probability assigned by a lower order interpolated
        #    n-gram model and multiply it by (1-\alpha) as follows:
        #    (1-alpha) * I_{K-1}(w|context_{-(k-2):}).
        #    (Hint: use the self.backoff_model to compute this probability).
        #
        # Note: Remember that the distributions are in logprobabilities.
        # Instead of exponentiating, summing the probabilities and then taking
        # the log again, a more stable operation is to apply logsumexp or, in
        # numpy, the `np.logaddexp`.
        # ---------------------------------------------------------------------
        current_model = self
        while current_model.model.counts_totals.get(curr_context, None) is None:
            current_model = current_model.backoff_model
            if current_model.ngram_size == 1:
                curr_context = current_model.get_context(context)
                break
            curr_context = current_model.model.get_context(context)

        kgram_size = current_model.ngram_size
        interp_list = []
        for i in range(kgram_size):
            interp_list.append(current_model)
            if i == kgram_size - 1:
                break
            current_model = current_model.backoff_model

        curr_context = interp_list[kgram_size - 1].get_context(context)
        kgram_prob = []
        kgram_prob.append(np.exp(interp_list[kgram_size - 1].cond_logprob(word,curr_context)))

        for j in range(kgram_size - 1):
            curr_context = interp_list[kgram_size - 2 - j].model.get_context(context)
            a = interp_list[kgram_size - 2 - j].alpha
            kgram_prob.append(a * np.exp(interp_list[kgram_size - 2 - j].model.cond_logprob(word,curr_context)) + (1 - a) * kgram_prob[j])

        logprob = np.log(kgram_prob[kgram_size - 1])
        #raise NotImplementedError("TO BE IMPLEMENTED BY THE STUDENT")
        # ---------------------------------------------------------------------
        return logprob
