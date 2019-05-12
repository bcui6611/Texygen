import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.contrib.seq2seq import Helper


class ThresholdHelper(Helper):
    """
        This helper can:
            1. Translate discrete tokens into embeddings
            2. Take tokens from the given sequence before a threshold T,
            and sample tokens after T so that:
                pretrain setting equals to T==seq_len
                generate setting equals to T==0
                rollout setting equals to T==given_len
            3. Add a decision layer after RNN
    """
    def __init__(
        self,
        threshold,
        seq_len,
        embedding,
        given_tokens,  # known input
        start_tokens,  # unknown input
        decision_variables,  # (W, b)
    ):
        self._threshold = threshold
        self._seq_len = seq_len
        self._batch_size = tf.size(start_tokens)

        if callable(embedding):
            self._embedding_fn = embedding
        else:
            self._embedding_fn = (
                lambda ids: tf.nn.embedding_lookup(embedding, ids))

        self._given_tokens = tf.convert_to_tensor( given_tokens, dtype=tf.int32, name="given_input")
        self._given_emb = self._embedding_fn(self._given_tokens)

        self._start_tokens = tf.convert_to_tensor(
            start_tokens, dtype=tf.int32, name="start_tokens")
        self._start_emb = self._embedding_fn(self._start_tokens)

        self._decision_W, self._decision_b = decision_variables

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def sample_ids_shape(self):
        return tf.TensorShape([])

    @property
    def sample_ids_dtype(self):
        return tf.int32

    def initialize(self, name=None):
        """Returns `(initial_finished, initial_inputs)`."""
        finished = tf.tile([False], [self._batch_size])
        return (finished, self._start_emb)

    def sample(self, time, outputs, state, name=None):
        """Returns `sample_ids`."""
        prob = tf.nn.softmax(
            tf.matmul(outputs, self._decision_W)+self._decision_b)
        log_prob = tf.log(prob)

        sample_ids = tf.cast(
            tf.reshape(
                tf.multinomial(log_prob, 1),
                [self._batch_size]
            ), tf.int32)
        return sample_ids

    def next_inputs(self, time, outputs, state, sample_ids, name=None):
        """Returns `(finished, next_inputs, next_state)`."""
        next_time = time + 1
        finished = tf.greater_equal(next_time, self._seq_len)

        if self._threshold <= 0:
            next_inputs = self._embedding_fn(sample_ids)
        else:
            next_inputs = tf.cond(
                tf.logical_or(tf.greater_equal(next_time, self._threshold),
                              finished),
                lambda: self._embedding_fn(sample_ids),
                lambda: self._given_emb[:, time, :],
            )
        return (finished, next_inputs, state)