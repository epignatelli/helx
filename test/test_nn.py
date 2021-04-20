import jax
from helx.nn import rnn


def test_lstm_cell():
    rng = jax.random.PRNGKey(0)
    input_shape = (8,)
    lstm = rnn.LSTMCell(16)
    x = jax.random.normal(rng, input_shape)
    out_shape, (params, prev_state) = lstm.init(rng, input_shape)
    outputs, state = lstm.apply(params, x, prev_state=prev_state)
    print(outputs.shape, state.h.shape, state.c.shape)
    jax.jit(lstm.apply)(params, x, prev_state=prev_state)
    jax.grad(lambda l: sum(lstm.apply(params, x, prev_state=prev_state)[0]))(1.0)


def test_lstm():
    rng = jax.random.PRNGKey(0)
    SEQ_LEN = 5
    INPUT_FEATURES = 8
    HIDDEN_SIZE = 16
    input_shape = (SEQ_LEN, INPUT_FEATURES)
    lstm = rnn.LSTM(HIDDEN_SIZE)
    x = jax.random.normal(rng, input_shape)
    out_shape, (params, prev_state) = lstm.init(rng, input_shape)
    outputs, prev_state = lstm.apply(params, x, prev_state=prev_state)
    print(outputs.shape)
    jax.jit(lstm.apply)(params, x, prev_state=prev_state)
    jax.grad(lambda l: sum(sum(lstm.apply(params, x, prev_state=prev_state)[0])))(1.0)


def test_gru_cell():
    rng = jax.random.PRNGKey(0)
    input_shape = (8,)
    gru = rnn.LSTMCell(16)
    x = jax.random.normal(rng, input_shape)
    out_shape, (params, prev_state) = gru.init(rng, input_shape)
    outputs, state = gru.apply(params, x, prev_state=prev_state)
    print(outputs.shape, state.h.shape, state.c.shape)
    jax.jit(gru.apply)(params, x, prev_state=prev_state)
    jax.grad(lambda l: sum(gru.apply(params, x, prev_state=prev_state)[0]))(1.0)


def test_gru():
    rng = jax.random.PRNGKey(0)
    SEQ_LEN = 5
    INPUT_FEATURES = 8
    HIDDEN_SIZE = 16
    input_shape = (SEQ_LEN, INPUT_FEATURES)
    gru = rnn.LSTM(HIDDEN_SIZE)
    x = jax.random.normal(rng, input_shape)
    out_shape, (params, prev_state) = gru.init(rng, input_shape)
    outputs, hidden_state = gru.apply(params, x, prev_state=prev_state)
    print(outputs.shape)
    jax.jit(gru.apply)(params, x, prev_state=prev_state)
    jax.grad(lambda l: sum(sum(gru.apply(params, x, prev_state=prev_state)[0])))(1.0)


if __name__ == "__main__":
    test_lstm_cell()
    test_lstm()
    test_gru_cell()
    test_gru()