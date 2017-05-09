import tensorflow as tf
import numpy as np
from tensorflow.contrib.rnn import LSTMCell, MultiRNNCell, ResidualWrapper, DeviceWrapper
from tensorflow.contrib.seq2seq.python.ops import helper as helper_py
from tensorflow.contrib.seq2seq.python.ops import basic_decoder
from attention_wrapper import AttentionWrapper

def whileloop_fib_exmpl(n):
    '''
    :param n: steps of fibonacci
    '''
    init_state = (0, (1, 1))
    condition = lambda i, _: i < n
    body = lambda i, fib: (i + 1, (fib[1], fib[0] + fib[1]))

    tf.reset_default_graph()

    # run the graph
    with tf.Session() as sess:
        (n, fib_n) = sess.run(tf.while_loop(condition, body, init_state))
        print(n)
        print(fib_n)


def tensorArray_exmpl(n):
    '''
    for accession to slices of tensor

    this is building block for tf.nn.dynamic_rnn,  tf.scan, etc..
    :return:
    '''
    # 1000 sequence in the length of 100
    matrix = tf.placeholder(tf.int32, shape=(100, 1000), name="input_matrix")
    matrix_rows = tf.shape(matrix)[0]
    ta = tf.TensorArray(tf.int32, size=matrix_rows)

    init_state = (0, ta)
    condition = lambda i, _: i < matrix_rows
    body = lambda i, ta: (i + 1, ta.write(i, matrix[i] * 2))
    m, ta_final = tf.while_loop(condition, body, init_state)
    # get the final result
    ta_final_result = ta_final.stack()

    # run the graph
    with tf.Session() as sess:
        # print the output of ta_final_result
        print (sess.run(ta_final_result, feed_dict={matrix: np.ones(dtype=np.int32, shape=(100,1000))}))



def simple_classification():
    batch_size = 100

    x = tf.placeholder(tf.float32, (None, 784))

    b = tf.Variable(tf.zeros((batch_size,)))
    W = tf.Variable(tf.random_uniform((784, batch_size), -1, 1))

    h = tf.nn.relu(tf.matmul(x, W) + b)

    prediction = tf.nn.softmax(h)

    label = tf.placeholder(tf.float32, [batch_size, 10])

    cross_entropy = -tf.reduc_sum(label * tf.log(prediction), axis=1)

    # 0.5  is learning rate
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    tf.reset_default_graph()
    sess = tf.Session()
    sess.run(tf.initialize_all_variable())

    for i in range(1000):
        batch_x, batch_label = data.next_batch()
        sess.run(train_step, feed_dict={x: batch_x, label: batch_label})



def simple_regression():
    def linear_regression():
        # None bc of variable length batch_size
        x = tf.placeholder(tf.float32, shape=(None,), name="x")
        y = tf.placeholder(tf.float32, shape=(None,), name="y")

        with tf.variable_scope('lreg') as scope:
            w = tf.Variable(np.random.normal(), name="W")
            y_pred = tf.multiply(w, x)
            cost = tf.reduce_mean(tf.square(y_pred - y))
        return x, y, y_pred, cost

    def run():
        x_batch = np.linspace(-1, 1, 101)
        y_batch = x_batch * 2 + np.random.randn(*x_batch.shape) * 0.3
        x, y, y_pred, cost = linear_regression()
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for t in range(30):
                cost_t, _ = sess.run([cost, optimizer], {x: x_batch, y: y_batch})
                print(cost_t.mean())

            # to see what are the outputs
            sess.run(y_pred, feed_dict={x: x_batch})

    tf.reset_default_graph()
    run()


def word2vec():
    return 1
    # def skipgram():
    #     batch_input=tf.placeholder(tf.int32, shape=[batch_size,])
    #     batch_label=tf.placeholder(tf.int32, shape=[batch_size,1])
    #     val_dataset=tf.constant(val_data, dtype=tf.int32)
    #
    #     with tf.variable_scope('word2vec') as scope:
    #         embedding = tf.Variable (tf.random_uniform([vocabulary_size, embedding_size],-1,1))
    #         batch_embeddings=tf.nn.embedding_lookup(embedding, batch_input)
    #         weight=tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size],stddev=1/math.sqrt(embedding_size)))
    #         biases=tf.Variable(tf.zeros([vocabulary_size]))
    #         # negative sampling
    #         loss= tf.reduce_mean(tf.nn.nce_loss(weights=weight, biases=biases, labels=batch_label, inputs=batch_input, num_sampled=num_n_samples, num_classes=vocabulary_size))
    #
    #     norm = tf.sqrt(tf.reduce_mean(tf.square(embedding),1,keep_dims=True))
    #     normalized_embeddings=embedding/norm
    #     val_embeddings=tf.embedding_lookup(normalized_embeddings, val_dataset)
    #     similarity=tf.multiply(val_embeddings, normalized_embeddings, transpose_b=Tue)
    #     return batch_input, batch_label, normalized_embeddings, loss, similarity
    #
    # def run():
    #     batch_input, batch_label, normalized_embeddings, loss, similarity=skipgram()
    #     optimizer=tf.train.GradientDescentOptimizer(0.1).minimize(loss)
    #     with tf.Session() as sess:
    #         sess.run(tf.global_variables_initializer())
    #         average_loss=0.0
    #         for step, batch_data in enumerate(train_data):
    #             inputs, labels=batch_data
    #             feed_dict={batch_input:inputs, batch_label:labels}
    #             _,loss_val=sess.run([optimizer,loss],feed_dict)
    #             average_loss+=loss_val
    #
    #             if step %1000==0:
    #                 if step>0:
    #                     average_loss/=1000
    #                     print ('loss at iter', step,':',average_loss)
    #                 average_loss=0
    #
    #             if step % 5000==0:
    #                 sim=similarity.eval()
    #                 for i in range(len(val_data)):
    #                     top_k=8
    #                     nearest=(-sim[i,:]).argsort()[1:top_k+1]
    #                     print_closest_words(val_data[i],nearest)
    #         final_embeddings=normalized_embeddings.eval()

def define_seq2seq_model():
    T=1000
    N=100
    input = tf.placeholder(tf.float32, shape=(N, T, 512), name="input_matrix")
    seq_lengths = tf.placeholder(tf.int32, shape=(N), name="input_lengths")

    cell= MultiRNNCell([DeviceWrapper(ResidualWrapper(LSTMCell(num_units=512)),device='/gpu:%d' %(i+1)) for i in range(2)])
    encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(cell,input, parallel_iterations=32, swap_memory=True, dtype=tf.float32)

    # Attention Mechanisms. Bahdanau is additive style attention
    attn_mech = tf.contrib.seq2seq.BahdanauAttention(
        num_units = 100, # depth of query mechanism
        memory = encoder_outputs, # hidden states to attend (output of RNN)
        #memory_sequence_length= T,#tf.sequence_mask(seq_lengths, T), # masks false memories
        normalize=False, # normalize energy term
        name='BahdanauAttention')

    cell_out= MultiRNNCell([DeviceWrapper(ResidualWrapper(LSTMCell(num_units=512)),device='/gpu:%d' %(i+1)) for i in range(2)])
        # Attention Wrapper: adds the attention mechanism to the cell

    # Attention Wrapper: adds the attention mechanism to the cell
    attn_cell = tf.contrib.seq2seq.AttentionWrapper(
        cell = cell,# Instance of RNNCell
        attention_mechanism = attn_mech, # Instance of AttentionMechanism
        attention_size = 100, # Int, depth of attention (output) tensor
        attention_history=False, # whether to store history in final output
        name="attention_wrapper")

    # TrainingHelper does no sampling, only uses inputs
    helper = tf.contrib.seq2seq.TrainingHelper(
        inputs = x, # decoder inputs
        sequence_length = seq_len_dec, # decoder input length
        name = "decoder_training_helper")

    # Decoder setup
    decoder = tf.contrib.seq2seq.BasicDecoder(
              cell = attn_cell,
              helper = helper, # A Helper instance
              initial_state = encoder_final_state, # initial state of decoder
              output_layer = None) # instance of tf.layers.Layer, like Dense

    # Perform dynamic decoding with decoder object
    outputs, final_state = tf.contrib.seq2seq.dynamic_decode(decoder)

    #helper = TrainingHelper(decoder_inputs, sequence_length)
    #
    #
    # decoder=tf.contrib.seq2seq.BasicDecoder(
    #     cell=cell_out,
    #     helper=helper,
    #     initial_state=encoder_final_state,
    #     attention='BahdanauAttention')
    #
    # decoder_outputs, final_decoder_state = dynamic_decode(decoder)
    #
    # decoder_logits = decoder_outputs.rnn_output
    # decoder_sample_ids= decoder_outputs.sample_id
    #
    #

    #

    #
    # # TrainingHelper does no sampling, only uses inputs
    # helper = tf.contrib.seq2seq.TrainingHelper(
    #     inputs = x, # decoder inputs
    #     sequence_length = seq_len_dec, # decoder input length
    #     name = "decoder_training_helper")
    #
    # # Decoder setup
    # decoder = tf.contrib.seq2seq.BasicDecoder(
    #           cell = attn_cell,
    #           helper = helper, # A Helper instance
    #           initial_state = encoder_state, # initial state of decoder
    #           output_layer = None) # instance of tf.layers.Layer, like Dense
    #
    # Perform dynamic decoding with decoder object
    #outputs, final_state = tf.contrib.seq2seq.dynamic_decode(decoder)

def primitive():
    with tf.variable_scope('foo', reuse=True):
        v = tf.get_variable('v')


if __name__ == '__main__':
    tensorArray_exmpl(6)
