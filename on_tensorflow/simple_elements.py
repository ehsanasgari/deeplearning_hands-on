import tensorflow as tf


def whileloop_fib_exp(n):
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

if __name__ == '__main__':
    whileloop_fib_exp(5)
