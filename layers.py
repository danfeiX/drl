import tensorflow as tf

class FC_Layer:
    def __init__(self, input_dim, output_dim, input_x, ns, nonlinearity=tf.nn.relu):
        with tf.variable_scope(ns):
            init_f =  tf.random_uniform_initializer(
                -2.0 / (input_dim+output_dim),
                2.0 / (input_dim+output_dim))
            Ws = tf.get_variable('weights', [input_dim, output_dim],
                                 initializer=init_f)
            bs = tf.get_variable("biases", (output_dim,), initializer=tf.constant_initializer(0))
            self._op = nonlinearity(tf.matmul(input_x, Ws) + bs, name=ns)
            self._vars = [Ws, bs]

    @property
    def op(self):
        return self._op

    @property
    def vars(self):
        return self._vars
