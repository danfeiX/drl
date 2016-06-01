import tensorflow as tf
from layers import FC_Layer

class MLN:
    def __init__(self, input_dim, output_dim, name_scope=''):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fc_dims = [10, 10, 10]
        self.vars = []
        self.name_scope = name_scope

    def inference(self, input_x):
        with tf.variable_scope(self.name_scope):
            fc1 = FC_Layer(self.input_dim, self.fc_dims[0], input_x, "fc1", tf.identity)
            fc2 = FC_Layer(fc1.output_dim, self.fc_dims[1], fc1.op, "fc2", tf.identity)
            fc_out = FC_Layer(fc2.output_dim, self.output_dim, fc2.op, "fc_out", tf.identity)

        self.vars.extend(fc1.vars)
        self.vars.extend(fc2.vars)
        self.vars.extend(fc_out.vars)
        return fc_out.op

    def copy_to(self, target_ns):
        op = []
        with tf.variable_scope(target_ns, reuse=True):
            for v in self.vars:
                ov = tf.get_variable(v.name.split(':')[0], v.get_shape())
                op.append(ov.assign(v))
        return op
