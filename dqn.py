import tensorflow as tf
# Global constants describing the CIFAR-10 data set.
# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 10000.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.0001  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.01       # Initial learning rate.

def td_loss(action, action_dim, current_y, reward, target_y, discount=0.99):
    """Simple off-policy TD error loss
    """
    action_mask = tf.one_hot(action, depth=action_dim,
                            on_value=1, off_value=0)
    action_mask = tf.cast(action_mask, tf.float32)
    current_q = tf.reduce_sum(tf.mul(action_mask, current_y), 1)
    target_q = tf.reduce_max(target_y, 1)*discount + reward
    td_error = tf.reduce_mean(tf.square(current_q - target_q))

    #tf.add_to_collection('losses', td_error)
    #return tf.add_n(tf.get_collection('losses'), name='total_loss')
    return td_error

def train(lr, total_loss, global_step):
    # Variables that affect learning rate.

    # Compute gradients.
    #with tf.control_dependencies([loss_averages_op]):
    opt = tf.train.GradientDescentOptimizer(lr)
    grads = opt.compute_gradients(total_loss)

    # Add histograms for gradients.
    for i, (grad, var) in enumerate(grads):
        if grad is not None:
            tf.histogram_summary(var.op.name + '/gradients', grad)
            grads[i] = (tf.clip_by_norm(grad, 5), var)

    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
        tf.histogram_summary(var.op.name, var)

    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')

    return train_op
