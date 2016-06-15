from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import time

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import sys
from mln import MLN
import dqn
from gym_input import GymSim

from IPython import embed

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', 'output/gym_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 1000000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('batch_size', 16,
                            """Number of observations to process in a batch.""")
tf.app.flags.DEFINE_integer('sample_neg_ratio', 1,
                            """neg/pos sample ratio""")
tf.app.flags.DEFINE_integer('sample_after', 1000,
                            """sample after # iteration""")
tf.app.flags.DEFINE_integer('learning_rate', 1e-6,
                            """learning rate""")


def train():
    rnd_seed = 0
    np.random.seed(rnd_seed)
    tf.set_random_seed(rnd_seed)
    sim = GymSim('CartPole-v0', 5000, seed=rnd_seed)
    sim.act_sample_batch(5000, FLAGS.sample_neg_ratio) # bootstrap with random actions
    sim.print_stats()
    #embed()
    #sys.exit()
    q_network = MLN(sim.INPUT_DIM, sim.ACTION_DIM)
    target_network = MLN(sim.INPUT_DIM, sim.ACTION_DIM, name_scope='target')

    with tf.Graph().as_default():

        global_step = tf.Variable(0, trainable=False)

        action_pl = tf.placeholder(tf.int64, name='action_pl')
        reward_pl = tf.placeholder(tf.float32, name='reward_pl')
        state_pl  = tf.placeholder(tf.float32, (None, sim.INPUT_DIM), name='state_pl')
        observ_pl = tf.placeholder(tf.float32, (None, sim.INPUT_DIM), name='observ_pl')

        action_q = q_network.inference(state_pl)
        target_q = tf.stop_gradient(target_network.inference(observ_pl))
        target_q_pt = tf.Print(target_q, [target_q])
        action_q_pt = tf.Print(action_q, [action_q])

        loss = dqn.td_loss(action_pl, sim.ACTION_DIM, action_q, reward_pl, target_q)

        train_op = dqn.train(FLAGS.learning_rate, loss, global_step)

        saver = tf.train.Saver(tf.all_variables())

        summary_op = tf.merge_all_summaries()

        action_op = tf.argmax(action_q, 1, name='action_op')

        copy_var = q_network.copy_to('target')

        init = tf.initialize_all_variables()

        sess = tf.Session(config=tf.ConfigProto(
            log_device_placement=FLAGS.log_device_placement))

        #initialize variables
        sess.run(init)

        summary_writer = tf.train.SummaryWriter(os.path.join(FLAGS.train_dir, 'logs')
                                                , sess.graph)

    for step in xrange(FLAGS.max_steps):
        start_time = time.time()

        if step % 4 == 0:
            sess.run(copy_var)

        feed = sim.feed_batch(state_pl,
                              action_pl,
                              reward_pl,
                              observ_pl,
                              FLAGS.batch_size)

        _, loss_value = sess.run([train_op, loss],
                                 feed_dict = feed)

        duration = time.time() - start_time

        assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

        if step % 10 == 0:
            num_examples_per_step = FLAGS.batch_size
            examples_per_sec = num_examples_per_step / duration
            sec_per_batch = float(duration)

            format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                        'sec/batch)')
            print (format_str % (datetime.now(), step, loss_value,
                                examples_per_sec, sec_per_batch))

        if step > FLAGS.sample_after:
            pred_act = sess.run(action_op,
                                feed_dict={state_pl: sim.state})
            pred_act = pred_act[0]
            sim.act_sample_once(pred_act, neg_ratio=FLAGS.sample_neg_ratio,
                                append_db=True)


        # visualization
        if step % 1000 == 0 and step != 0:
            sim.reset()
            survive = 0
            for _ in range(200):
                pred_act = sess.run(action_op,
                                    feed_dict={state_pl: sim.state})
                pred_act = pred_act[0]
                done = sim.act_demo(pred_act)
                if not done:
                    survive += 1
                else:
                    print('Survived for %i frame' % survive)
                    survive = 0

        #if step % 100 == 0:
        #    summary_str = sess.run(summary_op)
        #    summary_writer.add_summary(summary_str, step)

        # Save the model checkpoint periodically.
        if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
            checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
            saver.save(sess, checkpoint_path, global_step=step)


def main(argv=None):  # pylint: disable=unused-argument
    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)
    train()


if __name__ == '__main__':
  tf.app.run()
