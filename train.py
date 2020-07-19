import tensorflow as tf
import numpy as np
import pointer_net
from datagenerator import DataGenerator
import time
import os
import win_unicode_console
win_unicode_console.enable()

tf.app.flags.DEFINE_integer("batch_size", 128,"Batch size.")
tf.app.flags.DEFINE_integer("num_item", 5, "The number of the total items.")
tf.app.flags.DEFINE_integer("unit_max", 5, "The max number of the units of each bid")
tf.app.flags.DEFINE_integer("max_input_sequence_len", 10, "Maximum input sequence length.")
tf.app.flags.DEFINE_integer("max_output_sequence_len", 11, "Maximum output sequence length.")
tf.app.flags.DEFINE_integer("rnn_size", 128, "RNN unit size.")
tf.app.flags.DEFINE_integer("attention_size", 128, "Attention size.")
tf.app.flags.DEFINE_integer("num_layers", 1, "Number of layers.")
tf.app.flags.DEFINE_integer("beam_width", 2, "Width of beam search .")
tf.app.flags.DEFINE_float("learning_rate", 0.001, "Learning rate.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0, "Maximum gradient norm.")
tf.app.flags.DEFINE_boolean("forward_only", False, "Forward Only.")
tf.app.flags.DEFINE_string("log_dir", "./log", "Log directory")
tf.app.flags.DEFINE_string("data_path", "./data/convex_hull_5_test.txt", "Data path.")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 200, "frequence to do per checkpoint.")
tf.app.flags.DEFINE_integer("train_epoch", 150, "The train epochs")

FLAGS = tf.app.flags.FLAGS


def eval(pred, target):
    pred = pred[1].reshape(FLAGS.batch_size, -1)
    correct_count = 0
    for i in range(target.shape[0]):
        tar_length = sum([1 if target[i][v] > 0 else 0 for v in range(target.shape[1])])
        t1 = target[i][:tar_length]
        t2 = pred[i][:tar_length]
        if all(t1 == t2):
            correct_count = correct_count + 1
    print('Acc: {:.2f}% ({}/{})'.format(correct_count / target.shape[0] * 100, correct_count,
                                        target.shape[0]))
def main():
    train_data = DataGenerator(FLAGS, 'train', 40000)
    valid_data = DataGenerator(FLAGS, 'valid', 5000)
    # test_data = DataGenerator(FLAGS, 'test')

    my_model = pointer_net.PointerNet(batch_size=FLAGS.batch_size,
                                            max_input_sequence_len=FLAGS.max_input_sequence_len,
                                            max_output_sequence_len=FLAGS.max_output_sequence_len,
                                            rnn_size=FLAGS.rnn_size,
                                            attention_size=FLAGS.attention_size,
                                            num_layers=FLAGS.num_layers,
                                            beam_width=FLAGS.beam_width,
                                            learning_rate=FLAGS.learning_rate,
                                            max_gradient_norm=FLAGS.max_gradient_norm,
                                            )

    with tf.Session() as sess:
        writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
        ckpt = tf.train.get_checkpoint_state(FLAGS.log_dir)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print("Load model parameters from %s" % ckpt.model_checkpoint_path)
            my_model.saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print("Created model with fresh parameters.")
            sess.run(tf.global_variables_initializer())
        step_time = 0.0
        loss = 0.0
        current_step = 0
        train_flag_var = False
        for _ in range(FLAGS.train_epoch*(train_data.data_size//FLAGS.batch_size)):
            start_time = time.time()
            inputs, enc_input_weights, outputs, dec_input_weights = \
                train_data.get_batch(True)
            summary, step_loss, predicted_ids_with_logits, targets, debug_var = \
                my_model.step(sess, inputs, enc_input_weights, outputs, dec_input_weights, train_flag_var)
            step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
            loss += step_loss / FLAGS.steps_per_checkpoint
            current_step += 1

            # Time to print statistic and save model
            if current_step % FLAGS.steps_per_checkpoint == 0:
                train_flag_var = True
                with sess.as_default():
                    gstep = my_model.global_step.eval()
                print("global step %d step-time %.2f loss %.2f" % (gstep, step_time, loss))
                ####
                inputs, enc_input_weights, outputs, dec_input_weights = \
                    valid_data.get_batch(True)
                predicted_ids_with_logits, targets = \
                    my_model.step(sess, inputs, enc_input_weights, outputs, dec_input_weights, train_flag_var)

                eval(predicted_ids_with_logits, targets)

                # Write summary
                writer.add_summary(summary, gstep)
                # Randomly choose one to check
                # sample = np.random.choice(FLAGS.batch_size, 1)[0]
                # print('=' *20)
                # print(predicted_ids_with_logits[1].reshape(FLAGS.batch_size, -1))
                # print(targets)
                # print("=" * 20)
                # print("Predict: " + str(np.array(predicted_ids_with_logits[1][sample]).reshape(-1)))
                # print("Target : " + str(targets[sample]))
                # print("=" * 20)
                checkpoint_path = os.path.join(FLAGS.log_dir, "wdp.ckpt")
                my_model.saver.save(sess, checkpoint_path, global_step=my_model.global_step)
                step_time, loss = 0.0, 0.0
                train_flag_var = False

if __name__ == '__main__':
    main()