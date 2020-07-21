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

tf.app.flags.DEFINE_integer("steps_per_checkpoint", 312, "frequence to do per checkpoint.")
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
    return correct_count
def eval_valid(data, model, sess, var, trainsummary):
    inputs, enc_input_weights, outputs, dec_input_weights = \
        data.get_batch(False)
    count, correct_count = 0, 0
    while inputs.shape[0] >= FLAGS.batch_size:

        batch_inputs = inputs[:FLAGS.batch_size]
        batch_enc_input_weights = enc_input_weights[:FLAGS.batch_size]
        batch_outputs = outputs[:FLAGS.batch_size]
        batch_dec_input_weights = dec_input_weights[:FLAGS.batch_size]
        inputs = inputs[FLAGS.batch_size:]
        enc_input_weights = enc_input_weights[FLAGS.batch_size:]
        outputs = outputs[FLAGS.batch_size:]
        dec_input_weights = dec_input_weights[FLAGS.batch_size:]

        predicted_ids_with_logits, targets = \
            model.step(sess, batch_inputs, batch_enc_input_weights, batch_outputs, batch_dec_input_weights, var)
        temp = eval(predicted_ids_with_logits, targets)
        correct_count = correct_count + temp
        count = count + 1

    print('Acc: {:.2f}% ({}/{})'.format(correct_count / (count * FLAGS.batch_size) * 100, correct_count,
                                        count * FLAGS.batch_size))
    trainsummary.write('Acc: {:.2f}% ({}/{}) \n'.format(correct_count / (count * FLAGS.batch_size) * 100, correct_count,
                                        count * FLAGS.batch_size))
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
    trainsummary = open('trainingsummary.txt', 'w+')
    with tf.Session() as sess:
        writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
        ckpt = tf.train.get_checkpoint_state(FLAGS.log_dir)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print("Load model parameters from %s" % ckpt.model_checkpoint_path)
            my_model.saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print("Created model with fresh parameters.")
            sess.run(tf.global_variables_initializer())
        print('start!!!!!!!!!!!!!!!!!')
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
                trainsummary.write('Epoch %d \n' % (current_step/FLAGS.steps_per_checkpoint))
                trainsummary.write("global step %d step-time %.2f loss %.2f \n" % (gstep, step_time, loss))
                ####
                eval_valid(valid_data, my_model, sess, train_flag_var, trainsummary)

                writer.add_summary(summary, gstep)
                checkpoint_path = os.path.join(FLAGS.log_dir, "wdp.ckpt")
                my_model.saver.save(sess, checkpoint_path, global_step=my_model.global_step)
                step_time, loss = 0.0, 0.0
                train_flag_var = False

    trainsummary.close()

if __name__ == '__main__':
    main()