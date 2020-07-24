import tensorflow as tf
import numpy as np
import pointer_net
from datagenerator import DataGenerator
# from UnitGenerator import UnitGenerator
import time
import os
import win_unicode_console
win_unicode_console.enable()

tf.app.flags.DEFINE_integer("batch_size", 128, "Batch size.")
tf.app.flags.DEFINE_integer("num_item", 10, "The number of the total items.")
tf.app.flags.DEFINE_integer("unit_max", 5, "The max number of the units of each bid")
tf.app.flags.DEFINE_integer("max_input_sequence_len", 100, "Maximum input sequence length.")
tf.app.flags.DEFINE_integer("max_output_sequence_len", 101, "Maximum output sequence length.")
tf.app.flags.DEFINE_integer("rnn_size", 128, "RNN unit size.")
tf.app.flags.DEFINE_integer("attention_size", 128, "Attention size.")
tf.app.flags.DEFINE_integer("num_layers", 1, "Number of layers.")
tf.app.flags.DEFINE_integer("beam_width", 2, "Width of beam search .")
tf.app.flags.DEFINE_float("learning_rate", 0.001, "Learning rate.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0, "Maximum gradient norm.")
tf.app.flags.DEFINE_boolean("forward_only", False, "Forward Only.")
tf.app.flags.DEFINE_string("log_dir", "./log", "Log directory")
#tf.app.flags.DEFINE_string("data_path", "./data/convex_hull_5_test.txt", "Data path.")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 468, "frequence to do per checkpoint.")

FLAGS = tf.app.flags.FLAGS
def accuracy_check(pred, tar):
    total_sample =  pred.shape[0]
    accurate_sample = sum([1 if all(pred[i] == tar[i]) else 0 for i in range(pred.shape[0]) ])
    print('Acc: {:.2f}% ({}/{})'.format(accurate_sample / total_sample * 100, accurate_sample,
                                        total_sample))
def feasible_check(unit_,u):
    total_sample = u.shape[0]
    feasible_sample = sum([1 if all(u[i] >= unit_[i]) else 0 for i in range(u.shape[0])])
    print('feasible_rate: {:.2f}% ({}/{})'.format(feasible_sample / total_sample * 100, feasible_sample,
                                        total_sample))
def gap_check(current_revenue, target_revenue):
    diff_revenue = (target_revenue - current_revenue) / target_revenue
    diff_revenue = [i if i >= 0 else 0 for i in diff_revenue]
    print('The average gap is {:.4f}'.format(np.mean(diff_revenue)))

def eval(test_data, my_model):
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(FLAGS.log_dir)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print("Load model parameters from %s" % ckpt.model_checkpoint_path)
            my_model.saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            return
        inputs, enc_input_weights, outputs, dec_input_weights = \
            test_data.get_batch(False)
        test_data.read_unit()
        units = test_data.units_batch(False)

        # read the input, output and the units from the database
        pred, tar = [], []
        while inputs.shape[0] >= FLAGS.batch_size:
            batch_inputs = inputs[:FLAGS.batch_size]
            batch_enc_input_weights = enc_input_weights[:FLAGS.batch_size]
            batch_outputs = outputs[:FLAGS.batch_size]
            batch_dec_input_weights = dec_input_weights[:FLAGS.batch_size]
            batch_units = units[:FLAGS.batch_size]
            # get a batch of data which is used to evaluate

            inputs = inputs[FLAGS.batch_size:]
            enc_input_weights = enc_input_weights[FLAGS.batch_size:]
            outputs = outputs[FLAGS.batch_size:]
            dec_input_weights = dec_input_weights[FLAGS.batch_size:]
            units = units[FLAGS.batch_size:]
            # delete that batch from the read_in data

            predicted_ids_with_logits, targets = \
                my_model.step(sess, batch_inputs, batch_enc_input_weights, batch_outputs, batch_dec_input_weights, True)
            predicted = predicted_ids_with_logits[1].reshape(FLAGS.batch_size, -1)
            predicted = np.pad(predicted,((0,0),(0,FLAGS.max_output_sequence_len-predicted.shape[1])), 'constant')
            pred.append(predicted)
            tar.append(targets)
            print('...loading...')
        pred = np.vstack(pred)
        tar = np.vstack(tar)
        return pred, tar
def main():
    test_data = DataGenerator(FLAGS, 'test', 2000, 'single')

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

    pred, tar = eval(test_data, my_model)
    # use the model to predict
    inp = test_data.inputs[:tar.shape[0]]
    u = test_data.units[:tar.shape[0]]
    print('pred:', pred.shape)
    print('tar:', tar.shape)
    print('inp', inp.shape)
    print('u', u.shape)

    # current_u = np.zeros(u.shape[1])
    # current_r = 0
    unit_ = []
    current_revenue = np.zeros(tar.shape[0])
    target_revenue = np.zeros(tar.shape[0])
    for i in range(tar.shape[0]):
        bid_pred = pred[i]
        bid_tar = tar[i]
        current_u = np.zeros(u.shape[1])
        current_r = 0
        target_r = 0
        for k in range(tar.shape[1]):
            if not bid_tar[k] == 0:
                target_r = target_r + inp[i][bid_tar[k]-1][-1]
            else:
                target_revenue[i] = target_r
                break
        for j in range(tar.shape[1]):
            if bid_pred[j] == 0:
                current_revenue[i] = current_r
                unit_.append(current_u)
                break
            current_u = current_u + inp[i][bid_pred[j]-1][:-1]
            current_r = current_r + inp[i][bid_pred[j]-1][-1]
            if any(current_u > u[i]):
                current_u = current_u - inp[i][bid_pred[j]-1][:-1]
                current_r = current_r - inp[i][bid_pred[j] - 1][-1]
                unit_.append(current_u)
                current_revenue[i] = current_r
                pred[i][j:] = 0
                break
    unit_ = np.stack(unit_)

    #
    accuracy_check(pred, tar)
    feasible_check(unit_, u)
    gap_check(current_revenue, target_revenue)

if __name__ == '__main__':
    main()