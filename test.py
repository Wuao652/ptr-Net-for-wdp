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

FLAGS = tf.app.flags.FLAGS
def feasible_check(p, inp, u, p_len):
    u_pred = np.zeros(u.shape)
    # print('u_pred:')
    # print(u_pred.shape)
    # print('inp.shape')
    # print(inp.shape)

    for i in range(p_len):
        for j in range(inp.shape[1]-1):
            u_pred[j] = u_pred[j] + inp[p[i]-1][j]

    # print('umax:')
    # print(u)
    # print('u_pred')
    # print(u_pred)
    if all(u_pred <= u):
        return True
    else:
        return False
    pass
def gap_check(p, tar, inp):
    print('P_shape:')
    print(p.shape)
    print('Tar_shape:')
    print(tar.shape)
    p_revenue = sum([inp[p[i]-1][-1] if p[i]>0 else 0 for i in range(p.shape[0])])
    tar_revenue = sum([inp[tar[i] - 1][-1] if tar[i] > 0 else 0 for i in range(tar.shape[0])])
    print('p_revenue:')
    print(p_revenue)
    print('tar_revenue:')
    print(tar_revenue)
    return (tar_revenue - p_revenue)/tar_revenue
    pass
def correct_check(target, pred, tar_length, pred_length, i):
    t1 = target[i][:tar_length]
    t2 = pred[i][:pred_length]
    print(t1)
    print(t2)
    if len(t1) == len(t2) and all(t1 == t2):
        return True
    else:
        return False
    pass
def eval(pred, target, units, inputs):
    pred = pred[1].reshape(FLAGS.batch_size, -1)
    correct_count = 0
    feasible_count = 0
    current_max_gap = 0
    for i in range(target.shape[0]):
        tar_length = sum([1 if target[i][v] > 0 else 0 for v in range(target.shape[1])])
        pred_length = sum([1 if pred[i][v] > 0 else 0 for v in range(pred.shape[1])])
        if correct_check(target, pred, tar_length, pred_length, i):
            correct_count = correct_count + 1
            feasible_count = feasible_count + 1
        else:
            print("Print_not_correct", i)
            if feasible_check(pred[i], inputs[i], units[i], pred_length):
                feasible_count = feasible_count + 1
                current_max_gap = max(current_max_gap, gap_check(pred[i], target[i], inputs[i]))

    return correct_count, feasible_count, current_max_gap

    # return correct_count, feasible_count, max_gap
def main():
    test_data = DataGenerator(FLAGS, 'test', 10000)

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
        correct_count = 0
        feasible_count = 0
        largest_gap = 0
        count = 0
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
            # model prediction
            # print('*'*20)
            # print('Predicted_ids_with_logits:')
            # print(predicted_ids_with_logits[1].reshape(FLAGS.batch_size, -1))
            # print('Targets:')
            # print(targets[0])
            # print('Batch_units:')
            # print(batch_units.shape)
            temp_correct, temp_feasible, temp_gap = eval(predicted_ids_with_logits, targets, batch_units, batch_inputs)

            correct_count = correct_count + temp_correct
            feasible_count = feasible_count + temp_feasible
            largest_gap = max(largest_gap, temp_gap)
            count = count + 1
            print('='*20)
            print('round{}'.format(count))


        print('Acc: {:.2f}% ({}/{})'.format(correct_count / (count*FLAGS.batch_size)* 100, correct_count,
                                            count*FLAGS.batch_size))
        print('Feasible_rate: {:.2f}% ({}/{})'.format(feasible_count / (count * FLAGS.batch_size) * 100, feasible_count,
                                            count * FLAGS.batch_size))
        print('Largest_gap: {:.2f}'.format(largest_gap))

if __name__ == '__main__':
    main()