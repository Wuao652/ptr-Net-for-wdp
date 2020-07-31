import csv

import tensorflow as tf
import numpy as np
import pointer_net
import gzip
import pickle

# tf.app.flags.DEFINE_integer("batch_size", 32,"Batch size.")
# tf.app.flags.DEFINE_integer("num_item", 5, "The number of the total items.")
# tf.app.flags.DEFINE_integer("num_bid", 10, "The number of the total bids.")
# tf.app.flags.DEFINE_integer("unit_max", 5, "The max number of the units of each bid")
# tf.app.flags.DEFINE_integer("max_input_sequence_len", 20, "Maximum input sequence length.")
# tf.app.flags.DEFINE_integer("max_output_sequence_len", 21, "Maximum output sequence length.")
# tf.app.flags.DEFINE_integer("rnn_size", 128, "RNN unit size.")
# tf.app.flags.DEFINE_integer("attention_size", 128, "Attention size.")
# tf.app.flags.DEFINE_integer("num_layers", 1, "Number of layers.")
# tf.app.flags.DEFINE_integer("beam_width", 2, "Width of beam search .")
# tf.app.flags.DEFINE_float("learning_rate", 0.001, "Learning rate.")
# tf.app.flags.DEFINE_float("max_gradient_norm", 5.0, "Maximum gradient norm.")
# tf.app.flags.DEFINE_boolean("forward_only", False, "Forward Only.")
# tf.app.flags.DEFINE_string("log_dir", "./log", "Log directory")
# tf.app.flags.DEFINE_string("data_path", "./data/convex_hull_5_test.txt", "Data path.")
# tf.app.flags.DEFINE_integer("steps_per_checkpoint", 200, "frequence to do per checkpoint.")
# tf.app.flags.DEFINE_integer("train_epoch", 150, "The train epochs")
#
# FLAGS = tf.app.flags.FLAGS

class WdpDataGenerator(object):
    def __init__(self, FLAGS, typestr, data_size):
        self.typestr = typestr
        self.flag = FLAGS
        self.data_size = data_size

    def get_batch(self, is_batch):
        # data_size = self.inputs.shape[0]
        if is_batch:
            sample = np.random.choice(self.data_size, self.flag.batch_size, replace=True)
            self.read_data(sample)
            return self.inputs, self.enc_input_weights, \
                   self.outputs, self.dec_input_weights
        else:
            # sample = np.random.choice(self.data_size, self.data_size, replace=False)
            sample = np.arange(self.data_size)
            self.read_data(sample)
            return self.inputs, self.enc_input_weights, \
                   self.outputs, self.dec_input_weights
    def read_data(self, sample):
        inputs, outputs = [], []
        units = []
        enc_input_weights, dec_input_weights = [], []
        for index in sample:
            data_path = './data/sample/{}_{}/{}/{}/sample_{}.pkl'.format(self.flag.num_item,\
                                                                         self.flag.num_bid,
                                                                         self.flag.unit_max, self.typestr, index + 1)
            # print("...loading...data" + str(index + 1))
            with gzip.open(data_path, 'rb') as file:
                data_sample = pickle.load(file)
                instance_matrix = data_sample['instance_matrix']
                data = data_sample['bid_selection']
                unit_matrix = data_sample['unit_matrix']
                # print('-'*20)
                # print('DEBUG')
                # print('Instance_matrix:')
                # print(instance_matrix)
                # print('Unit_max:')
                # print(unit_matrix)
                # print('Solution:')
                # print(data)
                instance_matrix = np.pad(instance_matrix, ((0, self.flag.max_input_sequence_len - self.flag.num_bid),\
                                                          (0, 0)), 'constant')
                value = instance_matrix[:, -1].reshape(-1)
                input_len = sum([1 if value[t] > 0 else 0 for t in range(value.shape[0])])
                enc_weight = np.zeros(self.flag.max_input_sequence_len)
                enc_weight[:input_len] = 1
                inputs.append(instance_matrix)
                enc_input_weights.append(enc_weight)
                units.append(unit_matrix)

                data = np.where(data == 1)[0]
                output = [pointer_net.START_ID]
                for t in data:
                    output.append(t + 1 + 2)
                output.append(pointer_net.END_ID)
                dec_input_len = len(output) - 1
                output += [pointer_net.PAD_ID] * (self.flag.max_output_sequence_len - dec_input_len)
                output = np.array(output)
                weight = np.zeros(self.flag.max_output_sequence_len)
                weight[:dec_input_len] = 1
                dec_input_weights.append(weight)
                outputs.append(output)
                # print('input:')
                # print(instance_matrix)
                # print('Enc_input_weights:')
                # print(enc_weight)
                # print('output:')
                # print(output)
                # print("Dec_input_weights:")
                # print(weight)

        self.inputs = np.stack(inputs)
        self.outputs = np.stack(outputs)
        self.units = np.stack(units)
        self.enc_input_weights = np.stack(enc_input_weights)
        self.dec_input_weights = np.stack(dec_input_weights)
        # print("Load inputs:            " + str(self.inputs.shape))
        # print("Load enc_input_weights: " + str(self.enc_input_weights.shape))
        # print("Load outputs:           " + str(self.outputs.shape))
        # print("Load dec_input_weights: " + str(self.dec_input_weights.shape))
        # print("Load units:             " + str(self.units.shape))

def fromvector2index(vector):
    index = []
    for i in range(len(vector)):
        if vector[i] == 1:
            index.append(i+1)
    return index
def fromindex2vector(index,n):
    vector = [0]*n
    for i in index:
        vector[i-1] = 1
    return vector



def main():
    train_data = WdpDataGenerator(FLAGS,'train', 100)

    # print('SAMPLE!')
    # print(train_data.inputs[0])
    # print(train_data.enc_input_weights[0])
    # print(train_data.outputs[0])
    # print(train_data.dec_input_weights[0])

    # train_data.get_batch(True)
    train_data.get_batch(False)



if __name__ == "__main__":
    # tf.app.run()
    main()