import csv

import tensorflow as tf
import numpy as np
import pointer_net


# tf.app.flags.DEFINE_integer("batch_size", 128,"Batch size.")
# tf.app.flags.DEFINE_integer("num_item", 5, "The number of the total items.")
# tf.app.flags.DEFINE_integer("unit_max", 5, "The max number of the units of each bid")
# tf.app.flags.DEFINE_integer("max_input_sequence_len", 10, "Maximum input sequence length.")
# tf.app.flags.DEFINE_integer("max_output_sequence_len", 11, "Maximum output sequence length.")
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

class DataGenerator(object):
    def __init__(self, FLAGS, typestr, data_size):
        self.typestr = typestr
        self.flag = FLAGS
        self.data_size = data_size
        self.read_data()

    def takeIdx(self, reader):
        return int(reader[-1])

    def read_data(self):
        print("Start loading input data!")
        inputs = []
        enc_input_weights = []
        for i in range(self.data_size):
            data_path = './data/instance/' + str(self.flag.num_item) + '_' + str(self.flag.max_input_sequence_len) + '/' + \
                        str(self.flag.unit_max) + '/' + self.typestr + '/binary/instance_' + str(i + 1) + '.txt'
            print("...loading...data" + str(i + 1))
            with open(data_path, 'r') as file:
                recs = file.readlines()
                enc_input = []
                for rec in recs:
                    inp = rec[:-2].split(' ')
                    for t in inp:
                        enc_input.append(float(t))
                enc_input = np.array(enc_input).reshape([-1, (self.flag.num_item + 1)])
                inputs.append(enc_input)
                weight = np.ones(self.flag.max_input_sequence_len)
                enc_input_weights.append(weight)
        self.inputs = np.stack(inputs)
        self.enc_input_weights = np.stack(enc_input_weights)
        print("Load input data finished!")

        print("Start loading output data!")
        self.outputs = []
        self.dec_input_weights = []
        data_path = './results/' + 'eval_' + str(self.flag.num_item) + '_' + str(self.flag.max_input_sequence_len) \
                    + '/cplex_' + self.typestr + '_output.csv'

        with open(data_path, 'r') as csvfile:
            reader = csv.reader(csvfile)
            reader = list(reader)
            reader = reader[1:]
            # print(reader)
            reader.sort(key=self.takeIdx)
            for i in reader:
                strseq = i[2]
                strseq = strseq[1:-1].split()

                readoutput = []
                for t in strseq:
                    readoutput.append(int(t))
                readidx = fromvector2index(readoutput)
                # print(readidx)
                output = [pointer_net.START_ID]
                for t in readidx:
                    # Add 2 to value due to the sepcial tokens
                    output.append(t + 2)
                output.append(pointer_net.END_ID)
                dec_input_len = len(output) - 1

                output += [pointer_net.PAD_ID] * (self.flag.max_output_sequence_len-dec_input_len)
                output = np.array(output)
                self.outputs.append(output)
                weight = np.zeros(self.flag.max_output_sequence_len)
                weight[:dec_input_len] = 1
                self.dec_input_weights.append(weight)

        self.outputs = np.array(self.outputs)
        self.dec_input_weights = np.array(self.dec_input_weights)
        print("Load inputs:            " + str(self.inputs.shape))
        print("Load enc_input_weights: " + str(self.enc_input_weights.shape))
        print("Load outputs:            " + str(self.outputs.shape))
        print("Load dec_input_weights: " + str(self.dec_input_weights.shape))



    def get_batch(self, is_batch):
        # data_size = self.inputs.shape[0]
        if is_batch:
            sample = np.random.choice(self.data_size, self.flag.batch_size, replace=True)
            return self.inputs[sample], self.enc_input_weights[sample], \
                   self.outputs[sample], self.dec_input_weights[sample]
        else:
            # sample = np.random.choice(self.data_size, self.data_size, replace=False)
            return self.inputs, self.enc_input_weights, \
                   self.outputs, self.dec_input_weights
        # for i in sample :
        #     print('data{}'.format(i))
        #     print(self.inputs[i])
        #     print(self.enc_input_weights[i])
        #     print(self.outputs[i])
        #     print(self.dec_input_weights[i])
        # return self.inputs[sample], self.enc_input_weights[sample], \
        #        self.outputs[sample], self.dec_input_weights[sample]
    def read_unit(self):
        print("Start loading units information!")
        units = []
        for i in range(self.data_size):
            data_path = './data/instance/' + str(self.flag.num_item) + '_' + str(
                self.flag.max_input_sequence_len) + '/' + \
                        str(self.flag.unit_max) + '/' + self.typestr + '/unit/instance_' + str(i + 1) + '.txt'
            print("...loading...data" + str(i + 1))
            with open(data_path, 'r') as file:
                recs = file.readlines()
                enc_unit = []
                for rec in recs:
                    u = rec[:-2].split(' ')
                    for t in u:
                        enc_unit.append(float(t))
                # enc_unit = np.array(enc_unit).reshape([-1, (self.flag.num_item)])
                enc_unit = np.array(enc_unit)
                units.append(enc_unit)
        self.units = np.stack(units)
        print("Load units information finished!")
        print("Load units:            " + str(self.units.shape))

    def units_batch(self, is_batch):
        if is_batch:
            sample = np.random.choice(self.data_size, self.flag.batch_size, replace=True)
            return self.units
        else:
            return self.units
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
    train_data = DataGenerator(FLAGS,'test', 10000)

    print('SAMPLE!')
    print(train_data.inputs[0])
    print(train_data.enc_input_weights[0])
    print(train_data.outputs[0])
    print(train_data.dec_input_weights[0])

    train_data.read_unit()
    units = train_data.units_batch(False)
    print(units)


if __name__ == "__main__":
    # tf.app.run()
    main()