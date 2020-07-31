import tensorflow as tf
import numpy as np
import pointer_net
# from datagenerator import DataGenerator
# from datagenerator import fromindex2vector
# from datagenerator import fromvector2index
from wdpdatagenerator import WdpDataGenerator
from wdpdatagenerator import fromvector2index
from wdpdatagenerator import fromindex2vector
# from UnitGenerator import UnitGenerator
import time
import os
import random
import win_unicode_console
win_unicode_console.enable()

tf.app.flags.DEFINE_integer("batch_size", 128, "Batch size.")
tf.app.flags.DEFINE_integer("num_item", 5, "The number of the total items.")
tf.app.flags.DEFINE_integer("num_bid", 20, "The number of the total bids.")
tf.app.flags.DEFINE_integer("unit_max", 5, "The max number of the units of each bid")
tf.app.flags.DEFINE_integer("max_input_sequence_len", 20, "Maximum input sequence length.")
tf.app.flags.DEFINE_integer("max_output_sequence_len", 21, "Maximum output sequence length.")
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
    
    diff_revenue = [(target_revenue[i] - current_revenue[i]) / target_revenue[i] if (not target_revenue[i] == 0) and (target_revenue[i] - current_revenue[i]) / target_revenue[i]>0\
                                                                                else 0 for i in range(target_revenue.shape[0]) ]
    # diff_revenue = [i if i >= 0 else 0 for i in diff_revenue]
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
        units = test_data.units
        start = time.perf_counter()
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
            targets = np.array(targets)
            if targets.shape[1] > FLAGS.num_bid:
                targets = targets[:,:FLAGS.num_bid]
            else:
                targets = np.pad(targets,((0,0),(0,FLAGS.num_bid - targets.shape[1])), 'constant')
            if predicted.shape[1] > FLAGS.num_bid:
                predicted = predicted[:,:FLAGS.num_bid]
            else:
                predicted = np.pad(predicted,((0,0),(0,FLAGS.num_bid - predicted.shape[1])), 'constant')
            # predicted = np.pad(predicted,((0,0),(0,FLAGS.max_output_sequence_len-predicted.shape[1])), 'constant')
            pred.append(predicted)
            tar.append(targets)
            print('...loading...')
            # print(predicted.shape)
            # print(targets.shape)
        predtime = time.perf_counter() - start
        pred = np.vstack(pred)
        tar = np.vstack(tar)
        return pred, tar, predtime
def local_search(instance, unit, result_opt, wp = 0.85):
    bid_price = instance[:, -1].reshape(-1)
    revenue_opt = sum(result_opt * bid_price)
    result_tmp = result_opt.copy()
    max_iter = 50000
    iter_unchange = 0
    for i in range(max_iter):
        if iter_unchange > 5000:
            break
        # assume that the opt is steable
        bid_empty = np.where(result_tmp == 0)[0]
        if list(bid_empty) == [] or list(bid_price[bid_empty]) == []:
            break
        # print(list(bid_empty))
        # generate a random key and compare it with wp
        if np.random.random_sample(1)-1 < wp:
            # pick a random bid
            pick_index = random.sample(list(bid_empty), 1)[0]
            result_tmp[pick_index] = 1
        else:
            # pick the bid with the highest price greedily
            price_tmp = bid_price[bid_empty]
            pick_index = bid_empty[np.argmax(price_tmp)]
            result_tmp[pick_index] = 1
        # how to deal with the conflict
        while (np.dot(result_tmp.reshape(1,-1), instance[:,:-1])>unit).any():
            conflict_item = np.where(np.dot(result_tmp.reshape(1,-1),instance[:,:-1])>unit)[1]
            bid_choose = np.where(result_tmp == 1)[0]
            bundle_choose = instance[bid_choose, :-1]
            conflict_bid = np.array([])
            for item_index in list(conflict_item):
                # 对每个conlict的item来说
                conflict_bid = np.append(conflict_bid, np.where(bundle_choose[:, item_index] != 0)[0])
            conflict_bid = np.unique(conflict_bid).astype(int)
            conflict_bid_index = bid_choose[conflict_bid]
            #remove the one with the smallest price
            price_tmp = bid_price[conflict_bid_index]
            result_tmp[conflict_bid_index[np.argmin(price_tmp)]]=0

        # decide whether update the optimal result
        revenue_tmp = sum(result_tmp * bid_price)
        if revenue_tmp > revenue_opt:
            revenue_opt = revenue_tmp
            result_opt = result_tmp.copy()
            iter_unchange = 0
        else:
            iter_unchange = iter_unchange + 1
    result_opt = fromvector2index(result_opt)
    result_len = len(result_opt)
    result_opt = result_opt + [0]*(FLAGS.num_bid - result_len)
    return result_opt, revenue_opt
    pass
# def Casanova(instance, unit, result_opt, maxtries=3, wp=0.8, novelp=0.5):
#     maxsteps = 50000

#     bid_price = instance[:, -1]
#     bid_size = np.sum(instance[:, 0:-1], axis=1)
#     bid_density = bid_price / bid_size
#     revenue_global = 0
#     start = time.perf_counter()
#     for i in range(maxtries):
#         print("iter:{}".format(i+1))
#         revenue_opt = 0
#         # start with an empty allocation
#         result_tmp = np.zeros(FLAGS.max_input_sequence_len, dtype=int)
#         # age reset to 0
#         age = np.zeros(FLAGS.max_input_sequence_len)
#         iter_unchange = 0
#         for j in range(maxsteps):
#             if iter_unchange > 5000 or (time.perf_counter() - start) > 1800:
#                 break
#             # generate a random number an compare with wp
#             bid_empty = np.where(result_tmp == 0)[0]
#             if np.random.random_sample(1) < wp:
#                 # pick a random bid
#                 pick_index = random.sample(list(bid_empty), 1)[0]
#                 result_tmp[pick_index] = 1
#                 age = age + 1
#                 age[pick_index] = 0
#             else:
#                 score_tmp = bid_price[bid_empty] / bid_size[bid_empty]
#                 sorted_result = sorted(zip(score_tmp, bid_empty), reverse=True)
#                 _, bid_empty_new = zip(*sorted_result)
#                 if age[bid_empty_new[0]] >= age[bid_empty_new[1]]:
#                     pick_index = bid_empty_new[0]
#                     result_tmp[pick_index] = 1
#                 else:
#                     if np.random.random_sample(1) < novelp:
#                         pick_index = bid_empty_new[1]
#                         result_tmp[pick_index] = 1
#                     else:
#                         pick_index = bid_empty_new[0]
#                         result_tmp[pick_index] = 1
#             age = age + 1
#             bid_choose = np.where(result_tmp==1)[0]
#             age[bid_choose] = 0

#             while (np.dot(result_tmp.reshape(1, -1), instance[:, 0:-1]) > unit).any():
#                 conflict_item = np.where(np.dot(result_tmp.reshape(1, -1), instance[:, 0:-1]) > unit)[1]
#                 # result_tmp[pick_index]=0
#                 bid_choose = np.where(result_tmp == 1)[0]
#                 bundle_choose = instance[bid_choose, 0:-1]
#                 conflict_bid = np.array([])
#                 for item_index in list(conflict_item):
#                     conflict_bid = np.append(conflict_bid, np.where(bundle_choose[:, item_index] != 0)[0])
#                 conflict_bid = np.unique(conflict_bid).astype(int)
#                 conflict_bid_index = bid_choose[conflict_bid]
#                 # remove the one with the smallest price
#                 score_tmp = bid_price[conflict_bid_index] / bid_size[conflict_bid_index]
#                 result_tmp[conflict_bid_index[np.argmin(score_tmp)]] = 0
#                 # result_tmp[pick_index]=1
#             # decide whether update the optimal result
#             revenue_tmp = sum(result_tmp * bid_price)
#             if revenue_tmp > revenue_opt:
#                 revenue_opt = revenue_tmp
#                 result_opt = result_tmp.copy()
#                 iter_unchange = 0
#             else:
#                 iter_unchange = iter_unchange + 1
#         if revenue_opt > revenue_global:
#             revenue_global = revenue_opt
#             result_global = result_opt.copy()


#     result = fromvector2index(result_global)
#     result_len = len(result)
#     result = result + [0] * (FLAGS.max_input_sequence_len - result_len)
#     return result, revenue_global
#     pass

def main():
    test_data = WdpDataGenerator(FLAGS, 'test', 2000)

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

    pred, tar, pred_time = eval(test_data, my_model)
    print('The total time of prediction is ', pred_time/pred.shape[0])
   
   
    # use the model to predict
    inp = test_data.inputs[:tar.shape[0]]
    inp = np.delete(inp,range(FLAGS.num_bid,FLAGS.max_input_sequence_len),axis=1)
    u = test_data.units[:tar.shape[0]]
    print('pred:', pred.shape)
    print('tar:', tar.shape)
    print('inp', inp.shape)
    print('u', u.shape)

    # basic check algorithm
    unit_ = []
    current_revenue = np.zeros(tar.shape[0])
    target_revenue = np.zeros(tar.shape[0])
    for i in range(tar.shape[0]):
        bid_pred = pred[i]
        bid_tar = tar[i]
        current_u = np.zeros(FLAGS.num_item)
        current_r = 0
        target_revenue[i] = sum([inp[i][bid_tar[k]-1][-1] if bid_tar[k]>0 else 0 for k in range(FLAGS.num_bid)])
        for j in range(FLAGS.num_bid):
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

    accuracy_check(pred, tar)
    feasible_check(unit_, u)
    gap_check(current_revenue, target_revenue)

    # improvement using local searching
    local_p = []
    local_r = []
    start = time.perf_counter()
    for i in range(pred.shape[0]):
        if all(pred[i] == tar[i]):
            local_p.append(pred[i])
            local_r.append(current_revenue[i])
            pass
        else:
            instance = inp[i]
            unit = u[i]
            result_opt = fromindex2vector(pred[i][:], FLAGS.num_bid)
            result_opt = np.array(result_opt)
            if i == 20:
                print(instance)
                print(unit)
                print(result_opt)
            local_result, local_revenue  = local_search(instance, unit, result_opt, wp = 0.5)
            # print("instance:")
            # print(instance)
            # print("unit:")
            # print(unit)
            # print('tar:')
            # print(tar[i])
            # print('tar_revenue:')
            # print(target_revenue[i])
            # print('local_result:')
            # print(local_result)
            # print('local_revenue')
            # print(local_revenue)
            local_p.append(local_result)
            local_r.append(local_revenue)
        print('___finish___{}___search'.format(i+1))
    # use for debug
    #     if i == 0:
    #         break
    # return 0
    # use for debug
    walltime = time.perf_counter() - start
    print('Average time is {} s'.format(walltime/pred.shape[0]))
    local_p = np.stack(local_p)
    local_r = np.stack(local_r)
    accuracy_check(local_p, tar)
    gap_check(local_r, target_revenue)

    # using casavona
    # casa_p = []
    # casa_r = []
    # start = time.perf_counter()
    # for i in range(pred.shape[0]):
    #     if all(pred[i] == tar[i]):
    #         casa_p.append(pred[i])
    #         casa_r.append(current_revenue[i])
    #         pass
    #     else:
    #         instance = inp[i]
    #         unit = u[i]
    #         result_opt = fromindex2vector(pred[i][:], FLAGS.max_input_sequence_len)
    #         result_opt = np.array(result_opt)
    #         casa_result, casa_revenue = Casanova(instance, unit, result_opt, maxtries=3, wp=0.8, novelp=0.5)
    #         print('tar:')
    #         print(tar[i])
    #         print('tar_revenue:')
    #         print(target_revenue[i])
    #         print('case_result:')
    #         print(casa_result)
    #         print('case_revenue')
    #         print(casa_revenue)
    #         casa_p.append(casa_result)
    #         casa_r.append(casa_revenue)
    #     print('___finish___{}___search'.format(i+1))
    # walltime = time.perf_counter() - start
    # print('Average time is {} s'.format(walltime/pred.shape[0]))
    # casa_p = np.stack(casa_p)
    # casa_r = np.stack(casa_r)
    # accuracy_check(casa_p, tar)
    # gap_check(casa_r, target_revenue)
if __name__ == '__main__':
    main()