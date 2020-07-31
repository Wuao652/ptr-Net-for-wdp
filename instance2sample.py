######################
# Compared with instance2sample.py
# This file iterates over all data generation method
######################

import os
import argparse
import pickle
import glob
import numpy as np
import gzip

import utilities
from wdp import WDP
import win_unicode_console
win_unicode_console.enable()

# 方法2 指定index最小的被选择的bid为1  其他为0
# 然后去掉该bid 将剩下的图作为新的sample 一直到所有可选择的bid都被移除
def collect_samples_2(instances, instances_units, out_dir, rng, n_samples, n_items, n_bids, node_record_prob):
    i = 0
    i_instance = 0
    while i < n_samples and i_instance < len(instances):
        multi_flag = False
        instance = instances[i_instance]
        instance_matrix = np.loadtxt(instance)
        if len(instances_units) != 0:
            instance_units = instances_units[i_instance]
            units_matrix = np.loadtxt(instance_units)
            multi_flag = True
        else:
            units_matrix = np.ones(instance_matrix.shape[1] - 1)


        if multi_flag:
            bid_recover = utilities.bid_recover(instance, instance_units)
        else:
            bid_recover = utilities.bid_recover(instance)

        X = WDP(bid_recover)  # initialize WDP class
        X.initialize_mip(verbose=True)  # initialize MIP
        _, bid_result = X.solve_mip()  # solve MIP
        bid_result = bid_result.reshape(-1)

        # print('='*20)
        # print("generating the {}th sample".format(i + 1))
        # print('Original solution')
        # print(bid_result)
        # print('original_matrix:')
        # print(instance_matrix)
        # print('original_units:')
        # print(units_matrix)
        # print('=' * 20)

        filename = '{}/sample_{}.pkl'.format(out_dir, i + 1)
        with gzip.open(filename, 'wb') as f:
            pickle.dump({
                'instance_matrix': instance_matrix,
                'bid_selection': bid_result,
                'unit_matrix': units_matrix,
            }, f)
        i = i + 1
        # break
        while sum(bid_result) != 0 and len(bid_result) > 1 and i < n_samples:
            expert_action = np.where(bid_result != 0)[0][0]

            bid_select = instance_matrix[expert_action, :]

            units_matrix = units_matrix - bid_select[:-1]
            bid_del = [expert_action]

            for k in range(instance_matrix.shape[0]):
                if (instance_matrix[k, :-1] > units_matrix).any() and k != expert_action:
                    bid_del.append(k)

            unit_del = np.where(units_matrix == 0)
            units_matrix = np.delete(units_matrix, unit_del)
            instance_matrix = np.delete(instance_matrix, bid_del, axis=0)
            instance_matrix = np.delete(instance_matrix, unit_del, axis=1)
            bid_result = np.delete(bid_result, bid_del)
            if sum(bid_result) == 0:
                break
            if rng.random_sample(1) <= node_record_prob:
                # print(n_items - bid_result.shape[0])
                bid_result_output = np.pad(bid_result, (0, n_bids - bid_result.shape[0]), 'constant')
                units_matrix_output = np.pad(units_matrix, (0, n_items - units_matrix.shape[0]), 'constant')
                bundle = instance_matrix[:, :-1]
                value = instance_matrix[:, -1].reshape(-1)
                bundle = np.pad(bundle, ((0, n_bids-bundle.shape[0]), (0, n_items - bundle.shape[1])), 'constant')
                value = np.pad(value, (0, n_bids - value.shape[0]), 'constant')
                value = value[:, np.newaxis]
                instance_matrix_output = np.concatenate((bundle, value), axis=1)
                print("generating the {}th sample".format(i + 1))
                # print('instance_matrix:')
                # print(instance_matrix_output)
                # print('units:')
                # print(units_matrix_output)
                # print('solution')
                # print(bid_result_output)
                filename = '{}/sample_{}.pkl'.format(out_dir, i + 1)
                with gzip.open(filename, 'wb') as f:
                    pickle.dump({
                        'instance_matrix': instance_matrix_output,
                        'bid_selection': bid_result_output,
                        'unit_matrix': units_matrix_output,
                    }, f)
                i = i + 1

                if i == n_samples:
                    break

        i_instance = i_instance + 1

    if i < n_samples and i_instance >= len(instances):
        print("Instances are not enough to generate needed smaples.")


    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-u', '--usage',
        help='The data is generated for which process.',
        choices=['train', 'valid', 'test'],
    )
    parser.add_argument(
        '-s', '--seed',
        help='Random generator seed (default 0).',
        type=utilities.valid_seed,
        default=0,
    )
    parser.add_argument(
        '-ni', '--num_item',
        help='Number of items (default 0).',
        type=int,
        default=0,
    )
    parser.add_argument(
        '-nb', '--num_bid',
        help='Number of bids (default 0).',
        type=int,
        default=0,
    )
    parser.add_argument(
        '-ns', '--num_sample',
        help='Number of samples to be collected (default 0).',
        type=int,
        default=0,
    )
    parser.add_argument(
        '-um', '--unit_max',
        type=int,
        default=5,
    )
    parser.add_argument(
        '-bs', '--bid_size',
        help='Size of each bid for uniform distribution (default 0).',
        type=int,
        default=6,
    )
    parser.add_argument(
        '-p', '--prob',
        help='Probability of adding an item for decay distribution (default 0).',
        type=utilities.valid_prob,
        default=75,
    )
    args = parser.parse_args()
    # print(f"seed {args.seed}")

    size = args.num_sample
    nitems = args.num_item
    nbids = args.num_bid
    if nitems == 0:
        nitems = int(nbids / 10)

    instances_units = []
    node_record_prob = 0.8

    instances = glob.glob('./data/instance/{}_{}/{}/{}/binary/*.txt'.format(nitems, nbids, args.unit_max, args.usage))
    instances_units = glob.glob('data/instance/{}_{}/{}/{}/unit/*.txt'.format(nitems, nbids, args.unit_max, args.usage))
    out_dir = './data/sample/{}_{}/{}/{}'.format(nitems, nbids, args.unit_max, args.usage)

    print("{}{}instances for {} samples using the method ".format(len(instances), args.usage, size))
    # create output directory
    if not os.path.exists('{}'.format(out_dir)):
        os.makedirs('{}'.format(out_dir))
    print('Create the output file!')
    rng = np.random.RandomState(args.seed)
    collect_samples_2(instances, instances_units, '{}'.format(out_dir), rng, size, nitems, nbids, node_record_prob)

