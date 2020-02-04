
# Running a subset of a program in parallel with scoop / multiprocessing in
# Python requires the code in question to be separated into its own file and
# executed within an if __name__ == '__main__' clause

import argparse
import copy
import pickle
import numpy as np
from collections import OrderedDict
from scoop import futures


def find_indices(index_1, prop_val_1, node_val, scale):
    """
    Finds indices of propensity values in continuous scale for use in
    interpolation calculation
    """

    if index_1 < (scale.shape[0] - 1):
        if prop_val_1 <= node_val:
            index_2 = index_1 + 1
        elif prop_val_1 > node_val:
            index_2 = index_1 - 1
    else:
        index_2 = index_1 - 1

    return index_2


def linear_interpolation(node_val, aa_propensity_scale, dict_label):
    """
    Interpolates propensity value for node property value
    """

    if type(aa_propensity_scale) != np.ndarray:
        raise TypeError('Scale {} is not a numpy array'.format(dict_label))

    else:
        node_val_scale = aa_propensity_scale[0]
        propensity_scale = aa_propensity_scale[1]
        if node_val_scale[0] <= node_val <= node_val_scale[-1]:
            index_1 = (np.abs(node_val_scale-node_val)).argmin()
            prop_val_1 = node_val_scale[index_1]
            propensity_1 = propensity_scale[index_1]

            index_2 = find_indices(index_1, prop_val_1, node_val, node_val_scale)
            prop_val_2 = node_val_scale[index_2]
            propensity_2 = propensity_scale[index_2]

            # Linear interpolation
            weight_1 = abs(prop_val_2 - node_val) / abs(prop_val_2 - prop_val_1)
            weight_2 = abs(prop_val_1 - node_val) / abs(prop_val_2 - prop_val_1)
            propensity = (propensity_1*weight_1) + (propensity_2*weight_2)

        else:
            propensity = np.nan
            print('Parameter values of input backbone coordinate model '
                  'structural features ({}) are outside of the range of '
                  'parameter values used to construct propensity scales - node '
                  'value = {}, parameter range = ({}, {})'.format(
                      dict_label, node_val, aa_propensity_scale[0][0],
                      aa_propensity_scale[0][-1]
                ))

    return propensity


def look_up_indv_propensity(G, node_1, scale, label, weight, label_indices):
    """
    """

    aa_1 = G.nodes[node_1]['aa_id']

    node_prop = label.split('_')[label_indices['prop1']]
    node_val = np.nan

    if node_prop != '-':
        try:
            node_val = G.nodes[node_1][node_prop]
        except KeyError:
            raise KeyError('{} not defined for node {}'.format(node_prop, node_1))

    # Converts non-float values into np.nan
    if node_val in ['', 'nan', 'NaN', np.nan]:
        node_val = np.nan

    value = np.nan
    if (    label.split('_')[label_indices['discorcont']] == 'cont'
        and not np.isnan(node_val)
    ):
        # Interpolate dictionary
        value = linear_interpolation(node_val, scale[aa_1], label)
    elif label.split('_')[label_indices['discorcont']] == 'disc':
        # Filter dataframe
        scale_copy = copy.deepcopy(scale).set_index('FASTA', drop=True)
        if node_prop == '-':
            try:
                value = scale_copy.iloc[:,0][aa_1]
            except KeyError:
                raise Exception('{} not defined in {}'.format(aa_1, label))
        elif node_prop == 'phipsi':
            if not np.isnan(node_val):
                try:
                    value = scale_copy[node_val][aa_1]
                except KeyError:
                    raise Exception('{} not defined in {}'.format(aa_1, label))
        else:
            raise ValueError('Node property {} not recognised'.format(node_prop))
    else:
        raise ValueError('Scale {} not labelled as being continuous or '
                         'discrete'.format(label))

    if value < 0:
        raise ValueError('{} returned after interpolation of {} for node '
                         '{}'.format(value, label, node_1))
    elif not np.isnan(value) and value != 0:
        if label.split('_')[label_indices['proporfreq']] == 'propensity':
            # NOTE: Must take -ve logarithm of each
            # individual propensity score before summing
            # (rather than taking the -ve logarithm of the
            # summed propensities)
            value = weight*np.negative(np.log(value))
        elif label.split('_')[label_indices['proporfreq']] == 'frequency':
            value *= weight
    else:
        value = 0

    return value


def look_up_pair_propensity(G, node_1, scale, label, weight, label_indices):
    """
    """

    total_value = 0

    # Loops through each node pair to sum pairwise propensity values
    for node_pair in G.edges(node_1):
        # No need to randomly order pair since each will be counted
        # twice in this analysis (once for node 1 and once for node 2)
        node_2 = node_pair[1]
        aa_1 = G.nodes[node_1]['aa_id']
        aa_2 = G.nodes[node_2]['aa_id']
        aa_pair = '{}_{}'.format(aa_1, aa_2)

        # Loops through each interaction between a pair of nodes
        for edge in G[node_1][node_2]:
            edge_label = G[node_1][node_2][edge]['interaction']

            if label.split('_')[label_indices['interactiontype']] == edge_label:
                node_prop = label.split('_')[label_indices['prop1']]
                node_val = np.nan

                if node_prop != '-':
                    try:
                        node_val = G.nodes[node_1][node_prop]
                    except KeyError:
                        raise KeyError('{} not defined for node {}'.format(
                            node_prop, node_1
                        ))

                # Converts non-float values into np.nan
                if node_val in ['', 'nan', 'NaN', np.nan]:
                    node_val = np.nan

                value = np.nan
                if (    label.split('_')[label_indices['discorcont']] == 'cont'
                    and not np.isnan(node_val)
                ):
                    aa_scale = scale[aa_pair]
                    value = linear_interpolation(node_val, aa_scale, label)

                elif label.split('_')[label_indices['discorcont']] == 'disc':
                    # Filter dataframe
                    scale_copy = scale.set_index('FASTA', drop=True)
                    value = scale_copy[aa_1][aa_2]
                else:
                    raise ValueError('Scale {} not labelled as being continuous'
                                     ' or discrete'.format(label))

                if not np.isnan(value):
                    if label.split('_')[label_indices['proporfreq']] == 'propensity':
                        # NOTE: Must take -ve logarithm of each
                        # individual propensity score before
                        # summing (rather than taking the -ve
                        # logarithm of the summed propensities)
                        value = weight*np.negative(np.log(value))
                        total_value += value
                    elif label.split('_')[label_indices['proporfreq']] == 'frequency':
                        value *= weight
                        total_value += value

    return total_value


def measure_fitness_propensity(
    num, G, dicts, label_indices, barrel_or_sandwich
):
    """
    Measures fitness of amino acid sequences from their propensities for
    the structural features of the input backbone structure.
    """

    # Total propensity count (across all nodes in network)
    propensity_count = 0
    frequency_count = 0

    for node_1 in list(G.nodes):
        # Calculates interpolated propensity of each node for all individual
        # structural features considered

        for tup in dicts:
            label = tup[0]
            weight = tup[1]
            scale = tup[2]

            # Checks dict matches node surface (int or ext) and in the case of
            # sandwiches strand location (edge or central)
            dict_surf = label.split('_')[label_indices['intorext']]
            node_surf = G.nodes[node_1]['int_ext']
            dict_strand = label.split('_')[label_indices['edgeorcent']]
            node_strand = G.nodes[node_1]['eoc']
            if (
                   (dict_surf != node_surf)
                or (dict_strand != '-' and dict_strand != node_strand)
            ):  # Can't just be dict_strand != node_strand because there will
            # be some strand-generic dictionaries that you won't want to exclude
                continue

            # Looks up propensity / frequency of node in dictionary
            dict_indv_pair = label.split('_')[label_indices['pairorindv']]
            if dict_indv_pair == 'indv':
                value = look_up_indv_propensity(
                    G, node_1, scale, label, weight, label_indices
                )
            elif dict_indv_pair == 'pair':
                value = look_up_pair_propensity(
                    G, node_1, scale, label, weight, label_indices
                )
            else:
                raise ValueError('Scale {} not labelled as being for individual'
                                 ' or pairs of amino acids'.format(label))

            # Adds propensity/ frequency score to total
            dict_prop_freq = label.split('_')[label_indices['proporfreq']]
            if dict_prop_freq == 'propensity':
                propensity_count += value
            elif dict_prop_freq == 'frequency':
                frequency_count += value
            else:
                raise ValueError('Scale {} not labelled as either a propenity '
                                 'or a frequency scale'.format(label))

    if propensity_count == 0 or np.isnan(propensity_count):
        print('WARNING: propensity for network {} is '
              '{}'.format(num, propensity_count))
    if np.isnan(frequency_count):
        print('WARNING: frequency count for network {} is '
              '{}'.format(num, frequency_count))

    return [num, propensity_count, frequency_count]


if __name__ == '__main__':
    # Reads in command line inputs
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', help='Absolute file path to pickled dictionary '
                        'of sequence networks')
    parser.add_argument('-dicts', help='Absolute file path to pickled '
                        'list of propensity and frequency scales and their '
                        'weights')
    parser.add_argument('-indices', help='Absolute file path to pickled '
                        'dictionary of propensity and frequency scale labelling'
                        ' scheme')
    parser.add_argument('-bos', help='Specifies whether the structure is a '
                        'beta-barrel or -sandwich')
    parser.add_argument('-o', '--output', help='Location to which to save the '
                        'output pickled dictionary of propensity and frequency '
                        'scores')
    args = parser.parse_args()

    networks_dict = vars(args)['net']
    with open(networks_dict, 'rb') as f:
        networks_dict = pickle.load(f)
    dicts_list = vars(args)['dicts']
    with open(dicts_list, 'rb') as f:
        dicts_list = pickle.load(f)
    dicts_list = [copy.deepcopy(dicts_list)
                  for n in range(len(networks_dict))]
    label_indices = vars(args)['indices']
    with open(label_indices, 'rb') as f:
        label_indices = pickle.load(f)
    label_indices = [copy.deepcopy(label_indices)
                     for n in range(len(networks_dict))]
    barrel_or_sandwich = vars(args)['bos']
    barrel_or_sandwich = [copy.deepcopy(barrel_or_sandwich)
                          for n in range(len(networks_dict))]
    wd = vars(args)['output']

    network_fitness_list = futures.map(
        measure_fitness_propensity, list(networks_dict.keys()),
        list(networks_dict.values()), dicts_list, label_indices,
        barrel_or_sandwich
    )

    network_propensity_scores = OrderedDict()
    network_frequency_scores = OrderedDict()
    for tup in network_fitness_list:
        network_propensity_scores[tup[0]] = tup[1]
        network_frequency_scores[tup[0]] = tup[2]

    with open('{}/Network_prop_freq_scores.pkl'.format(wd), 'wb') as f:
        pickle.dump((network_propensity_scores, network_frequency_scores), f)
