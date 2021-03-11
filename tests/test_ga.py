
# python -m unittest tests/test_ga.py

import copy
import networkx as nx
import numpy as np
import pandas as pd
import unittest
from collections import OrderedDict
from betadesigner.subroutines.calc_propensity_in_parallel import measure_fitness_propensity
from betadesigner.subroutines.find_parameters import initialise_ga_object
from betadesigner.subroutines.run_genetic_algorithm import run_ga_calcs


def define_params():
    """
    """

    prop_freq_dicts = gen_prop_and_freq_distributions()

    params = {'paramopt': False,
              'inputdataframepath': '',
              'inputdataframe': '',
              'inputpdb': '',
              'propensityscales': {key: val for key, val in prop_freq_dicts.items()
                                   if 'propensity' in key},
              'frequencyscales': {key: val for key, val in prop_freq_dicts.items()
                                  if 'frequency' in key},
              'aacodes': ['A', 'R', 'N'],
              'scaleweights': {key: 1 for key in prop_freq_dicts.keys()},
              'dictnameindices': {'intorext': 0,
                                  'edgeorcent': 1,
                                  'prop1': 2,
                                  'interactiontype': 3,
                                  'pairorindv': 4,
                                  'discorcont': 5,
                                  'proporfreq': 6},
              'propensityweight': 0.5,
              'workingdirectory': '',
              'uniworkdir': '',
              'barrelorsandwich': '2.40',
              'jobid': '',
              'initialseqmethod': '',
              'fitnessscoremethod': 'split',
              'splitfraction': 0.5,
              'matingpopmethod': '',
              'unfitfraction': 0.5,
              'crossovermethod': '',
              'crossoverprob': 0.5,
              'swapstartprob': '',
              'swapstopprob': '',
              'mutationmethod': '',
              'mutationprob': 0.5,
              'populationsize': 2,
              'propensitypopsize': 2*0.5,
              'maxnumgenerations': 10}

    return params


def gen_prop_and_freq_distributions():
    """
    Defines example propensity and frequency dictionaries
    """

    dicts = {'-_-_z_-_indv_cont_propensity': {'A': np.array([[-10, 0, 10], [0.2, 0.2, 0.8]]),
                                              'R': np.array([[-10, 0, 10], [1.1, 0.9, 1.2]]),
                                              'N': np.array([[-10, 0, 10], [0.3, 0.7, 0.8]])},
             'int_-_z_-_indv_cont_propensity': {'A': np.array([[-10, 0, 10], [2.5, 1.5, 0.5]]),
                                                'R': np.array([[-10, 0, 10], [0.5, 1.5, 2.5]]),
                                                'N': np.array([[-10, 0, 10], [0.5, 2.5, 0.5]])},
             'int_-_z_hb_pair_cont_propensity': {'A_A': np.array([[-10, 0, 10], [0.5, 1.0, 1.5]]),
                                                 'A_R': np.array([[-10, 0, 10], [0.1, 2.0, 0.1]]),
                                                 'A_N': np.array([[-10, 0, 10], [2.0, 1.2, 0.4]]),
                                                 'R_A': np.array([[-10, 0, 10], [0.1, 2.0, 0.1]]),
                                                 'R_R': np.array([[-10, 0, 10], [2.0, 0.5, 1.8]]),
                                                 'R_N': np.array([[-10, 0, 10], [1.5, 1.2, 0.9]]),
                                                 'N_A': np.array([[-10, 0, 10], [2.0, 1.2, 0.4]]),
                                                 'N_R': np.array([[-10, 0, 10], [1.5, 1.2, 0.9]]),
                                                 'N_N': np.array([[-10, 0, 10], [0.6, 0.3, 1.7]])},
             'ext_-_z_plusminus1_pair_cont_propensity': {'A_A': np.array([[-10, 0, 10], [0.6, 2.0, 0.4]]),
                                                         'A_R': np.array([[-10, 0, 10], [2.5, 1.2, 0.4]]),
                                                         'A_N': np.array([[-10, 0, 10], [0.5, 0.2, 1.4]]),
                                                         'R_A': np.array([[-10, 0, 10], [0.6, 1.0, 1.2]]),
                                                         'R_R': np.array([[-10, 0, 10], [1.4, 1.3, 0.6]]),
                                                         'R_N': np.array([[-10, 0, 10], [0.3, 1.6, 0.7]]),
                                                         'N_A': np.array([[-10, 0, 10], [2.1, 1.0, 1.0]]),
                                                         'N_R': np.array([[-10, 0, 10], [1.9, 1.2, 0.1]]),
                                                         'N_N': np.array([[-10, 0, 10], [0.8, 0.9, 1.0]])},
             'ext_-_z_plusminus2_pair_cont_propensity': {'A_A': np.array([[-10, 0, 10], [1.0, 1.9, 0.6]]),
                                                         'A_R': np.array([[-10, 0, 10], [2.2, 0.8, 1.9]]),
                                                         'A_N': np.array([[-10, 0, 10], [1.7, 0.3, 1.9]]),
                                                         'R_A': np.array([[-10, 0, 10], [2.0, 1.1, 1.6]]),
                                                         'R_R': np.array([[-10, 0, 10], [1.4, 1.4, 1.5]]),
                                                         'R_N': np.array([[-10, 0, 10], [1.2, 0.8, 0.1]]),
                                                         'N_A': np.array([[-10, 0, 10], [1.2, 1.5, 1.0]]),
                                                         'N_R': np.array([[-10, 0, 10], [0.8, 1.1, 0.2]]),
                                                         'N_N': np.array([[-10, 0, 10], [0.6, 0.6, 0.4]])},
             'int_-_-_-_indv_disc_frequency': pd.DataFrame({'FASTA': ['A', 'R', 'N'],
                                                            'int': [0.8, 0.15, 0.05]}),
             'ext_-_-_-_indv_disc_frequency': pd.DataFrame({'FASTA': ['A', 'R', 'N'],
                                                            'ext': [0.5, 0.05, 0.45]})}

    return dicts


def gen_sequence_networks():
    """
    """

    network_1 = nx.MultiGraph()
    network_1.add_node(1, type='strand', aa_id='A', int_ext='int', eoc='edge', z=-5)
    network_1.add_node(2, type='strand', aa_id='A', int_ext='int', eoc='central', z=0)
    network_1.add_node(3, type='strand', aa_id='A', int_ext='int', eoc='edge', z=7)
    network_1.add_edge(1, 2, interaction='hb')
    network_1.add_edge(2, 3, interaction='hb')

    network_2 = nx.MultiGraph()
    network_2.add_node(1, type='strand', aa_id='A', int_ext='int', eoc='edge', z=-5)
    network_2.add_node(2, type='strand', aa_id='N', int_ext='int', eoc='central', z=0)
    network_2.add_node(3, type='strand', aa_id='R', int_ext='int', eoc='edge', z=7)
    network_2.add_edge(1, 2, interaction='hb')
    network_2.add_edge(2, 3, interaction='hb')

    network_3 = nx.MultiGraph()
    network_3.add_node(1, type='strand', aa_id='R', int_ext='int', eoc='-', z=-5)
    network_3.add_node(2, type='strand', aa_id='A', int_ext='int', eoc='-', z=0)
    network_3.add_node(3, type='strand', aa_id='N', int_ext='int', eoc='-', z=7)
    network_3.add_edge(1, 2, interaction='hb')
    network_3.add_edge(2, 3, interaction='hb')

    network_4 = nx.MultiGraph()
    network_4.add_node(1, type='strand', aa_id='N', int_ext='int', eoc='-', z=-5)
    network_4.add_node(2, type='strand', aa_id='R', int_ext='int', eoc='-', z=0)
    network_4.add_node(3, type='strand', aa_id='N', int_ext='int', eoc='-', z=7)
    network_4.add_edge(1, 2, interaction='hb')
    network_4.add_edge(2, 3, interaction='hb')

    network_5 = nx.MultiGraph()
    network_5.add_node(1, type='strand', aa_id='N', int_ext='int', eoc='-', z=-5)
    network_5.add_node(2, type='strand', aa_id='A', int_ext='int', eoc='-', z=0)
    network_5.add_node(3, type='loop', aa_id='N', int_ext='int', eoc='-', z=7)
    network_5.add_edge(1, 2, interaction='hb')
    network_5.add_edge(2, 3, interaction='hb')

    network_6 = nx.MultiGraph()
    network_6.add_node(1, type='loop', aa_id='A', int_ext='int', eoc='-', z=-5)
    network_6.add_node(2, type='loop', aa_id='R', int_ext='int', eoc='-', z=0)
    network_6.add_node(3, type='loop', aa_id='R', int_ext='int', eoc='-', z=7)
    network_6.add_edge(1, 2, interaction='hb')
    network_6.add_edge(2, 3, interaction='hb')

    test_dict = {1: network_1,
                 2: network_2,
                 3: network_3,
                 4: network_4,
                 5: network_5,
                 6: network_6}

    return test_dict


class testGeneticAlgorithm(unittest.TestCase):
    """
    """

    def gen_network_prop_and_freq(self):
        """
        """

        params = define_params()
        test_dict = gen_sequence_networks()
        prop_freq_dicts = gen_prop_and_freq_distributions()
        prop_freq_list = []
        for label, scale in prop_freq_dicts.items():
            weight = params['scaleweights'][label]
            prop_freq_list.append((label, weight, scale))

        network_prop = OrderedDict()
        network_freq = OrderedDict()
        for num, network in test_dict.items():
            num, prop, freq = measure_fitness_propensity(
                num, network, prop_freq_list, params['dictnameindices'], '2.60', True
            )
            network_prop[num] = prop
            network_freq[num] = freq

        return network_prop, network_freq

    def test_measure_fitness_propensity(self):
        """
        """

        print('Testing measure_fitness_propensity()')

        network_propensities, network_frequencies = self.gen_network_prop_and_freq()

        np.testing.assert_almost_equal(network_propensities[1], -0.887891, decimal=5)
        np.testing.assert_almost_equal(network_propensities[2], -3.222491, decimal=5)
        np.testing.assert_almost_equal(network_propensities[3], -0.978747, decimal=5)
        np.testing.assert_almost_equal(network_propensities[4], -1.560937, decimal=5)
        np.testing.assert_almost_equal(network_propensities[5], -1.645576, decimal=5)
        np.testing.assert_almost_equal(network_propensities[6], 0, decimal=5)
        np.testing.assert_almost_equal(network_frequencies[1], 2.4, decimal=5)
        np.testing.assert_almost_equal(network_frequencies[2], 1, decimal=5)
        np.testing.assert_almost_equal(network_frequencies[3], 1, decimal=5)
        np.testing.assert_almost_equal(network_frequencies[4], 0.25, decimal=5)
        np.testing.assert_almost_equal(network_frequencies[5], 0.85, decimal=5)
        np.testing.assert_almost_equal(network_frequencies[6], 0, decimal=5)

    def test_combine_prop_and_freq_scores(self):
        """
        """

        print('Testing combine_prop_and_freq_scores()')

        params = define_params()
        network_propensities, network_frequencies = self.gen_network_prop_and_freq()

        # Test 1
        bayes_params = {'propensityweight': 0.5,
                        'unfitfraction': 0.2,
                        'crossoverprob': 0.1,
                        'mutationprob': 0.05}
        ga_calcs = run_ga_calcs({**params, **bayes_params}, test=True)

        network_fitness_scores = ga_calcs.combine_prop_and_freq_scores(
            network_propensities, network_frequencies, 'raw'
        )
        np.testing.assert_almost_equal(network_fitness_scores[1], 0.247723, decimal=5)
        np.testing.assert_almost_equal(network_fitness_scores[2], 0.395932, decimal=5)
        np.testing.assert_almost_equal(network_fitness_scores[3], 0.123260, decimal=5)
        np.testing.assert_almost_equal(network_fitness_scores[4], 0.080633, decimal=5)
        np.testing.assert_almost_equal(network_fitness_scores[5], 0.140294, decimal=5)
        np.testing.assert_almost_equal(network_fitness_scores[6], 0.012156, decimal=5)

        # Test 2
        bayes_params = {'propensityweight': 0.8,
                        'unfitfraction': 0.2,
                        'crossoverprob': 0.1,
                        'mutationprob': 0.05}
        ga_calcs = run_ga_calcs({**params, **bayes_params}, test=True)
        network_fitness_scores = ga_calcs.combine_prop_and_freq_scores(
            network_propensities, network_frequencies, 'raw'
        )
        np.testing.assert_almost_equal(network_fitness_scores[1], 0.134538, decimal=5)
        np.testing.assert_almost_equal(network_fitness_scores[2], 0.524400, decimal=5)
        np.testing.assert_almost_equal(network_fitness_scores[3], 0.088125, decimal=5)
        np.testing.assert_almost_equal(network_fitness_scores[4], 0.101741, decimal=5)
        np.testing.assert_almost_equal(network_fitness_scores[5], 0.131743, decimal=5)
        np.testing.assert_almost_equal(network_fitness_scores[6], 0.019451, decimal=5)

        # Test 3
        bayes_params = {'propensityweight': 1,
                        'unfitfraction': 0.2,
                        'crossoverprob': 0.1,
                        'mutationprob': 0.05}
        ga_calcs = run_ga_calcs({**params, **bayes_params}, test=True)
        network_fitness_scores = ga_calcs.combine_prop_and_freq_scores(
            network_propensities, network_frequencies, 'raw'
        )
        np.testing.assert_almost_equal(network_fitness_scores[1], 0.059082, decimal=5)
        np.testing.assert_almost_equal(network_fitness_scores[2], 0.610045, decimal=5)
        np.testing.assert_almost_equal(network_fitness_scores[3], 0.064701, decimal=5)
        np.testing.assert_almost_equal(network_fitness_scores[4], 0.115813, decimal=5)
        np.testing.assert_almost_equal(network_fitness_scores[5], 0.126042, decimal=5)
        np.testing.assert_almost_equal(network_fitness_scores[6], 0.024313, decimal=5)

        # Test 4
        bayes_params = {'propensityweight': 0.6,
                        'unfitfraction': 0.2,
                        'crossoverprob': 0.1,
                        'mutationprob': 0.05}
        ga_calcs = run_ga_calcs({**params, **bayes_params}, test=True)
        network_fitness_scores = ga_calcs.combine_prop_and_freq_scores(
            network_propensities, network_frequencies, 'rank'
        )
        np.testing.assert_almost_equal(network_fitness_scores[1], 0.231688, decimal=5)
        np.testing.assert_almost_equal(network_fitness_scores[2], 0.244155, decimal=5)
        np.testing.assert_almost_equal(network_fitness_scores[3], 0.158441, decimal=5)
        np.testing.assert_almost_equal(network_fitness_scores[4], 0.132467, decimal=5)
        np.testing.assert_almost_equal(network_fitness_scores[5], 0.204675, decimal=5)
        np.testing.assert_almost_equal(network_fitness_scores[6], 0.028571, decimal=5)

    def test_convert_energy_to_probability(self):
        """
        """

        print('Testing convert_energies_to_probabilities()')

        params = define_params()
        bayes_params = {'propensityweight': 0.5,
                        'unfitfraction': 0.75,
                        'crossoverprob': 0.1,
                        'mutationprob': 0.05}
        network_energies = {1: -10,
                            2: 25,
                            3: 0,
                            4: 7,
                            5: -16}
        expected_network_prob = {1: 0.0783821,
                                 2: 0.0000000451189,
                                 3: 0.00129241,
                                 4: 0.0000730178,
                                 5: 0.920252}

        ga_calcs = run_ga_calcs({**params, **bayes_params}, test=True)
        network_fitness_scores = ga_calcs.convert_energies_to_probabilities(
            network_energies
        )

        np.testing.assert_almost_equal(
            network_fitness_scores[1], expected_network_prob[1], decimal=6
        )
        np.testing.assert_almost_equal(
            network_fitness_scores[2], expected_network_prob[2], decimal=12
        )
        np.testing.assert_almost_equal(
            network_fitness_scores[3], expected_network_prob[3], decimal=7
        )
        np.testing.assert_almost_equal(
            network_fitness_scores[4], expected_network_prob[4], decimal=9
        )
        np.testing.assert_almost_equal(
            network_fitness_scores[5], expected_network_prob[5], decimal=5
        )

    def test_create_mating_population(self):
        """
        """

        params = define_params()
        bayes_params = {'propensityweight': 0.5,
                        'unfitfraction': 0.75,
                        'crossoverprob': 0.1,
                        'mutationprob': 0.05}
        ga_calcs = run_ga_calcs({**params, **bayes_params}, test=True)
        sequences_dict = OrderedDict()
        for num in range(10):
            sequences_dict[num] = nx.MultiGraph()
        network_fitness_scores = OrderedDict({0: 0.01, 1: 0.2, 2: 0.06, 3: 0.13,
                                              4: 0.11, 5: 0.05, 6: 0.17, 7: 0.03,
                                              8: 0.1, 9: 0.14})

        # Tests creating mating population from fittest individuals (plus a
        # randomly selected sub-population of unfit individuals)
        print('Testing create_mat_pop_fittest()')

        random_nums = [1, 2, 4, 2, 3, 0]
        selected_networks = ga_calcs.create_mat_pop_fittest(
            sequences_dict, network_fitness_scores, 8,
            bayes_params['unfitfraction'], test=True, random_nums=random_nums
        )
        np.testing.assert_equal(list(selected_networks.keys()),
                                [1, 6, 3, 8, 7, 2, 0, 9])

        # Tests creating mating population using roulette wheel (where each
        # network is selected with a likelihood proportional to its fitness
        # score)
        print('Testing create_mat_pop_roulette_wheel()')

        random_nums = [0.653, 0.979, 0.116, 0.669, 0.704, 0.729]
        selected_networks = ga_calcs.create_mat_pop_roulette_wheel(
            sequences_dict, network_fitness_scores, 6,
            test=True, random_nums=random_nums
        )
        np.testing.assert_equal(list(selected_networks.keys()),
                                [6, 9, 1, 5, 4, 8])

    def test_crossover(self):
        """
        """

        test_dict = gen_sequence_networks()
        test_sub_dict = {5: copy.deepcopy(test_dict[5]),
                         6: copy.deepcopy(test_dict[6])}
        test_dict[6].nodes[1]['type'] = 'strand'
        test_dict[6].nodes[2]['type'] = 'strand'
        test_dict[6].nodes[3]['type'] = 'loop'
        sub_pairs = [(5, 6)]
        pairs = [(1, 2), (3, 4), (5, 6)]

        params = define_params()
        bayes_params = {'propensityweight': 0.5,
                        'unfitfraction': 0.75,
                        'crossoverprob': 0.1,
                        'mutationprob': 0.05}

        # Uniform crossover tests
        print('Testing uniform_crossover()')

        ga_calcs = run_ga_calcs({**params, **bayes_params}, test=True)

        # Checks networks with different strand / loop labelling of residues
        # raise an error
        with self.assertRaises(TypeError): ga_calcs.uniform_crossover(
            test_sub_dict, test=True, pairs=sub_pairs,
            crossover_prob={0: {1: 0.05, 2: 0.05, 3: 0.05}}
            )

        # Tests uniform crossover
        crossover_prob = {0: {1: 0.08, 2: 0.5, 3: 0.1},
                          1: {1: 0.04, 2: 0, 3: 0.9},
                          2: {1: 0.05, 2: 0.05, 3: 0.05}}

        uniform_crossover_dict = ga_calcs.uniform_crossover(
            test_dict, test=True, pairs=pairs, crossover_prob=crossover_prob
        )

        uniform_sequences = {}
        for i in [num for tup in pairs for num in tup]:
            uniform_seq = ''.join(
                [uniform_crossover_dict[i].nodes[x]['aa_id']
                 for x in uniform_crossover_dict[i].nodes]
            )
            uniform_sequences[i] = uniform_seq

        np.testing.assert_equal(uniform_sequences[1], 'AAR')
        np.testing.assert_equal(uniform_sequences[2], 'ANA')
        np.testing.assert_equal(uniform_sequences[3], 'NRN')
        np.testing.assert_equal(uniform_sequences[4], 'RAN')
        np.testing.assert_equal(uniform_sequences[5], 'ARN')
        np.testing.assert_equal(uniform_sequences[6], 'NAR')


        # Segmented crossover tests
        print('Testing segmented_crossover()')

        # Checks networks with different strand / loop labelling of residues
        # raise an error
        params['crossovermethod'] = 'segmented'
        params['swapstartprob'] = 0.1
        params['swapstopprob'] = 0.4
        ga_calcs = run_ga_calcs({**params, **bayes_params}, test=True)
        with self.assertRaises(TypeError): ga_calcs.segmented_crossover(
            test_sub_dict, test=True, pairs=sub_pairs,
            crossover_prob={0: {1: 0.05, 2: 0.5, 3: 0.8}}
            )

        # Tests segmented crossover
        crossover_prob = {0: {1: 0.5, 2: 0.05, 3: 0.6},
                          1: {1: 0.1, 2: 0.4, 3: 0.4},
                          2: {1: 0.05, 2: 0.5, 3: 0.8}}

        segmented_crossover_dict = ga_calcs.segmented_crossover(
            test_dict, test=True, pairs=pairs, crossover_prob=crossover_prob
        )

        segmented_sequences = {}
        for i in [num for tup in pairs for num in tup]:
            segmented_seq = ''.join(
                [segmented_crossover_dict[i].nodes[x]['aa_id']
                 for x in segmented_crossover_dict[i].nodes]
            )
            segmented_sequences[i] = segmented_seq

        np.testing.assert_equal(segmented_sequences[1], 'ANR')
        np.testing.assert_equal(segmented_sequences[2], 'AAA')
        np.testing.assert_equal(segmented_sequences[3], 'NAN')
        np.testing.assert_equal(segmented_sequences[4], 'RRN')
        np.testing.assert_equal(segmented_sequences[5], 'ARN')
        np.testing.assert_equal(segmented_sequences[6], 'NAR')

    def test_mutation(self):
        """
        """

        test_dict = gen_sequence_networks()

        params = define_params()
        bayes_params = {'propensityweight': 0.5,
                        'unfitfraction': 0.75,
                        'crossoverprob': 0.1,
                        'mutationprob': 0.2}

        mutation_prob = {1: {1: 0.2, 2: 0.5, 3: 0.3},
                         2: {1: 0.5, 2: 0.0, 3: 0.1},
                         3: {1: 0.7, 2: 0.6, 3: 0.9},
                         4: {1: 0.4, 2: 0.2, 3: 0.1},
                         5: {1: 0.2, 2: 0.1, 3: 0.1},
                         6: {1: 0.4, 2: 0.1, 3: 0.1}}

        ga_calcs = run_ga_calcs({**params, **bayes_params}, test=True)

        # Tests swap mutation
        print('Testing swap_mutate()')

        random_aas = 'BCDEFGHIJKLMOPQSTUVWXYZ'
        swap_mutation_dict = ga_calcs.swap_mutate(
            test_dict, test=True, mutation_prob=mutation_prob,
            random_aas=random_aas
        )

        for network in swap_mutation_dict.keys():
            for node in swap_mutation_dict[network].nodes:
                orig_aa_id = test_dict[network].nodes[node]['aa_id']
                new_aa_id = swap_mutation_dict[network].nodes[node]['aa_id']
                mut_prob = mutation_prob[network][node]
                if swap_mutation_dict[network].nodes[node]['type'] == 'strand':
                    if mut_prob <= 0.2:
                        self.assertNotEqual(orig_aa_id, new_aa_id)
                    else:
                        self.assertEqual(orig_aa_id, new_aa_id)
                else:
                    self.assertEqual(orig_aa_id, new_aa_id)

        # Tests scramble mutation
        mutation_prob = {1: {1: 0.2, 2: 0.5, 3: 0.3},
                         2: {1: 0.5, 2: 0.0, 3: 0.1},
                         3: {1: 0.7, 2: 0.6, 3: 0.9},
                         4: {1: 0.4, 2: 0.2, 3: 0.1},
                         5: {1: 0.2, 2: 0.1, 3: 0.1},
                         6: {1: 0.4, 2: 0.1, 3: 0.1}}

        print('Testing scramble_mutate()')
        scramble_mutation_dict = ga_calcs.scramble_mutate(
            test_dict, test=True, mutation_prob=mutation_prob
        )

        scramble_sequences = {}
        for i in list(scramble_mutation_dict.keys()):
            scramble_seq = ''.join(
                [scramble_mutation_dict[i].nodes[x]['aa_id']
                 for x in scramble_mutation_dict[i].nodes]
            )
            scramble_sequences[i] = scramble_seq

        np.testing.assert_equal(scramble_sequences[1], 'AAA')
        np.testing.assert_equal(scramble_sequences[2], 'ARN')
        np.testing.assert_equal(scramble_sequences[3], 'RAN')
        np.testing.assert_equal(scramble_sequences[4], 'NNR')
        np.testing.assert_equal(scramble_sequences[5], 'ANN')
        np.testing.assert_equal(scramble_sequences[6], 'ARR')

    def test_add_children_to_parents(self):
        """
        """

        print('Testing add_children_to_parents()')

        params = define_params()
        bayes_params = {'propensityweight': 0.5,
                        'unfitfraction': 0.75,
                        'crossoverprob': 0.1,
                        'mutationprob': 0.2}

        test_combinations = [
            (OrderedDict({1: '1', 2: '2', 3: '3'}), OrderedDict({6: '1', 5: '2', 4: '3'})),
            (OrderedDict({1: '1', 2: '3', 3: '2'}), OrderedDict({1: '1', 3: '2', 2: '3'})),
            (OrderedDict({'he': '3', 'l': '2', 'lo': '1'}), OrderedDict({'wo': '1', 'r': '2', 'ld': '3'})),
            (OrderedDict({}), OrderedDict({1: '1', 2: '2', 3: '3'})),
            (OrderedDict({}), OrderedDict({}))
        ]
        expected_combinations = [
            OrderedDict({'ra': '1', 'nd': '2', 'om': '3', 6: '1', 5: '2', 4: '3'}),
            OrderedDict({'ra': '1', 'nd': '3', 'om': '2', 1: '1', 3: '2', 2: '3'}),
            OrderedDict({'ra': '3', 'nd': '2', 'om': '1', 'wo': '1', 'r': '2', 'ld': '3'}),
            OrderedDict({1: '1', 2: '2', 3: '3'}),
            OrderedDict({})
        ]

        ga_calcs = run_ga_calcs({**params, **bayes_params}, test=True)

        for index, combination in enumerate(test_combinations):
            combined_dicts = ga_calcs.add_children_to_parents(
                combination[0], combination[1]
            )
            if index in range(3):
                exp_mut_keys = list(expected_combinations[index].keys())[0:3]
                exp_mat_keys = list(expected_combinations[index].keys())[3:6]
                act_mut_keys = list(combined_dicts.keys())[0:3]
                act_mat_keys = list(combined_dicts.keys())[3:6]
                self.assertEqual(exp_mat_keys, act_mat_keys)
                self.assertNotEqual(exp_mut_keys, act_mut_keys)
                assert(all(len(x) == 10 for x in act_mut_keys))
                self.assertEqual(list(combined_dicts.values()),
                                 list(expected_combinations[index].values()))
            elif index in range(3, 5):
                self.assertEqual(combined_dicts, expected_combinations[index])
