
import copy
import networkx as nx
import numpy as np
import pandas as pd
import unittest
from collections import OrderedDict
from betadesigner.subroutines.find_parameters import initialise_ga_object
from betadesigner.subroutines.run_genetic_algorithm import run_ga_calcs


def define_params():
    """
    """

    prop_freq_dicts = gen_prop_and_freq_distributions()

    params = {'inputdataframe': '',
              'inputpdb': '',
              'propensityscales': {key: val for key, val in prop_freq_dicts.items()
                                   if 'propensity' in key},
              'frequencyscales': {key: val for key, val in prop_freq_dicts.items()
                                  if 'frequency' in key},
              'aacodes': ['A', 'B', 'C'],
              'scaleweights': {key: 1 for key in prop_freq_dicts.keys()},
              'dictnameindices': {'intorext': 0,
                                  'edgeorcent': 1,
                                  'prop1': 2,
                                  'interactiontype': 3,
                                  'pairorindv': 4,
                                  'discorcont': 5,
                                  'proporfreq': 6},
              'workingdirectory': '',
              'barrelorsandwich': '2.40',
              'jobid': '',
              'initialseqmethod': '',
              'fitnessscoremethod': 'split',
              'splitfraction': 0.5,
              'matingpopmethod': '',
              'crossovermethod': '',
              'swapstartprob': '',
              'swapstopprob': '',
              'mutationmethod': '',
              'populationsize': 2,
              'maxnumgenerations': 10}

    return params


def gen_prop_and_freq_distributions():
    """
    Defines example propensity and frequency dictionaries
    """

    dicts = {'int_-_z_-_indv_cont_propensity': {'A': np.array([[-10, 0, 10], [2.5, 1.5, 0.5]]),
                                                'B': np.array([[-10, 0, 10], [0.5, 1.5, 2.5]]),
                                                'C': np.array([[-10, 0, 10], [0.5, 2.5, 0.5]])},
             'int_-_z_hb_pair_cont_propensity': {'A_A': np.array([[-10, 0, 10], [0.5, 1.0, 1.5]]),
                                                 'A_B': np.array([[-10, 0, 10], [0.1, 2.0, 0.1]]),
                                                 'A_C': np.array([[-10, 0, 10], [2.0, 1.2, 0.4]]),
                                                 'B_A': np.array([[-10, 0, 10], [0.1, 2.0, 0.1]]),
                                                 'B_B': np.array([[-10, 0, 10], [2.0, 0.5, 1.8]]),
                                                 'B_C': np.array([[-10, 0, 10], [1.5, 1.2, 0.9]]),
                                                 'C_A': np.array([[-10, 0, 10], [2.0, 1.2, 0.4]]),
                                                 'C_B': np.array([[-10, 0, 10], [1.5, 1.2, 0.9]]),
                                                 'C_C': np.array([[-10, 0, 10], [0.6, 0.3, 1.7]])},
             'int_-_-_-_indv_disc_frequency': pd.DataFrame({'FASTA': ['A', 'B', 'C'],
                                                            'int': [0.8, 0.15, 0.05]})}

    return dicts


def gen_sequence_networks():
    """
    """

    network_1 = nx.MultiGraph()
    network_1.add_node(1, aa_id='A', int_ext='int', z=-5)
    network_1.add_node(2, aa_id='A', int_ext='int', z=0)
    network_1.add_node(3, aa_id='A', int_ext='int', z=7)
    network_1.add_edge(1, 2, interaction='hb')
    network_1.add_edge(2, 3, interaction='hb')

    network_2 = nx.MultiGraph()
    network_2.add_node(1, aa_id='A', int_ext='int', z=-5)
    network_2.add_node(2, aa_id='C', int_ext='int', z=0)
    network_2.add_node(3, aa_id='B', int_ext='int', z=7)
    network_2.add_edge(1, 2, interaction='hb')
    network_2.add_edge(2, 3, interaction='hb')

    network_3 = nx.MultiGraph()
    network_3.add_node(1, aa_id='B', int_ext='int', z=-5)
    network_3.add_node(2, aa_id='A', int_ext='int', z=0)
    network_3.add_node(3, aa_id='C', int_ext='int', z=7)
    network_3.add_edge(1, 2, interaction='hb')
    network_3.add_edge(2, 3, interaction='hb')

    network_4 = nx.MultiGraph()
    network_4.add_node(1, aa_id='C', int_ext='int', z=-5)
    network_4.add_node(2, aa_id='B', int_ext='int', z=0)
    network_4.add_node(3, aa_id='C', int_ext='int', z=7)
    network_4.add_edge(1, 2, interaction='hb')
    network_4.add_edge(2, 3, interaction='hb')

    fit_test_dict = {'int': {1: network_1,
                             2: network_2}}

    cross_test_dict = {'int': {1: network_1,
                               2: network_2,
                               3: network_3,
                               4: network_4}}

    return fit_test_dict, cross_test_dict


class testGeneticAlgorithm(unittest.TestCase):
    """
    """

    def gen_network_prop_and_freq(self):
        """
        """

        params = define_params()
        bayes_params = {'propvsfreqweight': {'propensity': 0.5,
                                             'frequency': 0.5},
                        'unfitfraction': 0.2,
                        'crossoverprob': 0.1,
                        'mutationprob': 0.05}
        fit_test_dict, cross_test_dict = gen_sequence_networks()

        ga_calcs = run_ga_calcs({**params, **bayes_params}, test=True)
        network_propensities, network_frequencies = ga_calcs.measure_fitness_propensity(
            'int', fit_test_dict['int']
        )

        return network_propensities, network_frequencies

    def test_measure_fitness_propensity(self):
        """
        """

        print('Testing measure_fitness_propensity()')

        network_propensities, network_frequencies = self.gen_network_prop_and_freq()

        np.testing.assert_almost_equal(network_propensities[1], -0.887891, decimal=5)
        np.testing.assert_almost_equal(network_propensities[2], -3.222491, decimal=5)
        np.testing.assert_almost_equal(network_frequencies[1], 2.4, decimal=5)
        np.testing.assert_almost_equal(network_frequencies[2], 1, decimal=5)

    def test_combine_prop_and_freq_scores(self):
        """
        """

        print('Testing combine_prop_and_freq_scores()')

        params = define_params()
        network_propensities, network_frequencies = self.gen_network_prop_and_freq()

        # Test 1
        bayes_params = {'propvsfreqweight': {'propensity': 0.5,
                                             'frequency': 0.5},
                        'unfitfraction': 0.2,
                        'crossoverprob': 0.1,
                        'mutationprob': 0.05}
        ga_calcs = run_ga_calcs({**params, **bayes_params}, test=True)

        network_fitness_scores = ga_calcs.combine_prop_and_freq_scores(
            network_propensities, network_frequencies, 'raw'
        )
        np.testing.assert_almost_equal(network_fitness_scores[1], 0.397089, decimal=5)
        np.testing.assert_almost_equal(network_fitness_scores[2], 0.602910, decimal=5)

        # Test 2
        bayes_params = {'propvsfreqweight': {'propensity': 0.8,
                                             'frequency': 0.2},
                        'unfitfraction': 0.2,
                        'crossoverprob': 0.1,
                        'mutationprob': 0.05}
        ga_calcs = run_ga_calcs({**params, **bayes_params}, test=True)
        network_fitness_scores = ga_calcs.combine_prop_and_freq_scores(
            network_propensities, network_frequencies, 'raw'
        )
        np.testing.assert_almost_equal(network_fitness_scores[1], 0.211814, decimal=5)
        np.testing.assert_almost_equal(network_fitness_scores[2], 0.788185, decimal=5)

        # Test 3
        bayes_params = {'propvsfreqweight': {'propensity': 1,
                                             'frequency': 0},
                        'unfitfraction': 0.2,
                        'crossoverprob': 0.1,
                        'mutationprob': 0.05}
        ga_calcs = run_ga_calcs({**params, **bayes_params}, test=True)
        network_fitness_scores = ga_calcs.combine_prop_and_freq_scores(
            network_propensities, network_frequencies, 'raw'
        )
        np.testing.assert_almost_equal(network_fitness_scores[1], 0.088297, decimal=5)
        np.testing.assert_almost_equal(network_fitness_scores[2], 0.911702, decimal=5)

        # Test 4
        bayes_params = {'propvsfreqweight': {'propensity': 0.6,
                                             'frequency': 0.4},
                        'unfitfraction': 0.2,
                        'crossoverprob': 0.1,
                        'mutationprob': 0.05}
        ga_calcs = run_ga_calcs({**params, **bayes_params}, test=True)
        network_fitness_scores = ga_calcs.combine_prop_and_freq_scores(
            network_propensities, network_frequencies, 'rank'
        )
        np.testing.assert_almost_equal(network_fitness_scores[1], 0.482352, decimal=5)
        np.testing.assert_almost_equal(network_fitness_scores[2], 0.517647, decimal=5)

    def test_measure_fitness_bude_and_convert_to_probability(self):
        """
        ** Can't test since side-chain packing requires SCWRL4, and can't load
        SCWRL4 into circleci without a license
        """

        print('Testing measure_fitness_allatom()')

    def test_create_mating_population(self):
        """
        """

        params = define_params()
        bayes_params = {'propvsfreqweight': {'propensity': 0.5,
                                             'frequency': 0.5},
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
            'int', sequences_dict, network_fitness_scores, 8,
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
            'int', sequences_dict, network_fitness_scores, 6,
            test=True, random_nums=random_nums
        )
        np.testing.assert_equal(list(selected_networks.keys()),
                                [6, 9, 1, 5, 4, 8])

    def test_crossover(self):
        """
        """

        fit_test_dict, cross_test_dict = gen_sequence_networks()

        params = define_params()
        bayes_params = {'propvsfreqweight': {'propensity': 0.5,
                                             'frequency': 0.5},
                        'unfitfraction': 0.75,
                        'crossoverprob': 0.1,
                        'mutationprob': 0.05}

        # Tests uniform crossover
        print('Testing uniform_crossover()')

        ga_calcs = run_ga_calcs({**params, **bayes_params}, test=True)

        pairs = [(1, 2), (3, 4)]
        crossover_prob = {0: {1: 0.08, 2: 0.5, 3: 0.1},
                          1: {1: 0.04, 2: 0, 3: 0.9}}

        uniform_crossover_dict = ga_calcs.uniform_crossover(
            'int', cross_test_dict['int'], test=True, pairs=pairs,
            crossover_prob=crossover_prob
        )

        uniform_sequences = {}
        for i in [num for tup in pairs for num in tup]:
            uniform_seq = ''.join(
                [uniform_crossover_dict[i].nodes[x]['aa_id']
                 for x in uniform_crossover_dict[i].nodes]
            )
            uniform_sequences[i] = uniform_seq

        np.testing.assert_equal(uniform_sequences[1], 'AAB')
        np.testing.assert_equal(uniform_sequences[2], 'ACA')
        np.testing.assert_equal(uniform_sequences[3], 'CBC')
        np.testing.assert_equal(uniform_sequences[4], 'BAC')

        # Tests segmented crossover
        print('Testing segmented_crossover()')

        params['crossovermethod'] = 'segmented'
        params['swapstartprob'] = 0.1
        params['swapstopprob'] = 0.4
        ga_calcs = run_ga_calcs({**params, **bayes_params}, test=True)

        pairs = [(1, 2), (3, 4)]
        crossover_prob = {0: {1: 0.5, 2: 0.05, 3: 0.6},
                          1: {1: 0.1, 2: 0.4, 3: 0.4}}

        segmented_crossover_dict = ga_calcs.segmented_crossover(
            'int', cross_test_dict['int'], test=True, pairs=pairs,
            crossover_prob=crossover_prob
        )

        segmented_sequences = {}
        for i in [num for tup in pairs for num in tup]:
            segmented_seq = ''.join(
                [segmented_crossover_dict[i].nodes[x]['aa_id']
                 for x in segmented_crossover_dict[i].nodes]
            )
            segmented_sequences[i] = segmented_seq

        np.testing.assert_equal(segmented_sequences[1], 'ACB')
        np.testing.assert_equal(segmented_sequences[2], 'AAA')
        np.testing.assert_equal(segmented_sequences[3], 'CAC')
        np.testing.assert_equal(segmented_sequences[4], 'BBC')

    def test_mutation(self):
        """
        """

        fit_test_dict, mut_test_dict = gen_sequence_networks()

        params = define_params()
        bayes_params = {'propvsfreqweight': {'propensity': 0.5,
                                             'frequency': 0.5},
                        'unfitfraction': 0.75,
                        'crossoverprob': 0.1,
                        'mutationprob': 0.2}

        mutation_prob = {1: {1: 0.2, 2: 0.5, 3: 0.3},
                         2: {1: 0.5, 2: 0.0, 3: 0.1},
                         3: {1: 0.7, 2: 0.6, 3: 0.9},
                         4: {1: 0.4, 2: 0.2, 3: 0.1}}

        ga_calcs = run_ga_calcs({**params, **bayes_params}, test=True)

        # Tests swap mutation
        print('Testing swap_mutate()')

        random_aas = 'DEFGHIJKLMNOPQRSTUVWXYZ'
        swap_mutation_dict = ga_calcs.swap_mutate(
            'int', mut_test_dict['int'], test=True, mutation_prob=mutation_prob,
            random_aas=random_aas
        )

        for network in swap_mutation_dict.keys():
            for node in swap_mutation_dict[network].nodes:
                orig_aa_id = mut_test_dict['int'][network].nodes[node]['aa_id']
                new_aa_id = swap_mutation_dict[network].nodes[node]['aa_id']
                mut_prob = mutation_prob[network][node]
                if mut_prob <= 0.2:
                    self.assertNotEqual(orig_aa_id, new_aa_id)
                else:
                    self.assertEqual(orig_aa_id, new_aa_id)

        # Tests scramble mutation
        print('Testing scramble_mutate()')
        scramble_mutation_dict = ga_calcs.scramble_mutate(
            'int', mut_test_dict['int'], test=True, mutation_prob=mutation_prob
        )

        for network in scramble_mutation_dict.keys():
            scrambled_aa_ids = []

            for n in [0, 1]:
                scrambled_aa_ids = scrambled_aa_ids[::-1]
                count = 0

                for index, node in enumerate(list(scramble_mutation_dict[network].nodes)):
                    orig_aa_id = mut_test_dict['int'][network].nodes[node]['aa_id']
                    new_aa_id = scramble_mutation_dict[network].nodes[node]['aa_id']
                    mut_prob = mutation_prob[network][node]

                    if mut_prob <= 0.2:
                        if n == 0:
                            scrambled_aa_ids.append(orig_aa_id)
                        if n == 1:
                            exp_aa_id = scrambled_aa_ids[count]
                            count += 1
                            self.assertEqual(exp_aa_id, new_aa_id)

                    else:
                        if n == 1:
                            self.assertEqual(orig_aa_id, new_aa_id)

    def test_add_children_to_parents(self):
        """
        """

        print('Testing add_children_to_parents()')

        params = define_params()
        bayes_params = {'propvsfreqweight': {'propensity': 0.5,
                                             'frequency': 0.5},
                        'unfitfraction': 0.75,
                        'crossoverprob': 0.1,
                        'mutationprob': 0.2}

        test_combinations = [
            (OrderedDict({1: '1', 2: '2', 3: '3'}), OrderedDict({1: '1', 2: '2', 3: '3'}), 0),
            (OrderedDict({1: '1', 2: '2', 3: '3'}), OrderedDict({1: '1', 2: '2', 3: '3'}), 1),
            (OrderedDict({2: '3', 1: '2', 3: '1'}), OrderedDict({3: '1', 1: '2', 2: '3'}), 1),
            (OrderedDict({}), OrderedDict({1: '1', 2: '2', 3: '3'}), 0),
            (OrderedDict({}), OrderedDict({}), 1)
        ]
        expected_combinations = [
            OrderedDict({0: '1', 1: '2', 2: '3', 3: '1', 4: '2', 5: '3'}),
            OrderedDict({2: '1', 3: '2', 4: '3', 5: '1', 6: '2', 7: '3'}),
            OrderedDict({2: '3', 3: '2', 4: '1', 5: '1', 6: '2', 7: '3'}),
            OrderedDict({0: '1', 1: '2', 2: '3'}),
            OrderedDict({})
        ]

        ga_calcs = run_ga_calcs({**params, **bayes_params}, test=True)

        for index, combination in enumerate(test_combinations):
            combined_dicts = ga_calcs.add_children_to_parents(
                'int', combination[0], combination[1], combination[2]
            )
            self.assertEqual(combined_dicts, expected_combinations[index])
