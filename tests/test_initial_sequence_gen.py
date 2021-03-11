
# python -m unittest tests/test_initial_sequence_gen.py
# Tests code for adding initial sequences by propensity

import copy
import networkx as nx
import numpy as np
import pandas as pd
import unittest
from itertools import combinations
from betadesigner.subroutines.calc_propensity_in_parallel import (
    linear_interpolation, measure_fitness_propensity
)
from betadesigner.subroutines.generate_initial_sequences import (
    calc_probability_distribution, gen_cumulative_probabilities, gen_ga_input_calcs
)
from tests.test_ga import define_params, gen_prop_and_freq_distributions


def def_test_barrel_node_edge_properties():
    """
    """

    exp_node_properties = {'test_domain-4': {'type': 'loop', 'aa_id': 'A'},
                           'test_domain-3': {'type': 'loop', 'aa_id': 'R'},
                           'test_domain-2': {'type': 'loop', 'aa_id': 'N'},
                           'test_domain-1': {'type': 'loop', 'aa_id': 'R'},
                           'test_domain0': {'type': 'loop', 'aa_id': 'R'},
                           'test_domain1': {
                                'type': 'strand', 'aa_id': 'N', 'int_ext': 'int',
                                'eoc': '-', 'z': -9.5, 'phipsi': '-'
                            },
                           'test_domain2': {
                                'type': 'strand', 'aa_id': 'N', 'int_ext': 'ext',
                                'eoc': '-', 'z': -8, 'phipsi': '-'
                            },
                           'test_domain3': {
                                'type': 'strand', 'aa_id': 'N', 'int_ext': 'int',
                                'eoc': '-', 'z': -6.5, 'phipsi': '-'
                            },
                           'test_domain4': {
                                'type': 'strand', 'aa_id': 'N', 'int_ext': 'ext',
                                'eoc': '-', 'z': -5, 'phipsi': '-'
                            },
                           'test_domain5': {
                                'type': 'strand', 'aa_id': 'A', 'int_ext': 'int',
                                'eoc': '-', 'z': -2.5, 'phipsi': '-'
                            },
                           'test_domain6': {
                                'type': 'strand', 'aa_id': 'R', 'int_ext': 'ext',
                                'eoc': '-', 'z': 0, 'phipsi': '-'
                            },
                           'test_domain7': {
                                'type': 'strand', 'aa_id': 'A', 'int_ext': 'int',
                                'eoc': '-', 'z': 2.5, 'phipsi': '-'
                            },
                           'test_domain8': {
                                'type': 'strand', 'aa_id': 'A', 'int_ext': 'ext',
                                'eoc': '-', 'z': 5, 'phipsi': '-'
                            },
                           'test_domain9': {
                                'type': 'strand', 'aa_id': 'N', 'int_ext': 'int',
                                'eoc': '-', 'z': 7, 'phipsi': '-'
                            },
                           'test_domain10': {
                                'type': 'strand', 'aa_id': 'R', 'int_ext': 'ext',
                                'eoc': '-', 'z': 9.5, 'phipsi': '-'},
                           'test_domain11': {'type': 'loop', 'aa_id': 'N'},
                           'test_domain12': {'type': 'loop', 'aa_id': 'R'},
                           'test_domain13': {'type': 'loop', 'aa_id': 'A'},
                           'test_domain14': {'type': 'loop', 'aa_id': 'A'},
                           'test_domain15': {'type': 'loop', 'aa_id': 'N'}}

    exp_edge_properties = {'test_domain-4': {'test_domain4': ['vdw'],
                                             'test_domain5': ['nhb']},
                           'test_domain-3': {'test_domain3': ['vdw'],
                                             'test_domain4': ['hb'],
                                             'test_domain5': ['vdw']},
                           'test_domain-2': {'test_domain2': ['vdw'],
                                             'test_domain3': ['nhb'],
                                             'test_domain4': ['vdw']},
                           'test_domain-1': {'test_domain1': sorted(['plusminus2', 'vdw']),
                                             'test_domain2': ['hb'],
                                             'test_domain3': ['vdw']},
                           'test_domain0': {'test_domain1': sorted(['nhb', 'plusminus1']),
                                            'test_domain2': sorted(['plusminus2', 'vdw'])},
                           'test_domain1': {'test_domain10': ['hb'],
                                            'test_domain0': sorted(['nhb', 'plusminus1']),
                                            'test_domain-1': sorted(['plusminus2', 'vdw']),
                                            'test_domain2': ['plusminus1'],
                                            'test_domain3': ['plusminus2'],
                                            'test_domain9': ['vdw']},
                           'test_domain2': {'test_domain-1': ['hb'],
                                            'test_domain9': ['nhb'],
                                            'test_domain0': sorted(['plusminus2', 'vdw']),
                                            'test_domain1': ['plusminus1'],
                                            'test_domain3': ['plusminus1'],
                                            'test_domain4': ['plusminus2'],
                                            'test_domain-2': ['vdw'],
                                            'test_domain8': ['vdw'],
                                            'test_domain10': ['vdw']},
                           'test_domain3': {'test_domain8': ['hb'],
                                            'test_domain-2': ['nhb'],
                                            'test_domain1': ['plusminus2'],
                                            'test_domain2': ['plusminus1'],
                                            'test_domain4': ['plusminus1'],
                                            'test_domain5': ['plusminus2'],
                                            'test_domain-3': ['vdw'],
                                            'test_domain-1': ['vdw'],
                                            'test_domain7': ['vdw'],
                                            'test_domain9': ['vdw']},
                           'test_domain4': {'test_domain-3': ['hb'],
                                            'test_domain7': ['nhb'],
                                            'test_domain2': ['plusminus2'],
                                            'test_domain3': ['plusminus1'],
                                            'test_domain5': ['plusminus1'],
                                            'test_domain6': sorted(['plusminus2', 'vdw']),
                                            'test_domain-4': ['vdw'],
                                            'test_domain-2': ['vdw'],
                                            'test_domain8': ['vdw']},
                           'test_domain5': {'test_domain6': sorted(['hb', 'plusminus1']),
                                            'test_domain-4': ['nhb'],
                                            'test_domain3': ['plusminus2'],
                                            'test_domain4': ['plusminus1'],
                                            'test_domain7': sorted(['plusminus2', 'vdw']),
                                            'test_domain-3': ['vdw']},
                           'test_domain6': {'test_domain5': sorted(['hb', 'plusminus1']),
                                            'test_domain15': ['nhb'],
                                            'test_domain4': sorted(['plusminus2', 'vdw']),
                                            'test_domain7': ['plusminus1'],
                                            'test_domain8': ['plusminus2'],
                                            'test_domain14': ['vdw']},
                           'test_domain7': {'test_domain14': ['hb'],
                                            'test_domain4': ['nhb'],
                                            'test_domain5': sorted(['plusminus2', 'vdw']),
                                            'test_domain6': ['plusminus1'],
                                            'test_domain8': ['plusminus1'],
                                            'test_domain9': ['plusminus2'],
                                            'test_domain3': ['vdw'],
                                            'test_domain13': ['vdw'],
                                            'test_domain15': ['vdw']},
                           'test_domain8': {'test_domain3': ['hb'],
                                            'test_domain13': ['nhb'],
                                            'test_domain6': ['plusminus2'],
                                            'test_domain7': ['plusminus1'],
                                            'test_domain9': ['plusminus1'],
                                            'test_domain10': ['plusminus2'],
                                            'test_domain2': ['vdw'],
                                            'test_domain4': ['vdw'],
                                            'test_domain12': ['vdw'],
                                            'test_domain14': ['vdw']},
                           'test_domain9': {'test_domain12': ['hb'],
                                            'test_domain2': ['nhb'],
                                            'test_domain7': ['plusminus2'],
                                            'test_domain8': ['plusminus1'],
                                            'test_domain10': ['plusminus1'],
                                            'test_domain11': sorted(['plusminus2', 'vdw']),
                                            'test_domain1': ['vdw'],
                                            'test_domain3': ['vdw'],
                                            'test_domain13': ['vdw']},
                           'test_domain10': {'test_domain1': ['hb'],
                                             'test_domain11': sorted(['nhb', 'plusminus1']),
                                             'test_domain8': ['plusminus2'],
                                             'test_domain9': ['plusminus1'],
                                             'test_domain12': sorted(['plusminus2', 'vdw']),
                                             'test_domain2': ['vdw']},
                           'test_domain11': {'test_domain9': sorted(['vdw', 'plusminus2']),
                                             'test_domain10': sorted(['nhb', 'plusminus1'])},
                           'test_domain12': {'test_domain8': ['vdw'],
                                             'test_domain9': ['hb'],
                                             'test_domain10': sorted(['vdw', 'plusminus2'])},
                           'test_domain13': {'test_domain7': ['vdw'],
                                             'test_domain8': ['nhb'],
                                             'test_domain9': ['vdw']},
                           'test_domain14': {'test_domain6': ['vdw'],
                                             'test_domain7': ['hb'],
                                             'test_domain8': ['vdw']},
                           'test_domain15': {'test_domain6': ['nhb'],
                                             'test_domain7': ['vdw']}}

    return exp_node_properties, exp_edge_properties


class test_initial_sequence_generation(unittest.TestCase):
    """
    """

    def test_linear_interpolation(self):
        """
        Tests linear interpolation calculation.
        """

        aa_propensity_scale = np.array([[-20, -10, 0, 10, 20], [2.5, 2.0, 1.5, 1.0, 0.5]])
        node_vals = np.linspace(-22, 22, 45)
        expected_vals = np.array([np.nan, np.nan,
                                  2.5, 2.45, 2.4, 2.35, 2.3, 2.25, 2.2, 2.15, 2.1, 2.05,
                                  2.0, 1.95, 1.9, 1.85, 1.8, 1.75, 1.7, 1.65, 1.6, 1.55,
                                  1.5, 1.45, 1.4, 1.35, 1.3, 1.25, 1.2, 1.15, 1.1, 1.05,
                                  1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55,
                                  0.5, np.nan, np.nan])
        actual_vals = np.full(expected_vals.shape, 0.0)  # Array needs to
        # contain floats rather than ints to allow inclusion of np.nan
        for index, node_val in np.ndenumerate(node_vals):
            propensity = linear_interpolation(node_val, aa_propensity_scale, '')
            actual_vals[index[0]] = propensity
        np.testing.assert_array_almost_equal(expected_vals, actual_vals, decimal=2)

    def test_calc_probability_distribution(self):
        """
        Tests calculation of probability distribution from input array of
        propensity or frequency values
        """

        node_indv_propensities = np.array([-np.log(5), 0, -np.log(0.15)])
        node_indv_frequencies = np.array([3.4, 0.7, 0.9])

        # Test raw propensity calculation
        (node_indv_aa_index_propensity_raw, node_indv_propensities_raw,
         node_propensity_raw_probabilities) = calc_probability_distribution(
            node_indv_propensities, 'propensity', 'raw', True
        )
        np.testing.assert_equal(
            node_indv_aa_index_propensity_raw, np.array([0, 1, 2])
        )
        np.testing.assert_array_almost_equal(
            node_indv_propensities_raw, node_indv_propensities, decimal=6
        )
        np.testing.assert_array_almost_equal(
            node_propensity_raw_probabilities,
            np.array([0.81300813, 0.16260163, 0.02439024]), decimal=6
        )

        # Test rank propensity calculation
        (node_indv_aa_index_propensity_rank, node_indv_propensities_rank,
         node_propensity_rank_probabilities) = calc_probability_distribution(
            node_indv_propensities, 'propensity', 'rank', True
        )
        np.testing.assert_equal(
            node_indv_aa_index_propensity_rank, np.array([2, 1, 0])
        )
        np.testing.assert_array_almost_equal(
            node_indv_propensities_rank, np.array([1., 2., 3.]), decimal=1
        )
        np.testing.assert_array_almost_equal(
            node_propensity_rank_probabilities, np.array([(1/6), (1/3), 0.5]), decimal=6
        )

        # Test frequency calculation
        (node_indv_aa_index_frequency, node_indv_frequencies,
         node_frequency_probabilities) = calc_probability_distribution(
            node_indv_frequencies, 'frequency', 'raw', True
        )
        np.testing.assert_equal(
            node_indv_aa_index_frequency, np.array([0, 1, 2])
        )
        np.testing.assert_array_almost_equal(
            node_indv_frequencies, np.array([3.4, 0.7, 0.9]), decimal=1
        )
        np.testing.assert_array_almost_equal(
            node_frequency_probabilities, np.array([0.68, 0.14, 0.18]), decimal=2
        )

    def test_gen_cumulative_probabilities(self):
        """
        Tests calculation of cumulative probability scale
        """

        probability_arrays = [
            [], [1.0], [0.1, 0.2, 0.3, 0.4], [0.4, 0.5], [0.6, 0.5],
            [0.2, 0.3, np.nan], [0.1, 'a']
        ]
        exp_cum_prob_vals = [
            [], [1.0], [0.1, 0.3, 0.6, 1.0], [], [], [], [], []
        ]

        index = 0
        while index < 7:
            array = np.array(probability_arrays[index])
            if index in [0, 3, 4, 5, 6]:
                if index == 5:
                    self.assertRaises(
                        ValueError, gen_cumulative_probabilities, array, ''
                    )
                elif index == 6:
                    self.assertRaises(
                        TypeError, gen_cumulative_probabilities, array, ''
                    )
                else:
                    self.assertRaises(
                        Exception, gen_cumulative_probabilities, array, ''
                    )
            else:
                calc_array = gen_cumulative_probabilities(array, '')
                np.testing.assert_array_almost_equal(
                    calc_array, np.array(exp_cum_prob_vals[index]), decimal=1
                )
            index += 1

    def test_sequence_gen(self):
        """
        Tests generation of initial sequences using propensity and/or frequency
        scales
        """

        params = define_params()
        params['inputdataframepath'] = 'tests/test_files/example_barrel_input_df.pkl'
        params['inputdataframe'] = pd.read_pickle(params['inputdataframepath'])
        input_calcs = gen_ga_input_calcs(params, test=True)

        # Tests parsing of input dataframe into network
        G = input_calcs.generate_networks()

        obs_node_properties = {}
        obs_edge_properties = {}
        for node_1 in list(G.nodes()):
            obs_node_properties[node_1] = G.nodes[node_1]
            obs_edge_properties[node_1] = {}
            for edge in G.edges(node_1):
                node_2 = edge[1]
                if not node_2 in list(obs_edge_properties[node_1].keys()):
                    attr = sorted(
                        [val for num, prop_dict in dict(G[node_1][node_2]).items()
                         for key, val in prop_dict.items()]
                    )
                    obs_edge_properties[node_1][node_2] = attr

        exp_node_properties, exp_edge_properties = def_test_barrel_node_edge_properties()

        self.assertDictEqual(obs_node_properties, exp_node_properties)
        self.assertDictEqual(obs_edge_properties, exp_edge_properties)

        # Tests that input graph is scored correctly based upon its properties
        prop_freq_dicts_1 = gen_prop_and_freq_distributions()
        prop_freq_dicts_1 = [[key, 1, val] for key, val in prop_freq_dicts_1.items()]
        num_1, obs_prop_count_1, obs_freq_count_1 = measure_fitness_propensity(
            1, G, prop_freq_dicts_1, params['dictnameindices'],
            params['barrelorsandwich'], params['aacodes']
        )

        exp_prop_count_1 = 5.7992442
        exp_freq_count_1 = 20.2111111

        np.testing.assert_equal(num_1, 1)
        np.testing.assert_almost_equal(obs_prop_count_1, exp_prop_count_1, decimal=6)
        np.testing.assert_almost_equal(obs_freq_count_1, exp_freq_count_1, decimal=6)

        # Tests that np.nan is handled properly in propensity dicts when scoring
        prop_freq_dicts_2 = gen_prop_and_freq_distributions()
        prop_freq_dicts_2['-_-_z_-_indv_cont_propensity'] = {
            'A': np.array([[-10, 0, 10], [0.2, 0.2, 0.8]]),
            'R': np.array([[-10, 0, 10], [1.1, 0.9, 1.2]]),
            'N': np.array([[-10, 0, 10], [np.nan, 0.7, 0.8]])
        }
        prop_freq_dicts_2 = [[key, 1, val] for key, val in prop_freq_dicts_2.items()]
        num_2, obs_prop_count_2, obs_freq_count_2 = measure_fitness_propensity(
            2, G, prop_freq_dicts_2, params['dictnameindices'],
            params['barrelorsandwich'], params['aacodes']
        )

        exp_prop_count_2 = 39.0194597
        exp_freq_count_2 = 20.2111111

        np.testing.assert_equal(num_2, 2)
        np.testing.assert_almost_equal(obs_prop_count_2, exp_prop_count_2, decimal=6)
        np.testing.assert_almost_equal(obs_freq_count_2, exp_freq_count_2, decimal=6)

        # Tests that np.nan is handled properly in frequency dicts when scoring
        prop_freq_dicts_3 = gen_prop_and_freq_distributions()
        prop_freq_dicts_3['int_-_-_-_indv_disc_frequency'] = pd.DataFrame(
            {'FASTA': ['A', 'R', 'N'],
             'int': [np.nan, 0.15, 0.05]
        })
        prop_freq_dicts_3 = [[key, 1, val] for key, val in prop_freq_dicts_3.items()]
        num_3, obs_prop_count_3, obs_freq_count_3 = measure_fitness_propensity(
            3, G, prop_freq_dicts_3, params['dictnameindices'],
            params['barrelorsandwich'], params['aacodes']
        )

        exp_prop_count_3 = 5.7992442
        exp_freq_count_3 = 4018.7111111

        np.testing.assert_equal(num_3, 3)
        np.testing.assert_almost_equal(obs_prop_count_3, exp_prop_count_3, decimal=6)
        np.testing.assert_almost_equal(obs_freq_count_3, exp_freq_count_3, decimal=6)

    def test_add_random_initial_side_chains(self):
        """
        Tests that aa_ids of non-strand residues are not varied when adding
        initial side chains at random
        """

        params = define_params()
        params['inputdataframepath'] = 'tests/test_files/example_barrel_input_df.pkl'
        params['inputdataframe'] = pd.read_pickle(params['inputdataframepath'])
        input_calcs = gen_ga_input_calcs(params, test=True)

        # Tests parsing of input dataframe into network
        G = input_calcs.generate_networks()
        input_aa = {'test_domain-4': 'N',
                    'test_domain-3': 'A',
                    'test_domain-2': 'R',
                    'test_domain-1': 'R',
                    'test_domain0': 'R',
                    'test_domain1': 'R',
                    'test_domain2': 'N',
                    'test_domain3': 'N',
                    'test_domain4': 'A',
                    'test_domain5': 'A',
                    'test_domain6': 'N',
                    'test_domain7': 'A',
                    'test_domain8': 'R',
                    'test_domain9': 'R',
                    'test_domain10': 'N',
                    'test_domain11': 'A',
                    'test_domain12': 'N',
                    'test_domain13': 'N',
                    'test_domain14': 'R',
                    'test_domain15': 'R'}
        networks_dict = input_calcs.add_random_initial_side_chains(G, True, input_aa)
        self.assertTrue(len(networks_dict) == (params['populationsize']*2))

        for num, H in networks_dict.items():
          for node in list(H.nodes()):
              if H.nodes()[node]['type'] == 'loop':
                  self.assertEqual(H.nodes()[node]['aa_id'], G.nodes()[node]['aa_id'])
              else:
                  self.assertTrue(H.nodes()[node]['aa_id'] == input_aa[node])

    def test_add_initial_side_chains_from_propensities(self):
        """
        Tests that aa_ids of non-strand residues are not varied when adding
        initial side chains using propensity scores
        """

        # Tests propensity and frequency scoring
        params_1 = define_params()
        params_1['inputdataframepath'] = 'tests/test_files/example_barrel_input_df.pkl'
        params_1['inputdataframe'] = pd.read_pickle(params_1['inputdataframepath'])
        input_calcs_1 = gen_ga_input_calcs(params_1, test=True)

        G_1 = input_calcs_1.generate_networks()
        networks_dict_1 = input_calcs_1.add_initial_side_chains_from_propensities(
            G_1, 'raw', test=True, input_num={'test_domain-4': 0.633,
                                              'test_domain-3': 0.164,
                                              'test_domain-2': 0.887,
                                              'test_domain-1': 0.636,
                                              'test_domain0': 0.284,
                                              'test_domain1': 0.788,
                                              'test_domain2': 0.458,
                                              'test_domain3': 0.516,
                                              'test_domain4': 0.628,
                                              'test_domain5': 0.933,
                                              'test_domain6': 0.071,
                                              'test_domain7': 0.871,
                                              'test_domain8': 0.357,
                                              'test_domain9': 0.050,
                                              'test_domain10': 0.998,
                                              'test_domain11': 0.861,
                                              'test_domain12': 0.213,
                                              'test_domain13': 0.754,
                                              'test_domain14': 0.546,
                                              'test_domain15': 0.999}
        )
        self.assertTrue(len(networks_dict_1) == (params_1['populationsize']*2))

        exp_aa_ids_1 = {'test_domain-4': 'A',
                        'test_domain-3': 'R',
                        'test_domain-2': 'N',
                        'test_domain-1': 'R',
                        'test_domain0': 'R',
                        'test_domain1': 'R',
                        'test_domain2': 'R',
                        'test_domain3': 'A',
                        'test_domain4': 'N',
                        'test_domain5': 'N',
                        'test_domain6': 'A',
                        'test_domain7': 'N',
                        'test_domain8': 'A',
                        'test_domain9': 'A',
                        'test_domain10': 'N',
                        'test_domain11': 'N',
                        'test_domain12': 'R',
                        'test_domain13': 'A',
                        'test_domain14': 'A',
                        'test_domain15': 'N'}
        for num, H in networks_dict_1.items():
          for node in list(H.nodes()):
              if H.nodes()[node]['type'] == 'loop':
                  self.assertEqual(H.nodes()[node]['aa_id'], G_1.nodes()[node]['aa_id'])
              else:
                  self.assertTrue(H.nodes()[node]['aa_id'] == exp_aa_ids_1[node])

        # Tests propensity and frequency scoring with np.nan scores
        params_2 = define_params()
        params_2['propensityscales']['int_-_z_-_indv_cont_propensity'] = {
            'A': np.array([[-10, 0, 10], [2.5, 1.5, 0.5]]),
            'R': np.array([[-10, 0, 10], [0.5, 1.5, np.nan]]),
            'N': np.array([[-10, 0, 10], [0.5, 2.5, 0.5]])
        }
        params_2['frequencyscales']['ext_-_-_-_indv_disc_frequency'] = pd.DataFrame(
            {'FASTA': ['A', 'R', 'N'],
             'ext': [np.nan, 0.8, 0.2]
        })
        params_2['inputdataframepath'] = 'tests/test_files/example_barrel_input_df.pkl'
        params_2['inputdataframe'] = pd.read_pickle(params_2['inputdataframepath'])
        input_calcs_2 = gen_ga_input_calcs(params_2, test=True)

        G_2 = input_calcs_2.generate_networks()
        networks_dict_2 = input_calcs_2.add_initial_side_chains_from_propensities(
            G_2, 'raw', test=True, input_num={'test_domain-4': 0.633,
                                              'test_domain-3': 0.164,
                                              'test_domain-2': 0.887,
                                              'test_domain-1': 0.636,
                                              'test_domain0': 0.284,
                                              'test_domain1': 0.788,
                                              'test_domain2': 0.060,
                                              'test_domain3': 0.852,
                                              'test_domain4': 0.059,
                                              'test_domain5': 0.933,
                                              'test_domain6': 0.071,
                                              'test_domain7': 0.591,
                                              'test_domain8': 0.100,
                                              'test_domain9': 0.585,
                                              'test_domain10': 0.998,
                                              'test_domain11': 0.861,
                                              'test_domain12': 0.213,
                                              'test_domain13': 0.754,
                                              'test_domain14': 0.546,
                                              'test_domain15': 0.999}
        )
        self.assertTrue(len(networks_dict_2) == (params_2['populationsize']*2))

        exp_aa_ids_2 = {'test_domain-4': 'A',
                        'test_domain-3': 'R',
                        'test_domain-2': 'N',
                        'test_domain-1': 'R',
                        'test_domain0': 'R',
                        'test_domain1': 'R',
                        'test_domain2': 'A',
                        'test_domain3': 'N',
                        'test_domain4': 'R',
                        'test_domain5': 'N',
                        'test_domain6': 'R',
                        'test_domain7': 'N',
                        'test_domain8': 'A',
                        'test_domain9': 'R',
                        'test_domain10': 'N',
                        'test_domain11': 'N',
                        'test_domain12': 'R',
                        'test_domain13': 'A',
                        'test_domain14': 'A',
                        'test_domain15': 'N'}
        for num, H in networks_dict_2.items():
          for node in list(H.nodes()):
              if H.nodes()[node]['type'] == 'loop':
                  self.assertEqual(H.nodes()[node]['aa_id'], G_2.nodes()[node]['aa_id'])
              else:
                  self.assertTrue(H.nodes()[node]['aa_id'] == exp_aa_ids_2[node])
