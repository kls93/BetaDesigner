
# python -m unittest tests/test_initial_sequence_gen.py
# Tests code for adding initial sequences by propensity

import copy
import networkx as nx
import numpy as np
import pandas as pd
import unittest
from itertools import combinations
from betadesigner.subroutines.generate_initial_sequences import (
    linear_interpolation, combine_propensities, calc_probability_distribution,
    gen_cumulative_probabilities, gen_ga_input_calcs
)


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
        # contain floats to allow inclusion of np.nan
        for index, node_val in np.ndenumerate(node_vals):
            propensity = linear_interpolation(node_val, aa_propensity_scale, '')
            actual_vals[index[0]] = propensity
        np.testing.assert_array_almost_equal(expected_vals, actual_vals, decimal=2)

    def test_calc_probability_distribution(self):
        """
        Tests calculation of probability distribution from input array of
        propensity or frequency values
        """

        prop_array = np.array([[2.5, 2.0, np.nan, np.nan],
                               [1.0, 1.0, np.nan, np.nan],
                               [0.5, 0.3, np.nan, np.nan]])
        freq_array = np.array([[np.nan, np.nan, 0.8, 0.2],
                               [np.nan, np.nan, 0.1, 0.3],
                               [np.nan, np.nan, 0.1, 0.5]])
        prop_freq_scales = {'p1': [], 'p2': [], 'f1': [], 'f2':[]}
        prop_freq_weights = {'p1': 1, 'p2': 1, 'f1': 4, 'f2':1}

        (node_indv_propensities, node_indv_frequencies, filtered_aa_list
        ) = combine_propensities(prop_array, freq_array, prop_freq_scales,
                                 prop_freq_weights, ['A', 'R', 'N'])
        np.testing.assert_array_almost_equal(
            node_indv_propensities, np.array([-1.6094379, 0., 1.8971200]), decimal=6
        )
        np.testing.assert_array_almost_equal(
            node_indv_frequencies, np.array([3.4, 0.7, 0.9]), decimal=1
        )
        np.testing.assert_equal(filtered_aa_list, np.array(['A', 'R', 'N']))

        # Test raw propensity calculation
        (node_indv_aa_index_propensity_raw, node_indv_propensities_raw,
         node_propensity_raw_probabilities) = calc_probability_distribution(
            {1: [], 2: []}, node_indv_propensities, 'propensity', 'raw'
        )
        index = node_indv_aa_index_propensity_raw.argsort()
        node_indv_aa_index_propensity_raw = node_indv_aa_index_propensity_raw[index]
        node_indv_propensities_raw = node_indv_propensities_raw[index]
        node_propensity_raw_probabilities = node_propensity_raw_probabilities[index]
        np.testing.assert_equal(
            node_indv_aa_index_propensity_raw, np.array([0, 1, 2])
        )
        np.testing.assert_array_almost_equal(
            node_indv_propensities_raw,
            np.array([-1.6094379, 0., 1.8971200]), decimal=6
        )
        np.testing.assert_array_almost_equal(
            node_propensity_raw_probabilities,
            np.array([0.81300813, 0.16260163, 0.02439024]), decimal=6)

        # Test rank propensity calculation
        (node_indv_aa_index_propensity_rank, node_indv_propensities_rank,
         node_propensity_rank_probabilities) = calc_probability_distribution(
            {1: [], 2: []}, node_indv_propensities, 'propensity', 'rank'
        )
        index = node_indv_aa_index_propensity_rank.argsort()
        node_indv_aa_index_propensity_rank = node_indv_aa_index_propensity_rank[index]
        node_indv_propensities_rank = node_indv_propensities_rank[index]
        node_propensity_rank_probabilities = node_propensity_rank_probabilities[index]
        np.testing.assert_equal(
            node_indv_aa_index_propensity_rank, np.array([0, 1, 2])
        )
        np.testing.assert_array_almost_equal(
            node_indv_propensities_rank, np.array([3., 2., 1.]), decimal=1
        )
        np.testing.assert_array_almost_equal(
            node_propensity_rank_probabilities, np.array([0.5, (1/3), (1/6)]), decimal=6
        )

        # Test frequency calculation
        (node_indv_aa_index_frequency, node_indv_frequencies,
         node_frequency_probabilities) = calc_probability_distribution(
            {1: [], 2: []}, node_indv_frequencies, 'frequency', 'raw'
        )
        index = node_indv_aa_index_frequency.argsort()
        node_indv_aa_index_frequency = node_indv_aa_index_frequency[index]
        node_indv_frequencies = node_indv_frequencies[index]
        node_frequency_probabilities = node_frequency_probabilities[index]
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

        probability_arrays = np.array([[], [1.0], [0.1, 0.2, 0.3, 0.4],
                                       [0.4, 0.5], [0.6, 0.5]])
        exp_cum_prob_vals = np.array([[], [1.0], [0.1, 0.3, 0.6, 1.0], [], []])
        exp_fail_arrays = [np.array([]), np.array([0.4, 0.5]), np.array([0.6, 0.5])]

        index = 0
        for array in probability_arrays:
            array = np.array(array)
            if any(np.array_equal(array, x) for x in exp_fail_arrays):
                self.assertRaises(Exception, gen_cumulative_probabilities, [array, ''])
            else:
                calc_array = gen_cumulative_probabilities(array, '')
                np.testing.assert_array_almost_equal(
                    calc_array, exp_cum_prob_vals[index], decimal=1
                )
            index += 1

    def test_sequence_gen(self):
        """
        Tests generation of initial sequences using propensity and/or frequency
        scales
        """

        # Generates example network of interacting amino acids
        z_dict = np.array([-26, -20, -14, -8, -2, 4, 10, 16, 22, 28])
        node_pairs = [tup for tup in combinations(range(10), 2)]

        G = nx.MultiGraph()
        for n in range(10):
            node = n

            if n % 2 == 0:
                int_or_ext = 'int'
            else:
                int_or_ext = 'ext'

            G.add_node(node, aa_id='UNK', int_ext=int_or_ext, z=z_dict[n])

        for pair in node_pairs:
            G.add_edge(pair[0], pair[1], interaction='hb')

        # Adds side chains
        #H = add_initial_side_chains_from_propensities(G, dicts, ['A', 'B', 'C'])

        return G
