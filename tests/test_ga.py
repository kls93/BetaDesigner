
import copy
import networkx as nx
import numpy as np
import pandas as pd
import unittest
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

    sequences_dict = {'int': {1: network_1,
                              2: network_2}}

    return sequences_dict


class testGeneticAlgorithm(unittest.TestCase):
    """
    """

    def test_measure_fitness_propensity(self):
        """
        """

        params = define_params()
        bayes_params = {'propvsfreqweight': {'propensity': 0.5,
                                             'frequency': 0.5},
                        'unfitfraction': 0.2,
                        'crossoverprob': 0.1,
                        'mutationprob': 0.05}
        sequences_dict = gen_sequence_networks()

        ga_calcs = run_ga_calcs({**params, **bayes_params}, test=True)
        network_propensities, network_frequencies = ga_calcs.measure_fitness_propensity(
            'int', sequences_dict['int']
        )

        np.testing.assert_almost_equal(network_propensities[1], -0.887891, decimal=5)
        np.testing.assert_almost_equal(network_propensities[2], -3.222491, decimal=5)
        np.testing.assert_almost_equal(network_frequencies[1], 2.4, decimal=5)
        np.testing.assert_almost_equal(network_frequencies[2], 1, decimal=5)

        self.network_propensities = network_propensities
        self.network_frequencies = network_frequencies

    def test_combine_prop_and_freq_scores():
        """
        """

        params = define_params()
        bayes_params = {'propvsfreqweight': {'propensity': 0.5,
                                             'frequency': 0.5},
                        'unfitfraction': 0.2,
                        'crossoverprob': 0.1,
                        'mutationprob': 0.05}
        sequences_dict = gen_sequence_networks()

        ga_calcs = run_ga_calcs({**params, **bayes_params}, test=True)




    def test_measure_fitness_all_atom_scoring_function(
        self, network_propensities, network_frequencies
    ):
        """
        ** Can't test since side-chain packing requires SCWRL4, and can't load
        SCWRL4 into circleci without a license
        """

        return

    def test_create_mating_population_fittest_indv(self):
        """
        """

        params = define_params()
        bayes_params = {'propvsfreqweight': {'propensity': 0.5,
                                             'frequency': 0.5},
                        'unfitfraction': 0.2,
                        'crossoverprob': 0.1,
                        'mutationprob': 0.05}
        sequences_dict = gen_sequence_networks()

        ga_calcs = run_ga_calcs({**params, **bayes_params}, test=True)
        ga_calcs.create_mat_pop_fittest()

        return

    def test_create_mating_population_roulette_wheel(self):
        """
        """

        return

    def test_uniform_crossover(self):
        """
        """

        return

    def test_segmented_crossover(self):
        """
        """

        return

    def test_swap_mutate(self):
        """
        """

        return

    def test_scramble_mutate(self):
        """
        """

        return

    def test_add_children_to_parents(self):
        """
        """

        return
