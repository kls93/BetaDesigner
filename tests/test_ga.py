
import unittest
import numpy as np
from betadesigner.subroutines.find_parameters import find_parameters
from betadesigner.subroutines.generate_initial_sequences import
from betadesigner.subroutines.run_genetic_algorithm import
from betadesigner.subroutines.write_output_structures import

class testGeneticAlgorithm(unittest.TestCase):
    """
    """

    def test():
        return

class TestGeneticAlgorithm(unittest.TestCase):

    def setUp(self):
        # OVERWRITE ONCE HAVE COMPLETED GENERATION OF PROPENSITY SCALES FROM
        # BETASTATS.

        self.propensity_dicts = OrderedDict({'int_z_indv': {'ARG': np.array([[-50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50], [1.3, 1.3, 1.7, 1.8, 1.9, 2.0, 1.9, 1.8, 1.7, 1.3, 1.3]]),
                                                            'TRP': np.array([[-50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50], [3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0]]),
                                                            'VAL': np.array([[-50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50], [0.7, 0.7, 0.5, 0.5, 0.3, 0.2, 0.3, 0.5, 0.5, 0.7, 0.7]])},
                                             'ext_z_indv': {'ARG': np.array([[-50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50], [1.5, 1.5, 0.6, 0.4, 0.3, 0.2, 0.3, 0.4, 0.6, 1.5, 1.5]]),
                                                            'TRP': np.array([[-50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50], [3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0]]),
                                                            'VAL': np.array([[-50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50], [0.8, 0.8, 1.3, 1.5, 1.7, 1.7, 1.7, 1.5, 1.3, 0.8, 0.8]])}
                                           })
        self.aas = list(self.propensity_dicts['int_z_indv'].keys())
        self.propensity_dict_weights = OrderedDict({'int_z_indv': 1,
                                                    'ext_z_indv': 1})

    def test_pack_side_chains(self):

    def test_measure_fitness_propensity(self):

    def test_measure_fitness_all_atom_scoring_function(self):

    def test_create_mating_population_fittest_indv(self):

    def test_create_mating_population_roulette_wheel(self):

    def test_uniform_crossover(self):

    def test_segmented_crossover(self):

    def test_swap_mutate(self):

    def test_scramble_mutate(self):

    def test_add_children_to_parents(self):
