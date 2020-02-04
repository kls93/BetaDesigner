
# python -m unittest tests/test_parameter_parsing.py

import numpy as np
import pandas as pd
import pickle
import unittest
from collections import OrderedDict

from betadesigner.subroutines.find_parameters import (
    calc_parent_voronoi_cluster, def_input_df_path, def_input_pdb,
    def_propensity_scales, def_frequency_scales, convert_str_to_dict,
    def_prop_freq_scale_weights, def_propensity_weight,
    def_phipsi_cluster_coords, def_working_directory, def_barrel_or_sandwich,
    def_jobid, def_method_initial_seq, def_method_fitness_scoring,
    def_split_fraction, def_method_select_mating_pop, def_unfit_fraction,
    def_method_crossover, def_crossover_prob, def_swap_start_prob,
    def_swap_stop_prob, def_method_mutation, def_mutation_prob, def_pop_size,
    def_num_gens
)


class testParameterParsing(unittest.TestCase):
    """
    Tests parsing of input parameters
    """

    def test_calc_phi_psi_voronoi_cluster(self):
        """
        Tests that input dataframe is corretly updated to classify residues
        according to the voronoi cluster of their phi and psi angles
        """

        input_df = pd.DataFrame({
            'int_ext': ['int', 'ext', 'int', 'int', 'ext', 'int'],
            'phi': [-10, 0, -9, 1, -12, 2],
            'psi': [-10, 2, -7, 0, -7, 1]
        })
        cluster_coords = {'int': np.array([[-11, -9], [1, 1]]),
                          'ext': np.array([[-11, -10], [0, 1]])}

        exp_output = pd.DataFrame({
            'int_ext': ['int', 'ext', 'int', 'int', 'ext', 'int'],
            'phi': [-10, 0, -9, 1, -12, 2],
            'psi': [-10, 2, -7, 0, -7, 1],
            'phi_psi_class': [0, 1, 0, 1, 0, 1]
        })
        act_output = calc_parent_voronoi_cluster(input_df, cluster_coords)

        pd.testing.assert_frame_equal(exp_output, act_output)

    def test_input_df(self):
        """
        Tests that input dataframe path is parsed correctly
        """

        expected_vals = [
            [{},
             'File path to pickled input dataframe not recognised'],
            [{'inputdataframepath': 'mistake'},
             'File path to pickled input dataframe not recognised'],
            [{'inputdataframepath': 'tests/test_files/mistake.pkl'},
             'File path to pickled input dataframe not recognised'],
            [{'inputdataframepath': 'tests/test_files/example_pdb_253L.pdb'},
             'File path to pickled input dataframe not recognised'],
            [{'inputdataframepath': 'tests/test_files/example_prop_freq_scales.pkl'},
             'Input file is not a dataframe'],
            [{'inputdataframepath': 'tests/test_files/example_input_df.pkl'},
             'tests/test_files/example_input_df.pkl']
        ]

        for pair in expected_vals:
            params = pair[0]
            exp_output = pair[1]
            self.assertEqual(def_input_df_path(params, test=True), exp_output)

    def test_input_pdb(self):
        """
        Tests that input PDB file path is parsed correctly
        """

        expected_vals = [
            [{},
             'File path to input PDB file not recognised'],
            [{'inputpdb': 'mistake'},
             'File path to input PDB file not recognised'],
            [{'inputpdb': 'tests/test_files/mistake.pdb'},
             'File path to input PDB file not recognised'],
            [{'inputpdb': 'tests/test_files/example_prop_freq_scales.pkl'},
             'File path to input PDB file not recognised'],
            [{'inputpdb': 'tests/test_files/example_pdb_253L.pdb'},
             'tests/test_files/example_pdb_253L.pdb']
        ]

        for pair in expected_vals:
            params = pair[0]
            exp_output = pair[1]
            self.assertEqual(def_input_pdb(params, test=True), exp_output)

    def test_propensity_dicts(self):
        """
        Tests that path to pickled propensity dictionaries is parsed correctly,
        and that the dataframe is unpickled
        """

        expected_vals = [
            [{},
             'File path to pickled propensity scales not recognised'],
            [{'propensityscales': 'mistake'},
             'File path to pickled propensity scales not recognised'],
            [{'propensityscales': 'tests/test_files/mistake.pkl'},
             'File path to pickled propensity scales not recognised'],
            [{'propensityscales': 'tests/test_files/example_pdb_253L.pdb'},
             'File path to pickled propensity scales not recognised'],
            [{'propensityscales': 'tests/test_files/example_input_df.pkl'},
             'Data in tests/test_files/example_input_df.pkl is not a pickled dictionary'],
            [{'propensityscales': 'tests/test_files/example_prop_freq_scales.pkl'},
             OrderedDict({'scale_1': [1, 2, 3, 4, 5], 'scale_2': [6, 7, 8, 9, 10],
                          'phipsi_disc': [11, 12, 13, 14, 15]})]
        ]

        for pair in expected_vals:
            params = pair[0]
            exp_output = pair[1]
            self.assertEqual(def_propensity_scales(params, test=True), exp_output)

    def test_frequency_dicts(self):
        """
        Tests that path to pickled frequency dictionaries is parsed correctly,
        and that the dataframe is unpickled
        """

        sub1_params = {}
        sub1_params['includefrequencyscales'] = 'Yes'

        expected_vals = [
            [{},
             'File path to pickled frequency scales not recognised'],
            [{'frequencyscales': 'mistake'},
             'File path to pickled frequency scales not recognised'],
            [{'frequencyscales': 'tests/test_files/mistake.pkl'},
             'File path to pickled frequency scales not recognised'],
            [{'frequencyscales': 'tests/test_files/example_pdb_253L.pdb'},
             'File path to pickled frequency scales not recognised'],
            [{'frequencyscales': 'tests/test_files/example_input_df.pkl'},
             'Data in tests/test_files/example_input_df.pkl is not a pickled dictionary'],
            [{'frequencyscales': 'tests/test_files/example_input_df.pkl',
             'includefrequencyscales': 'No'},
             {}],
            [{'frequencyscales': 'tests/test_files/example_prop_freq_scales.pkl'},
             OrderedDict({'scale_1': [1, 2, 3, 4, 5], 'scale_2': [6, 7, 8, 9, 10],
                          'phipsi_disc': [11, 12, 13, 14, 15]})]
        ]

        for pair in expected_vals:
            sub2_params = pair[0]
            all_params = {**sub1_params, **sub2_params}
            exp_output = pair[1]
            self.assertEqual(def_frequency_scales(all_params, test=True), exp_output)

    def test_convert_str_to_dict(self):
        """
        Tests function correctly parses dictionary input in string format to
        Python dictionary format
        """

        dict_id = 'place fill'

        expected_vals = [
            ['', int, 'Dictionary describing place fill not recognised'],
            ['mistake', int, 'Dictionary describing place fill not recognised'],
            ['{mistake}', int, 'Dictionary describing place fill not recognised'],
            ['{"a": 1, "b": 2}', int, {'a': 1, 'b': 2}],
            ['{"a": 1.7, "b": 2.0}', float, {'a': 1.7, 'b': 2.0}],
            ['{"a": 1.7, "b":}', float, {}],
            ['{"a": 1.0, "b": 2.0}', int, {}],
            ['{"a": 1, "b": "mistake"}', int, {}]
        ]

        for triple in expected_vals:
            dict_string = triple[0]
            int_or_float = triple[1]
            exp_output = triple[2]
            self.assertEqual(
                convert_str_to_dict(dict_string, dict_id, int_or_float, test=True),
                exp_output
            )

    def test_prop_freq_dict_weights(self):
        """
        Tests relative weighting of propensity and frequency dictionaries is
        parsed correctly
        """

        params = {}
        params['dictnameindices'] = {'pairorindv': 1}

        params['propensityscales'] = {}
        params['frequencyscales'] = {}
        self.assertEqual(
            def_prop_freq_scale_weights(params, test=True), {}
        )

        params['propensityscales'] = {'test1_pair_propensity': 0.05,
                                      'test1_indv_propensity': 1}
        params['frequencyscales'] = {'test1_pair_frequency': 2.5,
                                     'test1_indv_frequency': 0.5}
        params['scaleweights'] = 'equal'
        self.assertEqual(
            def_prop_freq_scale_weights(params, test=True),
            {'test1_pair_propensity': 0.5,
             'test1_indv_propensity': 1,
             'test1_pair_frequency': 0.5,
             'test1_indv_frequency': 1}
        )

        params['scaleweights'] = (
            '{"test1_pair_propensity": 0.4,"test1_indv_propensity": 0.26,'
            '"test1_pair_frequency": 0.37,"test1_indv_frequency": 3}'
        )
        self.assertEqual(
            def_prop_freq_scale_weights(params, test=True),
            {'test1_pair_propensity': 0.4,
             'test1_indv_propensity': 0.26,
             'test1_pair_frequency': 0.37,
             'test1_indv_frequency': 3}
        )

    def test_propensity_weight(self):
        """
        Tests weighting of propensity and frequency scales is parsed correctly
        """

        sub1_params = {'frequencyscales': None,
                       'propensityscales': None}
        expected_vals = [
            [{'propensityweight': '2', 'frequencyscales': {}},
             ValueError],
            [{'propensityweight': '', 'frequencyscales': {}},
             1],
            [{'propensityweight': 'mistake'},
             ('Weighting for propensity scales not recognised - please enter a '
              'value between 0 and 1')],
            [{'propensityweight': '2'},
             ('Weighting for propensity scales not recognised - please enter a '
              'value between 0 and 1')],
            [{'propensityweight': '-1'},
             ('Weighting for propensity scales not recognised - please enter a '
              'value between 0 and 1')],
            [{'propensityweight': '1'},
             1.0],
            [{'propensityweight': '0'},
             0.0],
            [{'propensityweight': '0.42'},
             0.42]
        ]

        for index, pair in enumerate(expected_vals):
            sub2_params = pair[0]
            all_params = {**sub1_params, **sub2_params}
            exp_output = pair[1]

            if index == 0:
                self.assertRaises(exp_output, def_propensity_weight, all_params)
            else:
                self.assertEqual(def_propensity_weight(all_params, test=True), exp_output)

    def test_phipsi_cluster_coords(self):
        """
        """

        sub1_params = {}
        with open('tests/test_files/example_prop_freq_scales.pkl', 'rb') as f:
            comb_dict = pickle.load(f)
            sub1_params['propensityscales'] = {
                key: val for key, val in list(comb_dict.items())
                if key in ['scale_1', 'scale_2']
            }
            sub1_params['frequencyscales'] = {
                key: val for key, val in list(comb_dict.items())
                if key in ['phipsi_disc']
            }
        sub1_params['dictnameindices'] = {'prop1': 0,
                                          'discorcont': 1}

        expected_vals = [
            [{},
             ('File path to pickled phi / psi voronoi point coordinates not '
              'recognised')],
            [{'phipsiclustercoords': 'mistake'},
             ('File path to pickled phi / psi voronoi point coordinates not '
              'recognised')],
            [{'phipsiclustercoords': 'tests/test_files/mistake.pkl'},
             ('File path to pickled phi / psi voronoi point coordinates not '
              'recognised')],
            [{'phipsiclustercoords': 'example_pdb_253L.pdb'},
             ('File path to pickled phi / psi voronoi point coordinates not '
              'recognised')],
            [{'phipsiclustercoords': 'tests/test_files/example_input_df.pkl'},
             ('Data in tests/test_files/example_input_df.pkl is not a '
              'pickled dictionary')],
            [{'phipsiclustercoords': 'tests/test_files/example_prop_freq_scales.pkl'},
             OrderedDict({'scale_1': [1, 2, 3, 4, 5], 'scale_2': [6, 7, 8, 9, 10],
                          'phipsi_disc': [11, 12, 13, 14, 15]})]
        ]

        for pair in expected_vals:
            sub2_params = pair[0]
            all_params = {**sub1_params, **sub2_params}
            exp_output = pair[1]
            self.assertEqual(
                def_phipsi_cluster_coords(all_params, test=True), exp_output
            )

    def test_working_directory(self):
        """
        Tests that working directory has been parsed correctly
        """

        expected_vals = [
            [{},
             'Path to working directory not recognised'],
            [{'workingdirectory': '/mistake/in/dir/path'},
             'Path to working directory not recognised'],
            [{'workingdirectory': 'tests/test_files/example_input_df.pkl'},
             'Path to working directory not recognised'],
            [{'workingdirectory': 'tests/test_files/'},
             'tests/test_files']
        ]

        for pair in expected_vals:
            params = pair[0]
            exp_output = pair[1]
            self.assertEqual(
                def_working_directory(params, test=True), exp_output
            )

    def test_barrel_or_sandwich(self):
        """
        Tests correct selection of "barrel" or "sandwich" options.
        """

        expected_vals = [
            [{}, 'Backbone structure not recognised'],
            [{'barrelorsandwich': 'mistake'}, 'Backbone structure not recognised'],
            [{'barrelorsandwich': 'barrel'}, '2.40'],
            [{'barrelorsandwich': '2.40'}, '2.40'],
            [{'barrelorsandwich': 'sandwich'}, '2.60'],
            [{'barrelorsandwich': '2.60'}, '2.60']
        ]

        for pair in expected_vals:
            params = pair[0]
            exp_output = pair[1]
            self.assertEqual(
                def_barrel_or_sandwich(params, test=True), exp_output
            )

    def test_job_id(self):
        """
        Tests jobid is parsed correctly
        """

        expected_vals = [
            [{'jobid': ''}, ''],
            [{'jobid': 'test'}, 'test'],
            [{'jobid': 'random'}, 'random']
        ]

        for index, pair in enumerate(expected_vals):
            params = pair[0]
            exp_output = pair[1]

            if index == 2:
                self.assertNotEqual(def_jobid(params, test=True), exp_output)
            else:
                self.assertEqual(def_jobid(params, test=True), exp_output)

    def test_method_initial_side_chains(self):
        """
        Tests method of generating starting sequences is parsed correctly
        """

        expected_vals = [
            [{},
             ('Method not recognised - please select one of "random", '
              '"rawpropensity" or "rankpropensity"')],
            [{'initialseqmethod': 'mistake'},
             ('Method not recognised - please select one of "random", '
              '"rawpropensity" or "rankpropensity"')],
            [{'initialseqmethod': 'random'}, 'random'],
            [{'initialseqmethod': 'rawpropensity'}, 'rawpropensity'],
            [{'initialseqmethod': 'rankpropensity'}, 'rankpropensity']

        ]

        for index, pair in enumerate(expected_vals):
            params = pair[0]
            exp_output = pair[1]
            self.assertEqual(
                def_method_initial_seq(params, test=True), exp_output
            )

    def test_method_fitness_score(self):
        """
        Test method of measuring sequence fitness is parsed correctly
        """

        expected_vals = [
            [{},
             ('Method not recognised - please select one of "propensity", '
              '"allatom", "alternate", "split"')],
            [{'fitnessscoremethod': 'mistake'},
             ('Method not recognised - please select one of "propensity", '
              '"allatom", "alternate", "split"')],
            [{'fitnessscoremethod': 'propensity'}, 'propensity'],
            [{'fitnessscoremethod': 'allatom'}, 'allatom'],
            [{'fitnessscoremethod': 'alternate'}, 'alternate'],
            [{'fitnessscoremethod': 'split'}, 'split']
        ]

        for index, pair in enumerate(expected_vals):
            params = pair[0]
            exp_output = pair[1]
            self.assertEqual(
                def_method_fitness_scoring(params, test=True), exp_output
            )

    def test_split_fraction(self):
        """
        Tests that fraction of samples to have their fitness scored using
        propensity scales is parsed correctly
        """

        sub1_params = {'fitnessscoremethod': 'split'}

        expected_vals = [
            [{},
             ('Fraction of samples to be optimised against propensity not '
              'recognised - please enter a value between 0 and 1')],
            [{'splitfraction': 'mistake'},
             ('Fraction of samples to be optimised against propensity not '
              'recognised - please enter a value between 0 and 1')],
            [{'splitfraction': '2'},
             ('Fraction of samples to be optimised against propensity not '
              'recognised - please enter a value between 0 and 1')],
            [{'splitfraction': '-1'},
             ('Fraction of samples to be optimised against propensity not '
              'recognised - please enter a value between 0 and 1')],
            [{'splitfraction': '1'},
             1.0],
            [{'splitfraction': '0'},
             0.0],
            [{'splitfraction': '0.42'},
             0.42]
        ]

        for pair in expected_vals:
            sub2_params = pair[0]
            all_params = {**sub1_params, **sub2_params}
            exp_output = pair[1]
            self.assertEqual(
                def_split_fraction(all_params, test=True), exp_output
            )

    def test_method_select_mating_pop(self):
        """
        Tests that method of selecting sequences to form the mating population
        is parsed correctly
        """

        expected_vals = [
            [{},
             ('Method not recognised - please select one of "fittest", '
              '"roulettewheel" or "rankroulettewheel"')],
            [{'matingpopmethod': 'mistake'},
             ('Method not recognised - please select one of "fittest", '
              '"roulettewheel" or "rankroulettewheel"')],
            [{'matingpopmethod': 'fittest'}, 'fittest'],
            [{'matingpopmethod': 'roulettewheel'}, 'roulettewheel'],
            [{'matingpopmethod': 'rankroulettewheel'}, 'rankroulettewheel']
        ]

        for index, pair in enumerate(expected_vals):
            params = pair[0]
            exp_output = pair[1]
            self.assertEqual(
                def_method_select_mating_pop(params, test=True), exp_output
            )

    def test_unfit_fraction(self):
        """
        Tests that fraction of samples to have their fitness scored using
        propensity scales is parsed correctly
        """

        sub1_params = {'matingpopmethod': 'fittest'}

        expected_vals = [
            [{},
             ('Fraction of mating population to be comprised of unfit samples '
              'not recognised - please enter a value between 0 and 1')],
            [{'unfitfraction': 'mistake'},
             ('Fraction of mating population to be comprised of unfit samples '
              'not recognised - please enter a value between 0 and 1')],
            [{'unfitfraction': '2'},
             ('Fraction of mating population to be comprised of unfit samples '
              'not recognised - please enter a value between 0 and 1')],
            [{'unfitfraction': '-1'},
             ('Fraction of mating population to be comprised of unfit samples '
              'not recognised - please enter a value between 0 and 1')],
            [{'unfitfraction': '1'},
             1.0],
            [{'unfitfraction': '0'},
             0.0],
            [{'unfitfraction': '0.42'},
             0.42]
        ]

        for pair in expected_vals:
            sub2_params = pair[0]
            all_params = {**sub1_params, **sub2_params}
            exp_output = pair[1]
            self.assertEqual(
                def_unfit_fraction(all_params, test=True), exp_output
            )

    def test_method_crossover(self):
        """
        Tests method of sequence crossover is parsed correctly
        """

        expected_vals = [
            [{},
             ('Crossover method not recognised - please select one of '
              '"uniform" or "segmented"')],
            [{'crossovermethod': 'mistake'},
             ('Crossover method not recognised - please select one of '
              '"uniform" or "segmented"')],
            [{'crossovermethod': 'uniform'}, 'uniform'],
            [{'crossovermethod': 'segmented'}, 'segmented']
        ]

        for index, pair in enumerate(expected_vals):
            params = pair[0]
            exp_output = pair[1]
            self.assertEqual(
                def_method_crossover(params, test=True), exp_output
            )

    def test_crossover_prob(self):
        """
        Tests probability of (uniform) crossing over between sequence pairs
        (chromosomes) is parsed correctly
        """

        sub1_params = {'crossovermethod': 'uniform'}

        expected_vals = [
            [{},
             ('Probability of uniform crossover not recognised - please enter '
              'a value between 0 and 1')],
            [{'crossoverprob': 'mistake'},
             ('Probability of uniform crossover not recognised - please enter '
              'a value between 0 and 1')],
            [{'crossoverprob': '2'},
             ('Probability of uniform crossover not recognised - please enter '
              'a value between 0 and 1')],
            [{'crossoverprob': '-1'},
             ('Probability of uniform crossover not recognised - please enter '
              'a value between 0 and 1')],
            [{'crossoverprob': '1'},
             1.0],
            [{'crossoverprob': '0'},
             0.0],
            [{'crossoverprob': '0.42'},
             0.42]
        ]

        for pair in expected_vals:
            sub2_params = pair[0]
            all_params = {**sub1_params, **sub2_params}
            exp_output = pair[1]
            self.assertEqual(
                def_crossover_prob(all_params, test=True), exp_output
            )

    def test_swap_start_prob(self):
        """
        Tests probability of initiating a (segmented) crossover between
        sequence pairs (chromosomes) is parsed correctly
        """

        sub1_params = {'crossovermethod': 'segmented'}

        expected_vals = [
            [{},
             ('Probability of initiating segmented crossover not recognised - '
              'please enter a value between 0 and 1')],
            [{'swapstartprob': 'mistake'},
             ('Probability of initiating segmented crossover not recognised - '
              'please enter a value between 0 and 1')],
            [{'swapstartprob': '2'},
             ('Probability of initiating segmented crossover not recognised - '
              'please enter a value between 0 and 1')],
            [{'swapstartprob': '-1'},
             ('Probability of initiating segmented crossover not recognised - '
              'please enter a value between 0 and 1')],
            [{'swapstartprob': '1'},
             1.0],
            [{'swapstartprob': '0'},
             0.0],
            [{'swapstartprob': '0.42'},
             0.42]
        ]

        for pair in expected_vals:
            sub2_params = pair[0]
            all_params = {**sub1_params, **sub2_params}
            exp_output = pair[1]
            self.assertEqual(
                def_swap_start_prob(all_params, test=True), exp_output
            )

    def test_swap_stop_prob(self):
        """
        Tests probability of ending a (segmented) crossover between sequence
        pairs (chromosomes) is parsed correctly
        """

        sub1_params = {'crossovermethod': 'segmented'}

        expected_vals = [
            [{},
             ('Probability of ending segmented crossover not recognised - '
              'please enter a value between 0 and 1')],
            [{'swapstopprob': 'mistake'},
             ('Probability of ending segmented crossover not recognised - '
              'please enter a value between 0 and 1')],
            [{'swapstopprob': '2'},
             ('Probability of ending segmented crossover not recognised - '
              'please enter a value between 0 and 1')],
            [{'swapstopprob': '-1'},
             ('Probability of ending segmented crossover not recognised - '
              'please enter a value between 0 and 1')],
            [{'swapstopprob': '1'},
             1.0],
            [{'swapstopprob': '0'},
             0.0],
            [{'swapstopprob': '0.42'},
             0.42]
        ]

        for pair in expected_vals:
            sub2_params = pair[0]
            all_params = {**sub1_params, **sub2_params}
            exp_output = pair[1]
            self.assertEqual(
                def_swap_stop_prob(all_params, test=True), exp_output
            )

    def test_method_mutation(self):
        """
        Tests method of sequence mutation is parsed correctly
        """

        expected_vals = [
            [{},
             ('Mutation method not recognised - please select one of "swap" or '
              '"scramble"')],
            [{'mutationmethod': 'mistake'},
             ('Mutation method not recognised - please select one of "swap" or'
              ' "scramble"')],
            [{'mutationmethod': 'swap'}, 'swap'],
            [{'mutationmethod': 'scramble'}, 'scramble']
        ]

        for index, pair in enumerate(expected_vals):
            params = pair[0]
            exp_output = pair[1]
            self.assertEqual(def_method_mutation(params, test=True), exp_output)

    def test_mutation_prob(self):
        """
        Tests probability of sequence mutation is parsed correctly
        """

        expected_vals = [
            [{},
             ('Probability of mutation not recognised - please enter a value '
              'between 0 and 1')],
            [{'mutationprob': 'mistake'},
             ('Probability of mutation not recognised - please enter a value '
              'between 0 and 1')],
            [{'mutationprob': '2'},
             ('Probability of mutation not recognised - please enter a value '
              'between 0 and 1')],
            [{'mutationprob': '-1'},
             ('Probability of mutation not recognised - please enter a value '
              'between 0 and 1')],
            [{'mutationprob': '1'},
             1.0],
            [{'mutationprob': '0'},
             0.0],
            [{'mutationprob': '0.42'},
             0.42]
        ]

        for pair in expected_vals:
            params = pair[0]
            exp_output = pair[1]
            self.assertEqual(def_mutation_prob(params, test=True), exp_output)

    def test_pop_size(self):
        """
        Tests that population size is parsed correctly
        """

        expected_vals = [
            [{'fitnessscoremethod': 'propensity',
              'splitfraction': ''},
             ('Population size not recognised - please enter a positive even '
              'integer')],
            [{'fitnessscoremethod': 'propensity',
              'splitfraction': '',
              'populationsize': 'mistake'},
             ('Population size not recognised - please enter a positive even '
              'integer')],
            [{'fitnessscoremethod': 'propensity',
              'splitfraction': '',
              'populationsize': '4.2'},
             ('Population size not recognised - please enter a positive even '
              'integer')],
            [{'fitnessscoremethod': 'propensity',
              'splitfraction': '',
              'populationsize': '0'},
             ('Population size not recognised - please enter a positive even '
              'integer')],
            [{'fitnessscoremethod': 'propensity',
              'splitfraction': '',
              'populationsize': '-10'},
             ('Population size not recognised - please enter a positive even '
              'integer')],
            [{'fitnessscoremethod': 'propensity',
              'splitfraction': '',
              'populationsize': '13'},
             ('Population size not recognised - please enter a positive even '
              'integer')],
            [{'fitnessscoremethod': 'propensity',
              'splitfraction': '',
              'populationsize': '42'},
             (42)],
            [{'fitnessscoremethod': 'split',
              'splitfraction': 0.5,
              'populationsize': '0'},
             ('Population size not recognised - please enter a positive integer'
              ' that when multiplied by 0.5x the fraction of samples to be '
              'optimised against propensity gives an integer value')],
            [{'fitnessscoremethod': 'split',
              'splitfraction': 0.5,
              'populationsize': '42'},
             ('Population size not recognised - please enter a positive integer'
              ' that when multiplied by 0.5x the fraction of samples to be '
              'optimised against propensity gives an integer value')],
            [{'fitnessscoremethod': 'split',
              'splitfraction': 0.7,
              'populationsize': '20'},
             (20)],
        ]

        for pair in expected_vals:
            params = pair[0]
            exp_output = pair[1]
            self.assertEqual(def_pop_size(params, test=True), exp_output)

    def test_num_gens(self):
        """
        Tests that number of generations for which to run the genetic algorithm
        is parsed correctly
        """

        expected_vals = [
            [{},
             ('Maximum number of generations not recognised - please enter a '
              'positive integer')],
            [{'maxnumgenerations': 'mistake'},
             ('Maximum number of generations not recognised - please enter a '
              'positive integer')],
            [{'maxnumgenerations': '0'},
             ('Maximum number of generations not recognised - please enter a '
              'positive integer')],
            [{'maxnumgenerations': '-10'},
             ('Maximum number of generations not recognised - please enter a '
              'positive integer')],
            [{'maxnumgenerations': '1.5'},
             ('Maximum number of generations not recognised - please enter a '
              'positive integer')],
            [{'maxnumgenerations': '42.0'},
             ('Maximum number of generations not recognised - please enter a '
              'positive integer')],
            [{'maxnumgenerations': '42'},
             42]
        ]

        for pair in expected_vals:
            params = pair[0]
            exp_output = pair[1]
            self.assertEqual(def_num_gens(params, test=True), exp_output)
