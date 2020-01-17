
import pandas as pd
import unittest
from collections import OrderedDict

from betadesigner.subroutines.find_parameters import (
    def_input_df_path, def_input_pdb, def_propensity_scales, def_frequency_scales,
    convert_str_to_dict, def_prop_freq_scale_weights, def_propensity_weight,
    def_phipsi_cluster_coords, def_working_directory, def_barrel_or_sandwich,
    def_jobid

)


class testParameterParsing(unittest.TestCase):
    """
    """

    def test_input_df(self):
        """
        Tests that input dataframe path is parsed correctly, and that the
        dataframe is unpickled
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
             OrderedDict({'scale_1': [1, 2, 3, 4, 5], 'scale_2': [6, 7, 8, 9, 10]})]
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
             OrderedDict({'scale_1': [1, 2, 3, 4, 5], 'scale_2': [6, 7, 8, 9, 10]})]
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
             'Weighting for propensity scales not recognised - please enter a value between 0 and 1'],
            [{'propensityweight': '2'},
             'Weighting for propensity scales not recognised - please enter a value between 0 and 1'],
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

        expected_vals = [
            []
        ]

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
             'tests/test_files/']
        ]

        for pair in expected_vals:
            params = pair[0]
            exp_output = pair[1]
            self.assertEqual(def_working_directory(params, test=True), exp_output)

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
            self.assertEqual(def_barrel_or_sandwich(params, test=True), exp_output)

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
        """

        pass

    def test_method_fitness_score(self):
        """
        """

        pass

    def test_split_fraction(self):
        """
        """

        pass

    def test_method_select_mating_pop(self):
        """
        """

        pass

    def test_unfit_fraction(self):
        """
        """

        pass

    def test_method_crossover(self):
        """
        """

        pass

    def test_crossover_prob(self):
        """
        """

        pass

    def test_swap_start_prob(self):
        """
        """

        pass

    def test_swap_stop_prob(self):
        """
        """

        pass

    def test_method_mutation(self):
        """
        """

        pass

    def test_mutation_prob(self):
        """
        """

        pass

    def test_pop_size(self):
        """
        """

        pass

    def test_num_gens(self):
        """
        """

        pass
