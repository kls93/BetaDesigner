
import os
import pickle
import random
import shutil
import string
import numpy as np
import pandas as pd
from collections import OrderedDict

if __name__ == 'subroutines.find_params':
    from subroutines.variables import gen_amino_acids_dict
else:
    from betadesigner.subroutines.variables import gen_amino_acids_dict

prompt = '> '


def calc_parent_voronoi_cluster(input_df, cluster_coords):
    """
    Calculates to which discrete bins in Ramachandran (phi psi) space the
    residues in the input structure belong
    """

    phi_psi_list = ['']*input_df.shape[0]

    for row in range(input_df.shape[0]):
        try:
            int_or_ext = input_df['int_ext'][row]
            phi = input_df['phi'][row]
            psi = input_df['psi'][row]

            try:
                phi = float(phi)
                psi = float(psi)
                phi_psi = np.array([phi, psi])
                distances = np.sqrt(np.sum(np.square(cluster_coords[int_or_ext]-phi_psi), axis=1))
                voronoi_index = np.abs(distances).argmin()
                phi_psi_list[row] = voronoi_index
            except ValueError:
                phi_psi_list[row] = np.nan

        except KeyError:
            raise KeyError('One or more properties expected to be included in '
                           'the input dataframe are missing')

    phi_psi_class_df = pd.DataFrame({'phi_psi_class': phi_psi_list})
    input_df = pd.concat([input_df, phi_psi_class_df], axis=1)

    return input_df


def find_params(args):
    """
    Defines program parameter values. If an input file is provided, the code
    first tries to extract program params from this file. It then
    requests user input for all remaining undefined params and any
    defined params with unrecognised values.
    """

    params = OrderedDict()

    if not vars(args)['input_file'] is None:
        try:
            with open(vars(args)['input_file'], 'r') as f:
                for line in f.readlines():
                    key = line.split(':')[0].replace(' ', '').lower()
                    value = line.split(':')[1].replace('\n', '').strip()

                    if key in ['inputdataframepath', 'inputpdb', 'propensityscales',
                               'frequencyscales', 'scaleweights',
                               'phipsiclustercoords']:
                        value = value.replace('\\', '/')  # For windows file paths
                        value = '/{}'.format(value.strip('/'))
                    elif key in ['workingdirectory']:
                        value = value.replace('\\', '/')  # For windows file paths
                        value = '/{}/'.format(value.strip('/'))
                    elif key in ['jobid', 'dictnameindices']:
                        value = value.replace(' ', '')
                    elif key in ['propvsfreqweight', 'barrelorsandwich',
                                 'initialseqmethod', 'fitnessscoremethod',
                                 'splitfraction', 'matingpopmethod',
                                 'unfitfraction', 'crossovermethod',
                                 'crossoverprob', 'swapstartprob',
                                 'swapstopprob', 'mutationmethod',
                                 'populationsize', 'numberofgenerations']:
                        value = value.lower().replace(' ', '')

                    params[key] = value

        except FileNotFoundError:
            print('Path to input file not recognised')

    # Defines absolute file path to input dataframe
    if 'inputdataframepath' in params:
        if (
            (not os.path.isfile(params['inputdataframepath']))
            or
            (not params['inputdataframepath'].endswith('.pkl'))
        ):
            print('File path to pickled input dataframe not recognised')
            params.pop('inputdataframepath')

    if not 'inputdataframepath' in params:
        input_df = ''
        while (
            (not os.path.isfile(input_df))
            or
            (not input_df.endswith('.pkl'))
        ):
            print('Specify absolute file path of pickled input dataframe:')
            input_df = '/' + input(prompt).replace('\\').strip('/')

            if os.path.isfile(input_df) and input_df.endswith('.pkl'):
                params['inputdataframepath'] = input_df
                break
            else:
                print('File path to pickled input dataframe not recognised')

    # Defines absolute file path to input PDB file (the file fed into DataGen)
    if 'inputpdb' in params:
        if (
            (not os.path.isfile(params['inputpdb']))
            or
            (not params['inputpdb'].endswith('.pdb'))
        ):
            print('File path to input PDB file not recognised')
            params.pop('inputpdb')

    if not 'inputpdb' in params:
        input_pdb = ''
        while (
            (not os.path.isfile(input_pdb))
            or
            (not input_pdb.endswith('.pdb'))
        ):
            print('Specify absolute file path of input PDB file:')
            input_pdb = '/' + input(prompt).replace('\\').strip('/')

            if os.path.isfile(input_pdb) and input_pdb.endswith('.pdb'):
                params['inputpdb'] = input_pdb
                break
            else:
                print('File path to input PDB file not recognised')

    # Defines absolute file path to pickle file listing propensity scales
    if 'propensityscales' in params:
        if (
            (not os.path.isfile(params['propensityscales']))
            or
            (not params['propensityscales'].endswith('.pkl'))
        ):
            print('File path to pickled propensity scales not recognised')
            params.pop('propensityscales')
        else:
            with open(params['propensityscales'], 'rb') as pickle_file:
                propensity_scales_dict = pickle.load(pickle_file)
            if type(propensity_scales_dict) == dict:
                params['propensityscales'] = propensity_scales_dict
            else:
                params.pop('propensityscales')

    if not 'propensityscales' in params:
        propensity_scales_file = ''
        scales_provided = False
        while scales_provided is False:
            print('Specify absolute file path of pickled propensity scales:')
            propensity_scales_file = '/' + input(prompt).replace('\\').strip('/')

            if (
                (os.path.isfile(propensity_scales_file))
                and
                (propensity_scales_file.endswith('.pkl'))
            ):
                with open(propensity_scales_file, 'rb') as pickle_file:
                    propensity_scales_dict = pickle.load(pickle_file)
                if type(propensity_scales_dict) == dict:
                    params['propensityscales'] = propensity_scales_dict
                    scales_provided = True
                    break
                else:
                    print('Data in {} is not a pickled dictionary'.format(
                        propensity_scales_file)
                    )
            else:
                print('File path to pickled propensity scales not recognised')

    # Defines absolute file path to pickle file listing frequency scales
    if 'frequencyscales' in params:
        if (
            (not os.path.isfile(params['frequencyscales']))
            or
            (not params['frequencyscales'].endswith('.pkl'))
        ):
            print('File path to pickled frequency scales not recognised')
            params.pop('frequencyscales')
        else:
            with open(params['frequencyscales'], 'rb') as pickle_file:
                frequency_scales_dict = pickle.load(pickle_file)
            if type(frequency_scales_dict) == dict:
                params['frequencyscales'] = frequency_scales_dict
            else:
                params.pop('frequencyscales')

    if not 'frequencyscales' in params:
        print('Include frequency scales?')
        frequency_input = input(prompt)

        while not frequency_input in ['yes', 'no', 'y', 'n']:
            print('User input not recognised - please specify ("yes" or "no") '
                  'whether you would like to include frequency scales:')
            frequency_input = input(prompt).lower()

        if frequency_input in ['yes', 'y']:
            frequency_scales_file = ''

            scales_provided = False
            while scales_provided is False:
                print('Specify absolute file path of pickled frequency scales:')
                frequency_scales_file = '/' + input(prompt).replace('\\').strip('/')

                if (
                    (os.path.isfile(frequency_scales_file))
                    and
                    (frequency_scales_file.endswith('.pkl'))
                ):
                    with open(frequency_scales_file, 'rb') as pickle_file:
                        frequency_scales_dict = pickle.load(pickle_file)
                    if type(frequency_scales_dict) == dict:
                        params['frequencyscales'] = frequency_scales_dict
                        scales_provided = True
                        break
                    else:
                        print('Data in {} is not a pickled dictionary'.format(
                            frequency_scales_file)
                        )
                else:
                    print('File path to pickled frequency scales not recognised')
        else:
            params['frequencyscales'] = {}

    # Defines propensity scale weights
    # N.B. Have left this hyperparameter in for now in case I want to use it in
    # the future, but in general, and certainly in the case of my initial
    # design run, I think this is "an optimisation too far". Will keep all the
    # same (i.e. params['scaleweights'] = 'equal') for now
    params['scaleweights'] = 'equal'  # Comment out this line if want to use
    # different weightings for different scales
    scales = (  list(params['propensityscales'].keys())
              + list(params['frequencyscales'].keys()))
    if 'scaleweights' in params:
        if params['scaleweights'].strip('/').lower() == 'equal':
            scale_weights = {}
            for dict_name in scales:
                if dict_name.split('_')[5] == 'indv':
                    scale_weights[dict_name] = 1
                elif dict_name.split('_')[5] == 'pair':
                    scale_weights[dict_name] = 0.5  # Each pair is counted twice
            params['scaleweights'] = scale_weights
        elif (
            (os.path.isfile(params['scaleweights']))
            and
            (params['scaleweights'].endswith('.pkl'))
        ):
            with open(params['scaleweights'], 'rb') as pickle_file:
                scale_weights = pickle.load(pickle_file)
            if type(scale_weights) == dict:
                params['scaleweights'] = scale_weights
            else:
                print('Propensity scale weights dictionary not recognised')
                params['scaleweights'] = {}
        else:
            print('Propensity scale weights not provided')
            params['scaleweights'] = {}
    else:
        params['scaleweights'] = {}

    for dict_name in scales:
        if not dict_name in list(params['scaleweights'].keys()):
            print('Weight for {} not provided'.format(dict_name))
            scale_weight = ''
        else:
            scale_weight = params['scaleweights'][dict_name]

        while not type(scale_weight) in [int, float]:
            print('Weighting for {} not recognised.\n'
                  'Specify weight for {}'.format(dict_name, dict_name))
            scale_weight = input(prompt)

            try:
                scale_weight = float(scale_weight)
                params['scaleweights'][dict_name] = scale_weight
                break
            except ValueError:
                scale_weight = ''

    for scale in list(params['scaleweights'].keys()):
        if not scale in scales:
            del params['scaleweights'][scale]

    # Defines order of properties in names of input propensity / frequency
    # scales. For now this is fixed in the order below, update as necessary if
    # the current naming convention changes.
    params['dictnameindices'] = {'intorext': 0,
                                 'edgeorcent': 1,
                                 'prop1': 2,
                                 'interactiontype': 3,
                                 'pairorindv': 4,
                                 'discorcont': 5,
                                 'proporfreq': 6}
    """
    if 'dictnameindices' in params:
        if (
            (not os.path.isfile(params['dictnameindices']))
            or
            (not params['dictnameindices'].endswith('.pkl'))
        ):
            print('File path to pickled dictionary describing propensity scale'
                  ' naming convention not recognised')
            params.pop('dictnameindices')
        else:
            with open(params['dictnameindices'], 'rb') as pickle_file:
                dict_name_indices = pickle.load(pickle_file)
            if type(dict_name_indices) == dict:
                params['dictnameindices'] = dict_name_indices
            else:
                params.pop('dictnameindices')

    if not 'dictnameindices' in params:
        dict_name_desc = ''
        while not (    dict_name_desc[0] == '{'
                   and dict_name_desc[-1] == '}'
                   and ':' in dict_name_desc
        ):
            dict_name_indices = {}
            dict_name_desc = input('Input dictionary describing propensity '
                                   'scale naming convention').strip()
            if (    dict_name_desc[0] == '{'
                and dict_name_desc[-1] == '}'
                and ':' in dict_name_desc
            ):
                dict_name_desc = dict_name_desc.strip('{').strip('}')
                error = False
                for key_val_pair in dict_name_desc.split(','):
                    if len(key_val_pair.split(':')) == 2:
                        key = key_val_pair.split(':')[0]
                        val = key_val_pair.split(':')[1]
                        try:
                            int_val = int(val)
                            if val == str(int_val):
                                val = int_val
                            else:
                                error = True
                        except ValueError:
                            error = True
                        dict_name_indices[key] = val
                    else:
                        error = True
                if error is False:
                    params['dictnameindices'] = dict_name_indices
                    break
                else:
                    dict_name_desc = ''
                    print('File path to pickled dictionary describing propensity'
                          ' scale naming convention not recognised')
            else:
                dict_name_desc = ''
                print('File path to pickled dictionary describing propensity '
                      'scale naming convention not recognised')
    """

    # Defines weighting between propensity and frequency scales. Must be
    # defined after 'propensityscales' and 'frequencyscales'
    # Commented out because this hyperparameter has been selected for
    # optimisation with hyperopt
    if 'propvsfreqweight' in params:
        params.pop('propvsfreqweight')
    """
    if 'propvsfreqweight' in params:
        if params['frequencyscales'] == {}:  # This works for both unordered
        # and ordered dictionaries
            raise ValueError('Value provided for "propvsfreqweight", but no'
                             'input frequency scales have been defined.')

        try:
            prop_freq_weight = float(params['propvsfreqweight'])
            if 0 <= prop_freq_weight <= 1:
                params['propvsfreqweight'] = {'propensity': prop_freq_weight,
                                              'frequency': 1-prop_freq_weight}
            else:
                print('Weighting for propensity scales not recognised - please '
                      'enter a value between 0 and 1')
                params.pop('propvsfreqweight')
        except ValueError:
            print('Weighting for propensity scales not recognised - please '
                  'enter a value between 0 and 1')
            params.pop('propvsfreqweight')

    if not 'propvsfreqweight' in params:
        if params['frequencyscales'] == {}:
            params['propvsfreqweight'] = {'propensity': 1,
                                          'frequency': 0}
        else:
            prop_freq_weight = ''
            while not type(prop_freq_weight) == float:
                print('Specify weight for propensity scales:')
                prop_freq_weight = input(prompt)

                try:
                    prop_freq_weight = float(prop_freq_weight)
                    if 0 <= prop_freq_weight <= 1:
                        params['propvsfreqweight'] = {
                            'propensity': prop_freq_weight,
                            'frequency': 1-prop_freq_weight
                        }
                        break
                    else:
                        print('Weighting for propensity scales not recognised '
                              '- please enter a value between 0 and 1')
                        prop_freq_weight = ''
                except ValueError:
                    print('Weighting for propensity scales not recognised - '
                          'please enter a value between 0 and 1')
                    prop_freq_weight = ''
    """

    # Calculates phi and psi classes if discrete phi / psi dict is input
    for dict_label in scales:
        dict_label = dict_label.split('_')

        if all(x in dict_label for x in ['phi', 'psi', 'disc']):
            if 'phipsiclustercoords' in list(params.keys()):
                if (
                    (not os.path.isfile(params['phipsiclustercoords']))
                    or
                    (not params['phipsiclustercoords'].endswith('.pkl'))
                ):
                    print('File path to pickled phi / psi voronoi point '
                          'coordinates not recognised')
                    params.pop('phipsiclustercoords')
                else:
                    with open(params['phipsiclustercoords'], 'rb') as pickle_file:
                        cluster_coords = pickle.load(pickle_file)
                    params['phipsiclustercoords'] = cluster_coords

            if not 'phipsiclustercoords' in params:
                cluster_coords_file = ''
                while (
                    (not os.path.isfile(cluster_coords_file))
                    or
                    (not cluster_coords_file.endswith('.pkl'))
                ):
                    print('Specify absolute file path of pickled phi / psi '
                          'voronoi point coordinates:')
                    cluster_coords_file = '/' + input(prompt).replace('\\').strip('/')

                    if (
                        (os.path.isfile(cluster_coords_file))
                        and
                        (cluster_coords_file.endswith('.pkl'))
                    ):
                        with open(cluster_coords_file, 'rb') as pickle_file:
                            cluster_coords = pickle.load(pickle_file)
                        params['phipsiclustercoords'] = cluster_coords
                        break
                    else:
                        print('File path to pickled phi / psi voronoi point '
                              'coordinates not recognised')
            break

    # Checks whether for loop has been broken
    else:
        if 'phipsiclustercoords' in params:
            params.pop('phipsiclustercoords')

    # Defines working directory
    if 'workingdirectory' in params:
        if not os.path.isdir(params['workingdirectory']):
            print('File path to working directory not recognised')
            params.pop('workingdirectory')

    if not 'workingdirectory' in params:
        working_directory = ''
        while not os.path.isdir(working_directory):
            print('Specify absolute path of working directory')
            working_directory = (
                '/' + input(prompt).replace('\\', '/').strip('/') + '/'
            )

            if os.path.isdir(working_directory):
                params['workingdirectory'] = working_directory
                break
            else:
                print('File path to working directory not recognised')

    # Defines whether the input structure is a beta-sandwich or a beta-barrel
    # backbone
    if 'barrelorsandwich' in params:
        if params['barrelorsandwich'] == 'barrel':
            params['barrelorsandwich'] = '2.40'
        elif params['barrelorsandwich'] == 'sandwich':
            params['barrelorsandwich'] = '2.60'

        if not params['barrelorsandwich'] in ['2.40', '2.60']:
            print('Backbone structure not recognised')
            params.pop('barrelorsandwich')

    if not 'barrelorsandwich' in params:
        barrel_or_sandwich = ''
        while not barrel_or_sandwich in ['barrel', '2.40', 'sandwich', '2.60']:
            print('Specify structure type - please enter "barrel" or "sandwich":')
            barrel_or_sandwich = input(prompt).lower().replace(' ', '')

            if barrel_or_sandwich in ['2.40', '2.60']:
                params['barrelorsandwich'] = barrel_or_sandwich
                break
            elif barrel_or_sandwich in ['barrel', 'sandwich']:
                if barrel_or_sandwich == 'barrel':
                    params['barrelorsandwich'] = '2.40'
                elif barrel_or_sandwich == 'sandwich':
                    params['barrelorsandwich'] = '2.60'
                break
            else:
                print('Structure type not recognised')

    # Assigns unique identification code to job
    if not 'jobid' in params:
        print('Specify unique ID (without spaces) for input structure (if you '
              'would like BetaDesigner to assign a random ID, enter "random")):')
        job_id = input(prompt)
        if job_id.lower() == 'random':
            job_id = ''.join([random.choice(string.ascii_letters + string.digits)
                              for i in range(6)])
        params['jobid'] = job_id

    # Defines method used to generate initial sequences for backbone structure
    if 'initialseqmethod' in params:
        if not params['initialseqmethod'] in [
            'random', 'rawpropensity', 'rankpropensity'
        ]:
            print('Method for determining initial side chain assignments not '
                  'recognised - please select one of "random", "rawpropensity" '
                  'or "rankpropensity"')
            params.pop('initialseqmethod')

    if not 'initialseqmethod' in params:
        method_initial_side_chains = ''
        while not method_initial_side_chains in [
            'random', 'rawpropensity', 'rankpropensity'
        ]:
            print('Specify method for determining initial side chain assignments:')
            method_initial_side_chains = input(prompt).lower().replace(' ', '')

            if method_initial_side_chains in [
                'random', 'rawpropensity', 'rankpropensity'
            ]:
                params['initialseqmethod'] = method_initial_side_chains
                break
            else:
                print('Method not recognised - please select one of "random", '
                      '"rawpropensity" or "rankpropensity"')

    # Defines method used to measure sequence fitness
    # N.B. Have fixed this hyperparameter as "split" for now
    params['fitnessscoremethod'] = 'split'
    """
    if 'fitnessscoremethod' in params:
        if not params['fitnessscoremethod'] in [
            'propensity', 'allatom', 'alternate', 'split'
        ]:
            print('Method for measuring sequence fitness not recognised - '
                  'please select one of "propensity", "allatom", "alternate" '
                  'or "split"')
            params.pop('fitnessscoremethod')

    if not 'fitnessscoremethod' in params:
        method_fitness_score = ''
        while not method_fitness_score in [
            'propensity', 'allatom', 'alternate', 'split'
        ]:
            print('Specify method for measuring sequence fitnesses:')
            method_fitness_score = input(prompt).lower().replace(' ', '')

            if method_fitness_score in [
                'propensity', 'allatom', 'alternate', 'split'
            ]:
                params['fitnessscoremethod'] = method_fitness_score
                break
            else:
                print('Method not recognised - please select one of '
                      '"propensity", "allatom", "alternate", "split"')
    """

    # Defines fraction of samples to be optimised against propensity in each
    # generation of the genetic algorithm.
    # N.B. Have left this hyperparameter in for now in case I want to use it in
    # the future, but in general, and certainly in the case of my initial
    # design run, I think this is "an optimisation too far". Have fixed at 0.5
    # for now
    params['splitfraction'] = 0.5
    """
    if params['fitnessscoremethod'] == 'split':
        if 'splitfraction' in params:
            try:
                split_fraction = float(params['splitfraction'])
                if 0 <= split_fraction <= 1:
                    params['splitfraction'] = split_fraction
                else:
                    print('Fraction of samples to be optimised against '
                          'propensity not recognised - please enter a value '
                          'between 0 and 1')
                    params.pop('splitfraction')
            except ValueError:
                print('Fraction of samples to be optimised against propensity '
                      'not recognised - please enter a value between 0 and 1')
                params.pop('splitfraction')

        if not 'splitfraction' in params:
            split_fraction = ''
            while not type(split_fraction) == float:
                print('Specify fraction of samples to be optimised against '
                      'propensity:')
                split_fraction = input(prompt)

                try:
                    split_fraction = float(split_fraction)
                    if 0 <= split_fraction <= 1:
                        params['splitfraction'] = split_fraction
                        break
                    else:
                        print('Fraction of samples to be optimised against '
                              'propensity not recognised - please enter a '
                              'value between 0 and 1')
                        split_fraction = ''
                except ValueError:
                    print('Fraction of samples to be optimised against propensity '
                          'not recognised - please enter a value between 0 and 1')
                    split_fraction = ''

    else:
        if 'splitfraction' in params:
            params.pop('splitfraction')
    """

    # Defines method used to select a population of individuals for mating
    # Commented out because this hyperparameter has been selected to be fixed
    # as "fittest" (with the value of the hyperparameter "unfitfraction" being
    # optimised with hyperopt)
    params['matingpopmethod'] = 'fittest'
    """
    if 'matingpopmethod' in params:
        if not params['matingpopmethod'] in [
            'fittest', 'roulettewheel', 'rankroulettewheel'
        ]:
            print('Method for generating mating population not recognised - '
                  'please select one of "fittest", "roulettewheel" or '
                  '"rankroulettewheel"')
            params.pop('matingpopmethod')

    if not 'matingpopmethod' in params:
        method_select_mating_pop = ''
        while not method_select_mating_pop in [
            'fittest', 'roulettewheel', 'rankroulettewheel'
        ]:
            print('Specify method for generating mating population:')
            method_select_mating_pop = input(prompt).lower().replace(' ', '')

            if method_select_mating_pop in [
                'fittest', 'roulettewheel', 'rankroulettewheel'
            ]:
                params['matingpopmethod'] = method_select_mating_pop
                break
            else:
                print('Method not recognised - please select one of "fittest", '
                      '"roulettewheel" or "rankroulettewheel"')
    """

    # Defines fraction of unfit sequences to be included in the mating
    # population at each generation of the genetic algorithm if
    # self.method_select_mating_pop == 'fittest'
    # Commented out because this hyperparameter has been selected to be
    # optimised with hyperopt
    if 'unfitfraction' in params:
        params.pop('unfitfraction')
    """
    if params['matingpopmethod'] == 'fittest':
        if 'unfitfraction' in params:
            try:
                unfit_fraction = float(params['unfitfraction'])
                if 0 <= unfit_fraction <= 1:
                    params['unfitfraction'] = unfit_fraction
                else:
                    print('Fraction of mating population to be comprised of '
                          'unfit samples not recognised - please enter a '
                          'value between 0 and 1')
                    params.pop('unfitfraction')
            except ValueError:
                print('Fraction of mating population to be comprised of unfit '
                      'samples not recognised - please enter a value between '
                      '0 and 1')
                params.pop('unfitfraction')

        if not 'unfitfraction' in params:
            unfit_fraction = ''
            while not type(unfit_fraction) == float:
                print('Specify fraction of mating population to be comprised '
                      'of unfit samples:')
                unfit_fraction = input(prompt)

                try:
                    unfit_fraction = float(unfit_fraction)
                    if 0 <= unfit_fraction <= 1:
                        params['unfitfraction'] = unfit_fraction
                        break
                    else:
                        print('Fraction of mating population to be comprised '
                              'of unfit samples not recognised - please enter '
                              'a value between 0 and 1')
                        unfit_fraction = ''
                except ValueError:
                    print('Fraction of mating population to be comprised of '
                          'unfit samples not recognised - please enter a '
                          'value between 0 and 1')
                    unfit_fraction = ''

    else:
        if 'unfitfraction' in params:
            params.pop('unfitfraction')
    """

    # Defines method used to crossover parent sequences to generate children
    # Commented out because this hyperparameter has been selected to be fixed
    # as "uniform" (with the value of the hyperparameter "crossoverprob" being
    # optimised with hyperopt)
    params['crossovermethod'] = 'uniform'
    """
    if 'crossovermethod' in params:
        if not params['crossovermethod'] in ['uniform', 'segmented']:
            print('Crossover method not recognised - please select one of '
                  '"uniform" or "segmented"')
            params.pop('crossovermethod')

    if not 'crossovermethod' in params:
        method_crossover = ''
        while not method_crossover in ['uniform', 'segmented']:
            print('Specify crossover method:')
            method_crossover = input(prompt).lower().replace(' ', '')

            if method_crossover in ['uniform', 'segmented']:
                params['crossovermethod'] = method_crossover
                break
            else:
                print('Crossover method not recognised - please select one of '
                      '"uniform" or "segmented"')
    """

    # Defines probability of exchanging amino acid identities for each node in
    # the network as part of a uniform crossover
    # Commented out because this hyperparameter has been selected to be
    # optimised with hyperopt
    if 'crossoverprob' in params:
        params.pop('crossoverprob')
    """
    if params['crossovermethod'] == 'uniform':
        if 'crossoverprob' in params:
            try:
                crossover_probability = float(params['crossoverprob'])
                if 0 <= crossover_probability <= 1:
                    params['crossoverprob'] = crossover_probability
                else:
                    print('Probability of uniform crossover not recognised - '
                          'please enter a value between 0 and 1')
                    params.pop('crossoverprob')
            except ValueError:
                print('Probability of uniform crossover not recognised - '
                      'please enter a value between 0 and 1')
                params.pop('crossoverprob')

        if not 'crossoverprob' in params:
            crossover_probability = ''
            while not type(crossover_probability) == float:
                print('Specify probability of uniform crossover:')
                crossover_probability = input(prompt)

                try:
                    crossover_probability = float(crossover_probability)
                    if 0 <= crossover_probability <= 1:
                        params['crossoverprob'] = crossover_probability
                        break
                    else:
                        print('Probability of uniform crossover not recognised '
                              '- please enter a value between 0 and 1')
                        crossover_probability = ''
                except ValueError:
                    print('Probability of uniform crossover not recognised - '
                          'please enter a value between 0 and 1')
                    crossover_probability = ''

    else:
        if 'crossoverprob' in params:
            params.pop('crossoverprob')
    """

    # Defines probability of starting a (segmented) crossover
    # Commented out because crossover method has been fixed as "uniform" for now
    if 'swapstartprob'in params:
        params.pop('swapstartprob')
    """
    if params['crossovermethod'] == 'segmented':
        if 'swapstartprob' in params:
            try:
                start_crossover_prob = float(params['swapstartprob'])
                if 0 <= start_crossover_prob <= 1:
                    params['swapstartprob'] = start_crossover_prob
                else:
                    print('Probability of initiating segmented crossover not '
                          'recognised - please enter a value between 0 and 1')
                    params.pop('swapstartprob')
            except ValueError:
                print('Probability of initiating segmented crossover not '
                      'recognised - please enter a value between 0 and 1')
                params.pop('swapstartprob')

        if not 'swapstartprob' in params:
            start_crossover_prob = ''
            while not type(start_crossover_prob) == float:
                print('Specify probability of initiating crossover:')
                start_crossover_prob = input(prompt)

                try:
                    start_crossover_prob = float(start_crossover_prob)
                    if 0 <= start_crossover_prob <= 1:
                        params['swapstartprob'] = start_crossover_prob
                        break
                    else:
                        print('Probability of initiating segmented crossover '
                              'not recognised - please enter a value between '
                              '0 and 1')
                        start_crossover_prob = ''
                except ValueError:
                    print('Probability of initiating segmented crossover not '
                          'recognised - please enter a value between 0 and 1')
                    start_crossover_prob = ''

    else:
        if 'swapstartprob' in params:
            params.pop('swapstartprob')
    """

    # Defines probability of stopping a (segmented) crossover
    # Commented out because crossover method has been fixed as "uniform" for now
    if 'swapstopprob'in params:
        params.pop('swapstopprob')
    """
    if params['crossovermethod'] == 'segmented':
        if 'swapstopprob' in params:
            try:
                stop_crossover_prob = float(params['swapstopprob'])
                if 0 <= stop_crossover_prob <= 1:
                    params['swapstopprob'] = stop_crossover_prob
                else:
                    print('Probability of ending segmented crossover not '
                          'recognised - please enter a value between 0 and 1')
                    params.pop('swapstopprob')
            except ValueError:
                print('Probability of ending segmented crossover not '
                      'recognised - please enter a value between 0 and 1')
                params.pop('swapstopprob')

        if not 'swapstopprob' in params:
            stop_crossover_prob = ''
            while not type(stop_crossover_prob) == float:
                print('Specify probability of ending crossover:')
                stop_crossover_prob = input(prompt)

                try:
                    stop_crossover_prob = float(stop_crossover_prob)
                    if 0 <= stop_crossover_prob <= 1:
                        params['swapstopprob'] = stop_crossover_prob
                        break
                    else:
                        print('Probability of ending segmented crossover not '
                              'recognised - please enter a value between 0 '
                              'and 1')
                        stop_crossover_prob = ''
                except ValueError:
                    print('Probability of ending segmented crossover not '
                          'recognised - please enter a value between 0 and 1')
                    stop_crossover_prob = ''

    else:
        if 'swapstopprob' in params:
            params.pop('swapstopprob')
    """

    # Defines method used to mutate children sequences (generated in the
    # previous step from parent crossover)
    if 'mutationmethod' in params:
        if not params['mutationmethod'] in ['swap', 'scramble']:
            print('Mutation method not recognised - please select one of '
                  '"swap" or "scramble"')
            params.pop('mutationmethod')

    if not 'mutationmethod' in params:
        method_mutation = ''
        while not method_mutation in ['swap', 'scramble']:
            print('Specify mutation method:')
            method_mutation = input(prompt).lower().replace(' ', '')

            if method_mutation in ['swap', 'scramble']:
                params['mutationmethod'] = method_mutation
                break
            else:
                print('Mutation method not recognised - please select one of '
                      '"swap" or "scramble"')

    # Defines probability of mutation of each node in the network
    # Commented out because this parameter has been selected to be optimised
    # with hyperopt
    if 'mutationprob' in params:
        params.pop('mutationprob')
    """
    if 'mutationprob' in params:
        try:
            mutation_probability = float(params['mutationprob'])
            if 0 <= mutation_probability <= 1:
                params['mutationprob'] = mutation_probability
            else:
                print('Probability of mutation not recognised - please enter '
                      'a value between 0 and 1')
                params.pop('mutationprob')
        except ValueError:
            print('Probability of mutation not recognised - please enter a '
                  'value between 0 and 1')
            params.pop('mutationprob')

    if not 'mutationprob' in params:
        mutation_probability = ''
        while not type(mutation_probability) == float:
            print('Specify probability of mutation:')
            mutation_probability = input(prompt)

            try:
                mutation_probability = float(mutation_probability)
                if 0 <= mutation_probability <= 1:
                    params['mutationprob'] = mutation_probability
                    break
                else:
                    print('Probability of mutation not recognised - please '
                          'enter a value between 0 and 1')
                    mutation_probability = ''
            except ValueError:
                print('Probability of mutation not recognised - please enter '
                      'a value between 0 and 1')
                mutation_probability = ''
    """

    # Defines the size of the population of sequences to be optimised by the
    # genetic algorithm. The population size should be an even number, in order
    # that all parent sequences can be paired off for crossover (mating).
    # N.B. must be defined after fitnessscoremethod
    # TO BE CHECKED!
    if 'populationsize' in params:
        try:
            new_population_size = int(params['populationsize'])
            if params['fitnessscoremethod'] == 'split':
                population_fraction = (
                    new_population_size * 0.5 * params['splitfraction']
                )
            else:
                population_fraction = ''

            if (
                    params['fitnessscoremethod'] != 'split'
                and str(new_population_size) == params['populationsize']
                and new_population_size > 0
                and new_population_size % 2 == 0
            ):
                params['populationsize'] = new_population_size
            elif (
                    params['fitnessscoremethod'] == 'split'
                and str(new_population_size) == params['populationsize']
                and new_population_size > 0
                and float(population_fraction).is_integer()
            ):
                params['populationsize'] = new_population_size
            else:
                if params['fitnessscoremethod'] != 'split':
                    print('Population size not recognised - please enter a '
                          'positive even integer')
                else:
                    print('Population size not recognised - please enter a '
                          'positive integer that when multiplied by 2x the '
                          'fraction of samples to be optimised against '
                          'propensity gives an integer value')
                params.pop('populationsize')
        except ValueError:
            if params['fitnessscoremethod'] != 'split':
                print('Population size not recognised - please enter a '
                      'positive even integer')
            else:
                print('Population size not recognised - please enter a '
                      'positive integer that when multiplied by 2x the '
                      'fraction of samples to be optimised against propensity '
                      'gives an integer value')
            params.pop('populationsize')

    if not 'populationsize' in params:
        population_size = ''
        while type(population_size) != int:
            print('Specify number of sequences in population:')
            population_size = input(prompt).strip()

            try:
                new_population_size = int(population_size)
                if params['fitnessscoremethod'] == 'split':
                    population_fraction = (
                        new_population_size * 0.5 * params['splitfraction']
                    )
                else:
                    population_fraction = ''

                if (
                        params['fitnessscoremethod'] != 'split'
                    and str(new_population_size) == population_size
                    and new_population_size > 0
                    and new_population_size % 2 == 0
                ):
                    params['populationsize'] = new_population_size
                    break
                elif (
                        params['fitnessscoremethod'] == 'split'
                    and str(new_population_size) == population_size
                    and new_population_size > 0
                    and float(population_fraction).is_integer()
                ):
                    params['populationsize'] = new_population_size
                    break
                else:
                    if params['fitnessscoremethod'] != 'split':
                        print('Population size not recognised - please enter '
                              'a positive even integer')
                    elif params['fitnessscoremethod'] == 'split':
                        print('Population size not recognised - please enter a '
                              'positive integer that when multiplied by 2x the '
                              'fraction of samples to be optimised against '
                              'propensity gives an integer value')
                    population_size = ''
            except ValueError:
                if params['fitnessscoremethod'] != 'split':
                    print('Population size not recognised - please enter a '
                          'positive even integer')
                else:
                    print('Population size not recognised - please enter a '
                          'positive integer that when multiplied by 2x the '
                          'fraction of samples to be optimised against '
                          'propensity gives an integer value')
                population_size = ''

    # Defines the number of generations for which to run the genetic algorithm
    if 'maxnumgenerations' in params:
        try:
            new_num_gens = int(params['maxnumgenerations'])
            if (
                    str(new_num_gens) == params['maxnumgenerations']  # Checks
                    # that value provided in input file is an integer
                and new_num_gens > 0
            ):
                params['maxnumgenerations'] = new_num_gens
            else:
                print('Maximum number of generations not recognised - please '
                      'enter a positive integer')
                params.pop('maxnumgenerations')
        except ValueError:
            print('Maximum number of generations not recognised - please enter '
                  'a positive integer')
            params.pop('maxnumgenerations')

    if not 'maxnumgenerations' in params:
        num_gens = ''
        while type(num_gens) != int:
            print('Specify maximum number of generations for which to run GA:')
            num_gens = input(prompt)

            try:
                new_num_gens = int(num_gens)
                if (
                        str(new_num_gens) == num_gens
                    and new_num_gens > 0
                ):
                    params['maxnumgenerations'] = new_num_gens
                    break
                else:
                    print('Maximum number of generations not recognised - '
                          'please enter a positive integer')
                    num_gens = ''
            except ValueError:
                print('Maximum number of generations not recognised - please '
                      'enter a positive integer')
                num_gens = ''

    # Changes directory to user-specified "working directory" and copies across
    # necessary input files in preparation for running the genetic algorithm
    params['workingdirectory'] = '{}BetaDesigner_results/{}'.format(
        params['workingdirectory'], params['jobid']
    )
    if not os.path.isdir(params['workingdirectory']):
        os.mkdir(params['workingdirectory'])

    if os.path.isdir('{}'.format(params['workingdirectory'])):
        print('Directory {} already exists'.format(params['workingdirectory']))
        delete_dir = ''

        while not delete_dir in ['yes', 'y', 'no', 'n']:
            print('Delete {}?'.format(params['workingdirectory']))
            delete_dir = input(prompt).lower()

            if delete_dir in ['yes', 'y']:
                shutil.rmtree('{}'.format(params['workingdirectory']))
                os.mkdir('{}'.format(params['workingdirectory']))
                break
            elif delete_dir in ['no', 'n']:
                raise OSError(
                    'Exiting BetaDesigner - please provide a jobid that is not '
                    'already a directory in {}/ for future '
                    'runs'.format(''.join(params['workingdirectory'].split('/')[:-1]))
                )
            else:
                print('Input not recognised - please specify "yes" or "no"')
                delete_dir = ''

    # Unpickles input dataframe generated by DataGen
    input_df = pd.read_pickle(params['inputdataframepath'])
    if params['phipsiclustercoords']:
        params['inputdataframe'] = calc_parent_voronoi_cluster(
            input_df, params['phipsiclustercoords']
        )

    return params


def setup_input_output(params):
    """
    Creates directories for input and output data and copies the necessary
    input files across to the input directory
    """

    os.makedirs('{}/Program_input'.format(params['workingdirectory']))
    os.makedirs('{}/Program_output'.format(params['workingdirectory']))

    shutil.copy('{}'.format(params['inputdataframepath']),
                '{}/Program_input/Input_DataFrame.pkl'.format(params['workingdirectory']))
    shutil.copy('{}'.format(params['inputpdb']),
                '{}/Program_input/Input_PDB.pdb'.format(params['workingdirectory']))
    with open('{}/Program_input/Propensity_scales.pkl'.format(
        params['workingdirectory']), 'wb') as pickle_file:
        pickle.dump((params['propensityscales']), pickle_file)
    if params['frequencyscales']:
        with open('{}/Program_input/Frequency_scales.pkl'.format(
            params['workingdirectory']), 'wb') as pickle_file:
            pickle.dump((params['frequencyscales']), pickle_file)
    if 'phipsiclustercoords' in list(params.keys()):
        with open('{}/Program_input/Ramachandran_voronoi_cluster_coords.pkl'.format(
            params['workingdirectory']), 'wb') as pickle_file:
            pickle.dump((params['phipsiclustercoords']), pickle_file)

    # Writes program params to a txt file for user records
    with open('{}/Program_input/BetaDesigner_params.txt'.format(
        params['workingdirectory']), 'w') as f:
        for key, parameter in params.items():
            f.write('{}: {}\n'.format(key, parameter))


class initialise_ga_object():

    def __init__(self, params, test=False):
        aa_code_dict = gen_amino_acids_dict()
        if params['barrelorsandwich'] == '2.40':
            aa_code_dict.pop('CYS')
        params['aacodes'] = aa_code_dict.values()

        self.input_df = params['inputdataframe']
        self.input_pdb = params['inputpdb']
        self.propensity_dicts = params['propensityscales']
        self.frequency_dicts = params['frequencyscales']
        self.aa_list = params['aacodes']
        self.dict_weights = params['scaleweights']
        self.dict_name_indices = params['dictnameindices']
        # self.propensity_weight = np.nan  # params['propvsfreqweight']  To be optimised with hyperopt
        self.working_directory = params['workingdirectory']
        self.barrel_or_sandwich = params['barrelorsandwich']
        self.job_id = params['jobid']
        self.method_initial_side_chains = params['initialseqmethod']
        self.method_fitness_score = params['fitnessscoremethod']
        if self.method_fitness_score == 'split':
            self.split_fraction = params['splitfraction']
        self.method_select_mating_pop = params['matingpopmethod']
        # if self.method_select_mating_pop == 'fittest':
        #    self.unfit_fraction = np.nan  # params['unfitfraction']  To be optimised with hyperopt
        self.method_crossover = params['crossovermethod']
        # if self.method_crossover == 'uniform':
        #    self.crossover_prob = np.nan  # params['crossoverprob']  To be optimised with hyperopt
        if self.method_crossover == 'segmented':  # Change to "if" if re-comment out lines above
            self.swap_start_prob = params['swapstartprob']
            self.swap_stop_prob = params['swapstopprob']
        self.method_mutation = params['mutationmethod']
        # self.mutation_prob = np.nan  params['mutationprob']  To be optimised with hyperopt
        self.pop_size = params['populationsize']
        if self.method_fitness_score == 'split':
            params['propensitypopsize'] = self.pop_size*self.split_fraction
            self.propensity_pop_size = params['propensitypopsize']
        self.num_gens = params['maxnumgenerations']
        self.test = test

        if self.test is False:
            with open('{}/Program_input/Input_parameters.pkl'.format(
                params['workingdirectory']), 'wb') as f:
                pickle.dump((params), f)
