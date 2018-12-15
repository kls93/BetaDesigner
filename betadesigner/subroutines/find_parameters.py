
import os
import pickle
import random
import shutil
import string
import numpy as np
import pandas as pd
from collections import OrderedDict

if __name__ == 'subroutines.find_parameters':
    from subroutines.variables import gen_amino_acids_dict
else:
    from betadesigner.subroutines.variables import gen_amino_acids_dict

prompt = '> '


def calc_parent_voronoi_cluster(input_df, cluster_coords):
    # Calculates to which discrete bins in Ramachandran (phi psi) space the
    # residues in the input structure belong
    phi_psi_list = ['']*input_df.shape[0]

    if cluster_coords != '':
        for row in range(input_df.shape[0]):
            phi_psi = np.array([input_df['phi'][row], input_df['psi'][row]])
            distances = np.sqrt(np.sum(np.square(cluster_coords-phi_psi), axis=1))
            voronoi_index = np.abs(distances).argmin()
            phi_psi_list[row] = voronoi_index

    phi_psi_class_df = pd.DataFrame({'phi_psi_class': phi_psi_list})
    input_df = pd.concat([input_df, phi_psi_class_df], axis=1)

    return input_df


def find_parameters(args):
    # Defines program parameter values. If an input file is provided, the code
    # first tries to extract program parameters from this file. It then
    # requests user input for all remaining undefined parameters and any
    # defined parameters with unrecognised values.
    parameters = OrderedDict()

    if not vars(args)['input_file'] is None:
        try:
            with open(vars(args)['input_file'], 'r') as f:
                for line in f.readlines():
                    key = line.split(':')[0].replace(' ', '').lower()
                    value = line.split(':')[1].replace('\n', '').strip()

                    if key in ['inputdataframe', 'inputpdb', 'propensityscales',
                               'frequencyscales', 'scaleweights',
                               'phipsiclustercoords']:
                        value = value.replace('\\', '/')  # For windows file paths
                        value = '/{}'.format(value.strip('/'))
                    elif key in ['workingdirectory']:
                        value = value.replace('\\', '/')  # For windows file paths
                        value = '/{}/'.format(value.strip('/'))
                    elif key in ['jobid']:
                        value = value.replace(' ', '')
                    elif key in ['propvsfreqweight', 'barrelorsandwich',
                                 'initialseqmethod', 'fitnessscoremethod',
                                 'splitfraction', 'matingpopmethod',
                                 'unfitfraction' 'crossovermethod',
                                 'crossoverprob', 'swapstartprob',
                                 'swapstopprob', 'mutationmethod',
                                 'populationsize', 'numberofgenerations']:
                        value = value.lower().replace(' ', '')

                    parameters[key] = value

        except FileNotFoundError:
            print('Path to input file not recognised')

    # Defines absolute file path to input dataframe
    if 'inputdataframe' in parameters:
        if (
            (not os.path.isfile(parameters['inputdataframe']))
            or
            (not parameters['inputdataframe'].endswith('.pkl'))
        ):
            print('File path to pickled input dataframe not recognised')
            parameters.pop('inputdataframe')

    if not 'inputdataframe' in parameters:
        input_df = ''
        while (
            (not os.path.isfile(input_df))
            or
            (not input_df.endswith('.pkl'))
        ):
            print('Specify absolute file path of pickled input dataframe:')
            input_df = input(prompt)

            if os.path.isfile(input_df) and input_df.endswith('.pkl'):
                parameters['inputdataframe'] = input_df
                break
            else:
                print('File path to pickled input dataframe not recognised')

    # Defines absolute file path to input PDB file (the file fed into DataGen)
    if 'inputpdb' in parameters:
        if (
            (not os.path.isfile(parameters['inputpdb']))
            or
            (not parameters['inputpdb'].endswith('.pdb'))
        ):
            print('File path to input PDB file not recognised')
            parameters.pop('inputpdb')

    if not 'inputpdb' in parameters:
        input_pdb = ''
        while (
            (not os.path.isfile(input_pdb))
            or
            (not input_pdb.endswith('.pdb'))
        ):
            print('Specify absolute file path of input PDB file:')
            input_pdb = input(prompt)

            if os.path.isfile(input_pdb) and input_pdb.endswith('.pdb'):
                parameters['inputpdb'] = input_pdb
                break
            else:
                print('File path to input PDB file not recognised')

    # Defines absolute file path to pickle file listing propensity scales
    if 'propensityscales' in parameters:
        if (
            (not os.path.isfile(parameters['propensityscales']))
            or
            (not parameters['propensityscales'].endswith('.pkl'))
        ):
            print('File path to pickled propensity scales not recognised')
            parameters.pop('propensityscales')
        else:
            with open(parameters['propensityscales'], 'rb') as pickle_file:
                propensity_scales_dict = pickle.load(pickle_file)
            parameters['propensityscales'] = propensity_scales_dict

    if not 'propensityscales' in parameters:
        propensity_scales_file = ''
        while (
            (not os.path.isfile(propensity_scales_file))
            or
            (not propensity_scales_file.endswith('.pkl'))
        ):
            print('Specify absolute file path of pickled propensity scales:')
            propensity_scales_file = input(prompt)

            if (
                (os.path.isfile(propensity_scales_file))
                and
                (propensity_scales_file.endswith('.pkl'))
            ):
                with open(propensity_scales_file, 'rb') as pickle_file:
                    propensity_scales_dict = pickle.load(pickle_file)
                parameters['propensityscales'] = propensity_scales_dict
                break
            else:
                print('File path to pickled propensity scales not recognised')

    # Defines absolute file path to pickle file listing frequency scales
    if 'frequencyscales' in parameters:
        if (
            (not os.path.isfile(parameters['frequencyscales']))
            or
            (not parameters['frequencyscales'].endswith('.pkl'))
        ):
            print('File path to pickled frequency scales not recognised')
            parameters.pop('frequencyscales')
        else:
            with open(parameters['frequencyscales'], 'rb') as pickle_file:
                frequency_scales_dict = pickle.load(pickle_file)
            parameters['frequencyscales'] = frequency_scales_dict

    if not 'frequencyscales' in parameters:
        print('Include frequency scales?')
        frequency_input = input(prompt)

        while not frequency_input in ['yes', 'no', 'y', 'n']:
            print('User input not recognised - please specify ("yes" or "no") '
                  'whether you would like to include frequency scales:')
            frequency_input = input(prompt)

        if frequency_input in ['yes', 'y']:
            frequency_scales_file = ''

            while (
                (not os.path.isfile(frequency_scales_file))
                or
                (not frequency_scales_file.endswith('.pkl'))
            ):
                print('Specify absolute file path of pickled frequency scales:')
                frequency_scales_file = input(prompt)

                if (
                    (os.path.isfile(frequency_scales_file))
                    and
                    (frequency_scales_file.endswith('.pkl'))
                ):
                    with open(frequency_scales_file, 'rb') as pickle_file:
                        frequency_scales_dict = pickle.load(pickle_file)
                    parameters['frequencyscales'] = frequency_scales_dict
                    break
                else:
                    print('File path to pickled frequency scales not recognised')
        else:
            parameters['frequencyscales'] = {}

    # Defines propensity scale weights
    scales = (  list(parameters['propensityscales'].keys())
              + list(parameters['frequencyscales'].keys()))
    if 'scaleweights' in parameters:
        if parameters['scaleweights'] == 'equal':
            scale_weights = {}
            for dict_name in scales:
                if dict_name.split('_')[5] == 'indv':
                    scale_weights[dict_name] = 1
                elif dict_name.split('_')[5] == 'pair':
                    scale_weights[dict_name] = 0.5
            parameters['scaleweights'] = scale_weights
        elif (
            (os.path.isfile(parameters['scaleweights']))
            and
            (parameters['scaleweights'].endswith('.pkl'))
        ):
            with open(parameters['scaleweights'], 'rb') as pickle_file:
                scale_weights = pickle.load(pickle_file)
                if type(scale_weights) == dict:
                    parameters['scaleweights'] = scale_weights
                else:
                    print('Propensity scale weights dictionary not recognised')
                    parameters['scaleweights'] = {}
        else:
            print('Propensity scale weights not provided')
            parameters['scaleweights'] = {}
    else:
        parameters['scaleweights'] = {}

    for dict_name in scales:
        if not dict_name in parameters['scaleweights']:
            print('Weight for {} not provided'.format(dict_name))
            scale_weight = ''
        else:
            scale_weight = parameters['scaleweights'][dict_name]

        while not type(scale_weight) in [int, float]:
            print('Weighting for {} not recognised.\n'
                  'Specify weight for {}'.format(dict_name, dict_name))
            scale_weight = input(prompt)

            try:
                scale_weight = float(scale_weight)
                parameters['scaleweights'][dict_name] = scale_weight
                break
            except ValueError:
                scale_weight = ''

    # Defines weighting between propensity and frequency scales
    if 'propvsfreqweight' in parameters:
        try:
            prop_freq_weight = float(parameters['propvsfreqweight'])
            if 0 <= prop_freq_weight <= 1:
                parameters['propvsfreqweight'] = {'propensity': prop_freq_weight,
                                                  'frequency': 1-prop_freq_weight}
            else:
                print('Weighting for propensity scales not recognised - please '
                      'enter a value between 0 and 1')
                parameters.pop('propvsfreqweight')
        except ValueError:
            print('Weighting for propensity scales not recognised - please '
                  'enter a value between 0 and 1')
            parameters.pop('propvsfreqweight')

    if not 'propvsfreqweight' in parameters:
        if parameters['frequencyscales'] == {}:
            parameters['propvsfreqweight'] = 1
        else:
            prop_freq_weight = ''
            while not type(prop_freq_weight) == float:
                print('Specify weight for propensity scales:')
                prop_freq_weight = input(prompt)

                try:
                    prop_freq_weight = float(prop_freq_weight)
                    if 0 <= prop_freq_weight <= 1:
                        parameters['propvsfreqweight'] = {
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

    # Calculates phi and psi classes if discrete phi / psi dict is input
    for dict_label in scales:
        dict_label = dict_label.split('_')

        if all(x in dict_label for x in ['phi', 'psi', 'disc']):
            if parameters['phipsiclustercoords']:
                if (
                    (not os.path.isfile(parameters['phipsiclustercoords']))
                    or
                    (not parameters['phipsiclustercoords'].endswith('.pkl'))
                ):
                    print('File path to pickled phi / psi voronoi point '
                          'coordinates not recognised')
                    parameters.pop('phipsiclustercoords')
                else:
                    with open(parameters['phipsiclustercoords'], 'rb') as pickle_file:
                        cluster_coords = pickle.load(pickle_file)
                    parameters['phipsiclustercoords'] = cluster_coords

            if not 'phipsiclustercoords' in parameters:
                cluster_coords_file = ''
                while (
                    (not os.path.isfile(cluster_coords_file))
                    or
                    (not cluster_coords_file.endswith('.pkl'))
                ):
                    print('Specify absolute file path of pickled phi / psi '
                          'voronoi point coordinates:')
                    cluster_coords_file = input(prompt)

                    if (
                        (os.path.isfile(cluster_coords_file))
                        and
                        (cluster_coords_file.endswith('.pkl'))
                    ):
                        with open(cluster_coords_file, 'rb') as pickle_file:
                            cluster_coords = pickle.load(pickle_file)
                        parameters['phipsiclustercoords'] = cluster_coords
                        break
                    else:
                        print('File path to pickled phi / psi voronoi point '
                              'coordinates not recognised')
            break

    # Checks whether for loop has been broken
    else:
        parameters['phipsiclustercoords'] = ''

    # Defines working directory
    if 'workingdirectory' in parameters:
        if not os.path.isdir(parameters['workingdirectory']):
            print('File path to working directory not recognised')
            parameters.pop('workingdirectory')

    if not 'workingdirectory' in parameters:
        working_directory = ''
        while not os.path.isdir(working_directory):
            print('Specify absolute path of working directory')
            working_directory = input(prompt)

            if os.path.isdir(working_directory):
                parameters['workingdirectory'] = working_directory
                break
            else:
                print('File path to working directory not recognised')

    # Defines whether the input structure is a beta-sandwich or a beta-barrel
    # backbone
    if 'barrelorsandwich' in parameters:
        if parameters['barrelorsandwich'] == 'barrel':
            parameters['barrelorsandwich'] = '2.40'
        elif parameters['barrelorsandwich'] == 'sandwich':
            parameters['barrelorsandwich'] = '2.60'

        if not parameters['barrelorsandwich'] in ['2.40', '2.60']:
            print('Backbone structure not recognised')
            parameters.pop('barrelorsandwich')

    if not 'barrelorsandwich' in parameters:
        barrel_or_sandwich = ''
        while not barrel_or_sandwich in ['barrel', '2.40', 'sandwich', '2.60']:
            print('Specify structure type - please enter "barrel" or "sandwich":')
            barrel_or_sandwich = input(prompt).lower().replace(' ', '')

            if barrel_or_sandwich in ['2.40', '2.60']:
                parameters['barrelorsandwich'] = barrel_or_sandwich
                break
            elif barrel_or_sandwich in ['barrel', 'sandwich']:
                if barrel_or_sandwich == 'barrel':
                    parameters['barrelorsandwich'] = '2.40'
                elif barrel_or_sandwich == 'sandwich':
                    parameters['barrelorsandwich'] = '2.60'
                break
            else:
                print('Structure type not recognised')

    # Assigns unique identification code to job
    if not 'jobid' in parameters:
        print('Specify unique ID for input structure (if you would like '
              'BetaDesigner to assign a random ID, enter "random")):')
        job_id = input(prompt)
        if job_id.lower() == 'random':
            job_id = ''.join([random.choice(string.ascii_letters + string.digits)
                              for i in range(6)])
        parameters['jobid'] = job_id

    # Defines method used to generate initial sequences for backbone structure
    if 'initialseqmethod' in parameters:
        if not parameters['initialseqmethod'] in [
            'random', 'rawpropensity', 'rankpropensity'
        ]:
            print('Method for determining initial side chain assignments not '
                  'recognised - please select one of "random", "rawpropensity" '
                  'or "rankpropensity"')
            parameters.pop('initialseqmethod')

    if not 'initialseqmethod' in parameters:
        method_initial_side_chains = ''
        while not method_initial_side_chains in [
            'random', 'rawpropensity', 'rankpropensity'
        ]:
            print('Specify method for determining initial side chain assignments:')
            method_initial_side_chains = input(prompt).lower().replace(' ', '')

            if method_initial_side_chains in [
                'random', 'rawpropensity', 'rankpropensity'
            ]:
                parameters['initialseqmethod'] = method_initial_side_chains
                break
            else:
                print('Method not recognised - please select one of "random", '
                      '"rawpropensity" or "rankpropensity"')

    # Defines method used to measure sequence fitness
    if 'fitnessscoremethod' in parameters:
        if not parameters['fitnessscoremethod'] in [
            'propensity', 'allatom', 'alternate', 'split'
        ]:
            print('Method for measuring sequence fitness not recognised - '
                  'please select one of "propensity", "all-atom", "alternate" '
                  'or "split"')
            parameters.pop('fitnessscoremethod')

    if not 'fitnessscoremethod' in parameters:
        method_fitness_score = ''
        while not method_fitness_score in [
            'propensity', 'allatom', 'alternate', 'split'
        ]:
            print('Specify method for measuring sequence fitnesses:')
            method_fitness_score = input(prompt).lower().replace(' ', '')

            if method_fitness_score in [
                'propensity', 'allatom', 'alternate', 'split'
            ]:
                parameters['fitnessscoremethod'] = method_fitness_score
                break
            else:
                print('Method not recognised - please select one of '
                      '"propensity", "all-atom", "alternate", "split"')

    # Defines fraction of samples to be optimised against propensity in each
    # generation of the genetic algorithm
    if parameters['fitnessscoremethod'] == 'split':
        if 'splitfraction' in parameters:
            try:
                split_fraction = float(parameters['splitfraction'])
                if 0 <= split_fraction <= 1:
                    parameters['splitfraction'] = split_fraction
                else:
                    print('Fraction of samples to be optimised against '
                          'propensity not recognised - please enter a value '
                          'between 0 and 1')
                    parameters.pop('splitfraction')
            except ValueError:
                print('Fraction of samples to be optimised against propensity '
                      'not recognised - please enter a value between 0 and 1')
                parameters.pop('splitfraction')

        if not 'splitfraction' in parameters:
            split_fraction = ''
            while not type(split_fraction) == float:
                print('Specify fraction of samples to be optimised against '
                      'propensity:')
                split_fraction = input(prompt)

                try:
                    split_fraction = float(split_fraction)
                    if 0 <= split_fraction <= 1:
                        parameters['splitfraction'] = split_fraction
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

    # Defines method used to select a population of individuals for mating
    if 'matingpopmethod' in parameters:
        if not parameters['matingpopmethod'] in [
            'fittest', 'roulettewheel', 'rankroulettewheel'
        ]:
            print('Method for genertaing mating population not recognised - '
                  'please select one of "fittest", "roulettewheel" or '
                  '"rankroulettewheel"')
            parameters.pop('matingpopmethod')

    if not 'matingpopmethod' in parameters:
        method_select_mating_pop = ''
        while not method_select_mating_pop in [
            'fittest', 'roulettewheel', 'rankroulettewheel'
        ]:
            print('Specify method for generating mating population:')
            method_select_mating_pop = input(prompt).lower().replace(' ', '')

            if method_select_mating_pop in [
                'fittest', 'roulettewheel', 'rankroulettewheel'
            ]:
                parameters['matingpopmethod'] = method_select_mating_pop
                break
            else:
                print('Method not recognised - please select one of "fittest", '
                      '"roulettewheel" or "rankroulettewheel"')

    # Defines fraction of unfit sequences to be included in the mating
    # population at each generation of the genetic algorithm if
    # self.method_select_mating_pop == 'fittest'
    if parameters['matingpopmethod'] == 'fittest':
        if 'unfitfraction' in parameters:
            try:
                unfit_fraction = float(parameters['unfitfraction'])
                if 0 <= unfit_fraction <= 1:
                    parameters['unfitfraction'] = unfit_fraction
                else:
                    print('Fraction of mating population to be comprised of '
                          'unfit samples not recognised - please enter a '
                          'value between 0 and 1')
                    parameters.pop('unfitfraction')
            except ValueError:
                print('Fraction of mating population to be comprised of unfit '
                      'samples not recognised - please enter a value between '
                      '0 and 1')
                parameters.pop('unfitfraction')

        if not 'unfitfraction' in parameters:
            unfit_fraction = ''
            while not type(unfit_fraction) == float:
                print('Specify fraction of mating population to be comprised '
                      'of unfit samples:')
                unfit_fraction = input(prompt)

                try:
                    unfit_fraction = float(unfit_fraction)
                    if 0 <= unfit_fraction <= 1:
                        parameters['unfitfraction'] = unfit_fraction
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

    # Defines method used to crossover parent sequences to generate children
    if 'crossovermethod' in parameters:
        if not parameters['crossovermethod'] in ['uniform', 'segmented']:
            print('Crossover method not recognised - please select one of '
                  '"uniform" or "segmented"')
            parameters.pop('crossovermethod')

    if not 'crossovermethod' in parameters:
        method_crossover = ''
        while not method_crossover in ['uniform', 'segmented']:
            print('Specify crossover method:')
            method_crossover = input(prompt).lower().replace(' ', '')

            if method_crossover in ['uniform', 'segmented']:
                parameters['crossovermethod'] = method_crossover
                break
            else:
                print('Crossover method not recognised - please select one of '
                      '"uniform" or "segmented"')

    # Defines probability of exchanging amino acid identities for each node in
    # the network as part of a uniform crossover
    if parameters['crossovermethod'] == 'uniform':
        if 'crossoverprob' in parameters:
            try:
                crossover_probability = float(parameters['crossoverprob'])
                if 0 <= crossover_probability <= 1:
                    parameters['crossoverprob'] = crossover_probability
                else:
                    print('Probability of uniform crossover not recognised - '
                          'please enter a value between 0 and 1')
                    parameters.pop('crossoverprob')
            except ValueError:
                print('Probability of uniform crossover not recognised - '
                      'please enter a value between 0 and 1')
                parameters.pop('crossoverprob')

        if not 'crossoverprob' in parameters:
            crossover_probability = ''
            while not type(crossover_probability) == float:
                print('Specify probability of uniform crossover:')
                crossover_probability = input(prompt)

                try:
                    crossover_probability = float(crossover_probability)
                    if 0 <= crossover_probability <= 1:
                        parameters['crossoverprob'] = crossover_probability
                        break
                    else:
                        print('Probability of uniform crossover not recognised '
                              '- please enter a value between 0 and 1')
                        crossover_probability = ''
                except ValueError:
                    print('Probability of uniform crossover not recognised - '
                          'please enter a value between 0 and 1')
                    crossover_probability = ''

    # Defines probability of starting a (segmented) crossover
    if parameters['crossovermethod'] == 'segmented':
        if 'swapstartprob' in parameters:
            try:
                start_crossover_prob = float(parameters['swapstartprob'])
                if 0 <= start_crossover_prob <= 1:
                    parameters['swapstartprob'] = start_crossover_prob
                else:
                    print('Probability of initiating segmented crossover not '
                          'recognised - please enter a value between 0 and 1')
                    parameters.pop('swapstartprob')
            except ValueError:
                print('Probability of initiating segmented crossover not '
                      'recognised - please enter a value between 0 and 1')
                parameters.pop('swapstartprob')

        if not 'swapstartprob' in parameters:
            start_crossover_prob = ''
            while not type(start_crossover_prob) == float:
                print('Specify probability of initiating crossover:')
                start_crossover_prob = input(prompt)

                try:
                    start_crossover_prob = float(start_crossover_prob)
                    if 0 <= start_crossover_prob <= 1:
                        parameters['swapstartprob'] = start_crossover_prob
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

    # Defines probability of stopping a (segmented) crossover
    if parameters['crossovermethod'] == 'segmented':
        if 'swapstopprob' in parameters:
            try:
                stop_crossover_prob = float(parameters['swapstopprob'])
                if 0 <= stop_crossover_prob <= 1:
                    parameters['swapstopprob'] = stop_crossover_prob
                else:
                    print('Probability of ending segmented crossover not '
                          'recognised - please enter a value between 0 and 1')
                    parameters.pop('swapstopprob')
            except ValueError:
                print('Probability of ending segmented crossover not '
                      'recognised - please enter a value between 0 and 1')
                parameters.pop('swapstopprob')

        if not 'swapstopprob' in parameters:
            stop_crossover_prob = ''
            while not type(stop_crossover_prob) == float:
                print('Specify probability of ending crossover:')
                stop_crossover_prob = input(prompt)

                try:
                    stop_crossover_prob = float(stop_crossover_prob)
                    if 0 <= stop_crossover_prob <= 1:
                        parameters['swapstopprob'] = stop_crossover_prob
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

    # Defines method used to mutate children sequences (generated in the
    # previous step from parent crossover)
    if 'mutationmethod' in parameters:
        if not parameters['mutationmethod'] in ['swap', 'scramble']:
            print('Mutation method not recognised - please select one of '
                  '"swap" or "scramble"')
            parameters.pop('mutationmethod')

    if not 'mutationmethod' in parameters:
        method_mutation = ''
        while not method_mutation in ['swap', 'scramble']:
            print('Specify mutation method:')
            method_mutation = input(prompt).lower().replace(' ', '')

            if method_mutation in ['swap', 'scramble']:
                parameters['mutationmethod'] = method_mutation
                break
            else:
                print('Mutation method not recognised - please select one of '
                      '"swap" or "scramble"')

    # Defines probability of mutation of each node in the network
    if 'mutationprob' in parameters:
        try:
            mutation_probability = float(parameters['mutationprob'])
            if 0 <= mutation_probability <= 1:
                parameters['mutationprob'] = mutation_probability
            else:
                print('Probability of mutation not recognised - please enter '
                      'a value between 0 and 1')
                parameters.pop('mutationprob')
        except ValueError:
            print('Probability of mutation not recognised - please enter a '
                  'value between 0 and 1')
            parameters.pop('mutationprob')

    if not 'mutationprob' in parameters:
        mutation_probability = ''
        while not type(mutation_probability) == float:
            print('Specify probability of mutation:')
            mutation_probability = input(prompt)

            try:
                mutation_probability = float(mutation_probability)
                if 0 <= mutation_probability <= 1:
                    parameters['mutationprob'] = mutation_probability
                    break
                else:
                    print('Probability of mutation not recognised - please '
                          'enter a value between 0 and 1')
                    mutation_probability = ''
            except ValueError:
                print('Probability of mutation not recognised - please enter '
                      'a value between 0 and 1')
                mutation_probability = ''

    # Defines the size of the population of sequences to be optimised by the
    # genetic algorithm. The population size should be an even number, in order
    # that all parent sequences can be paired off for crossover (mating).
    # NOTE must be defined after fitnessscoremethod
    if 'populationsize' in parameters:
        try:
            new_population_size = int(parameters['populationsize'])
            if parameters['fitnessscoremethod'] == 'split':
                population_fraction = (
                    new_population_size * 0.5 * parameters['splitfraction']
                )
            else:
                population_fraction = ''

            if (
                    parameters['fitnessscoremethod'] != 'split'
                and str(new_population_size) == parameters['populationsize']
                and new_population_size > 0
                and new_population_size % 2 == 0
            ):
                parameters['populationsize'] = new_population_size
            elif (
                    parameters['fitnessscoremethod'] == 'split'
                and str(new_population_size) == parameters['populationsize']
                and new_population_size > 0
                and float(population_fraction).is_integer()
            ):
                parameters['populationsize'] = new_population_size
            else:
                if parameters['fitnessscoremethod'] != 'split':
                    print('Population size not recognised - please enter a '
                          'positive even integer')
                else:
                    print('Population size not recognised - please enter a '
                          'positive integer divisible by 4')
                parameters.pop('populationsize')
        except ValueError:
            if parameters['fitnessscoremethod'] != 'split':
                print('Population size not recognised - please enter a '
                      'positive even integer')
            else:
                print('Population size not recognised - please enter a '
                      'positive integer divisible by 4')
            parameters.pop('populationsize')

    if not 'populationsize' in parameters:
        population_size = ''
        while type(population_size) != int:
            print('Specify number of sequences in population:')
            population_size = input(prompt).strip()

            try:
                new_population_size = int(population_size)
                if parameters['fitnessscoremethod'] == 'split':
                    population_fraction = (
                        new_population_size * 0.5 * parameters['splitfraction']
                    )
                else:
                    population_fraction = ''

                if (
                        parameters['fitnessscoremethod'] != 'split'
                    and str(new_population_size) == population_size
                    and new_population_size > 0
                    and new_population_size % 2 == 0
                ):
                    parameters['populationsize'] = new_population_size
                    break
                elif (
                        parameters['fitnessscoremethod'] == 'split'
                    and str(new_population_size) == population_size
                    and new_population_size > 0
                    and float(population_fraction).is_integer()
                ):
                    parameters['populationsize'] = new_population_size
                    break
                else:
                    if parameters['fitnessscoremethod'] != 'split':
                        print('Population size not recognised - please enter '
                              'a positive even integer')
                    elif parameters['fitnessscoremethod'] == 'split':
                        print('Population size not recognised - please enter '
                              'a positive integer divisible by 4')
                    population_size = ''
            except ValueError:
                if parameters['fitnessscoremethod'] != 'split':
                    print('Population size not recognised - please enter a '
                          'positive even integer')
                else:
                    print('Population size not recognised - please enter a '
                          'positive integer divisible by 4')
                population_size = ''

    # Defines the number of generations for which to run the genetic algorithm
    if 'numberofgenerations' in parameters:
        try:
            new_num_gens = int(parameters['numberofgenerations'])
            if (
                    str(new_num_gens) == parameters['numberofgenerations']
                and new_num_gens > 0
            ):
                parameters['numberofgenerations'] = new_num_gens
            else:
                print('Number of generations not recognised - please enter a '
                      'positive integer')
                parameters.pop('numberofgenerations')
        except ValueError:
            print('Number of generations not recognised - please enter a '
                  'positive integer')
            parameters.pop('numberofgenerations')

    if not 'numberofgenerations' in parameters:
        num_gens = ''
        while type(num_gens) != int:
            print('Specify number of generations for which to run GA:')
            num_gens = input(prompt).strip()

            try:
                new_num_gens = int(num_gens)
                if (
                        str(new_num_gens) == num_gens
                    and new_num_gens > 0
                ):
                    parameters['numberofgenerations'] = new_num_gens
                    break
                else:
                    print('Number of generations not recognised - please '
                          'enter a positive integer')
                    num_gens = ''
            except ValueError:
                print('Number of generations not recognised - please enter a '
                      'positive integer')
                num_gens = ''

    # Changes directory to user-specified "working directory"
    os.chdir(parameters['workingdirectory'])
    if not os.path.isdir('BetaDesigner_results'):
        os.mkdir('BetaDesigner_results')

    working_directory = 'BetaDesigner_results/{}'.format(parameters['jobid'])
    if os.path.isdir(working_directory):
        shutil.rmtree(working_directory)
    os.makedirs('{}/Program_input'.format(working_directory))
    os.makedirs('{}/Program_output'.format(working_directory))
    os.chdir(working_directory)

    shutil.copy('{}'.format(parameters['inputdataframe']),
                'Program_input/Input_DataFrame.pkl')
    shutil.copy('{}'.format(parameters['inputpdb']),
                'Program_input/Input_PDB.pdb')
    with open('Program_input/Propensity_scales.pkl', 'wb') as pickle_file:
        pickle.dump(parameters['propensityscales']), pickle_file)
    if parameters['frequencyscales']:
        with open('Program_input/Frequency_scales.pkl', 'wb') as pickle_file:
            pickle.dump(parameters['frequencyscales']), pickle_file)
    if parameters['phipsiclustercoords']:
        with open('Program_input/Ramachandran_voronoi_cluster_coords.pkl', 'wb') as pickle_file:
            pickle.dump(parameters['phipsiclustercoords']), pickle_file)

    # Writes program parameters to a txt file for user records
    with open('Program_input/BetaDesigner_parameters.txt', 'w') as f:
        for key, parameter in parameters.items():
            f.write('{}: {}\n'.format(key, parameter))

    # Unpickles dataframe
    input_df = pd.read_pickle(parameters['inputdataframe'])
    parameters['inputdataframe'] = calc_parent_voronoi_cluster(
        input_df, parameters['phipsiclustercoords']
    )

    return parameters

class initialise_class():

    def __init__(self, parameters):
        self.parameters = parameters

        self.input_df = parameters['inputdataframe']
        self.input_pdb = parameters['inputpdb']
        self.propensity_dicts = parameters['propensityscales']
        self.frequency_dicts = parameters['frequencyscales']
        self.aa_list = list(gen_amino_acids_dict.values())
        self.dict_weights = parameters['scaleweights']
        self.propensity_weight = parameters['propvsfreqweight']
        self.working_directory = parameters['workingdirectory']
        self.barrel_or_sandwich = parameters['barrelorsandwich']
        self.job_id = parameters['jobid']
        self.method_initial_side_chains = parameters['initialseqmethod']
        self.method_fitness_score = parameters['fitnessscoremethod']
        if self.method_fitness_score == 'split':
            self.split_fraction = parameters['splitfraction']
        self.method_select_mating_pop = parameters['matingpopmethod']
        if self.method_select_mating_pop == 'fittest':
            self.unfit_fraction = parameters['unfitfraction']
        self.method_crossover = parameters['crossovermethod']
        if self.method_crossover == 'uniform':
            self.crossover_prob = parameters['crossoverprob']
        elif self.method_crossover == 'segmented':
            self.swap_start_prob = parameters['swapstartprob']
            self.swap_stop_prob = parameters['swapstopprob']
        self.method_mutation = parameters['mutationmethod']
        self.mutation_prob = parameters['mutationprob']
        self.pop_size = parameters['populationsize']
        if self.method_fitness_score == 'split':
            self.propensity_pop_size = round(self.pop_size*self.split_fraction, 0)
        self.num_gens = parameters['numberofgenerations']
