
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


def def_input_df_path(params):
    """
    Defines absolute file path to input dataframe
    """

    try:
        input_df = params['inputdataframepath']
        if not os.path.isfile(input_df) or not input_df.endswith('.pkl'):
            print('File path to pickled input dataframe not recognised')
            input_df = ''
    except KeyError:
        input_df = ''

    if input_df == '':
        while not os.path.isfile(input_df) or not input_df.endswith('.pkl'):
            print('Specify absolute file path of pickled input dataframe:')
            input_df = '/' + input(prompt).replace('\\', '/').strip('/')

            if os.path.isfile(input_df) and input_df.endswith('.pkl'):
                break
            else:
                print('File path to pickled input dataframe not recognised')

    return input_df


def def_input_df(params):
    """
    Unpickled input dataframe.
    N.B. Must be run AFTER "inputdataframepath" and "phipsiclustercoords" have
    been defined.
    """

    input_df = pd.read_pickle(params['inputdataframepath'])

    if 'phipsiclustercoords' in params.keys():
        input_df = calc_parent_voronoi_cluster(input_df, params['phipsiclustercoords'])

    return input_df


def def_input_pdb(params):
    """
    Defines absolute file path to input PDB file (the file fed into DataGen)
    """

    try:
        input_pdb = params['inputpdb']
        if not os.path.isfile(input_pdb) or not input_pdb.endswith('.pdb'):
            print('File path to input PDB file not recognised')
            input_pdb = ''
    except KeyError:
        input_pdb = ''

    if input_pdb == '':
        while not os.path.isfile(input_pdb) or not input_pdb.endswith('.pdb'):
            print('Specify absolute file path of input PDB file:')
            input_pdb = '/' + input(prompt).replace('\\', '/').strip('/')

            if os.path.isfile(input_pdb) and input_pdb.endswith('.pdb'):
                break
            else:
                print('File path to input PDB file not recognised')

    return input_pdb


def def_propensity_scales(params):
    """
    Defines absolute file path to pickle file listing propensity scales
    """

    try:
        prop_scales = params['propensityscales']
        if not os.path.isfile(prop_scales) or not prop_scales.endswith('.pkl'):
            print('File path to pickled propensity scales not recognised')
            prop_scales = ''
        else:
            with open(prop_scales, 'rb') as pickle_file:
                prop_scales_dict = pickle.load(pickle_file)
            if any(type(prop_scales_dict) == x for x in [dict, OrderedDict]):
                prop_scales = prop_scales_dict
            else:
                print('Data in {} is not a pickled dictionary'.format(prop_scales))
                prop_scales = ''
    except KeyError:
        prop_scales = ''

    if prop_scales == '':
        scales_provided = False
        while scales_provided is False:
            print('Specify absolute file path of pickled propensity scales:')
            prop_scales = '/' + input(prompt).replace('\\', '/').strip('/')

            if os.path.isfile(prop_scales) and prop_scales.endswith('.pkl'):
                with open(prop_scales, 'rb') as pickle_file:
                    prop_scales_dict = pickle.load(pickle_file)
                if any(type(prop_scales_dict) == x for x in [dict, OrderedDict]):
                    prop_scales = prop_scales_dict
                    scales_provided = True
                    break
                else:
                    print('Data in {} is not a pickled dictionary'.format(prop_scales))
            else:
                print('File path to pickled propensity scales not recognised')

    return prop_scales


def def_frequency_scales(params):
    """
    Defines absolute file path to pickle file listing frequency scales
    """

    try:
        freq_scales = params['frequencyscales']
        if not os.path.isfile(freq_scales) or not freq_scales.endswith('.pkl'):
            print('File path to pickled frequency scales not recognised')
            freq_scales = ''
        else:
            with open(freq_scales, 'rb') as pickle_file:
                freq_scales_dict = pickle.load(pickle_file)
            if any(type(freq_scales_dict) == x for x in [dict, OrderedDict]):
                freq_scales = frequency_scales_dict
            else:
                print('Data in {} is not a pickled dictionary'.format(freq_scales))
                freq_scales = ''
    except KeyError:
        freq_scales = ''

    if freq_scales == '':
        print('Include frequency scales?')
        freq_input = input(prompt)

        while not freq_input in ['yes', 'no', 'y', 'n']:
            print('User input not recognised - please specify ("yes" or "no") '
                  'whether you would like to include frequency scales:')
            freq_input = input(prompt).lower()

        if freq_input in ['yes', 'y']:
            freq_scales = ''

            scales_provided = False
            while scales_provided is False:
                print('Specify absolute file path of pickled frequency scales:')
                freq_scales = '/' + input(prompt).replace('\\', '/').strip('/')

                if os.path.isfile(freq_scales) and freq_scales.endswith('.pkl'):
                    with open(freq_scales, 'rb') as pickle_file:
                        freq_scales_dict = pickle.load(pickle_file)
                    if any(type(freq_scales_dict) == x for x in [dict, OrderedDict]):
                        freq_scales = freq_scales_dict
                        scales_provided = True
                        break
                    else:
                        print('Data in {} is not a pickled dictionary'.format(freq_scales))
                else:
                    print('File path to pickled frequency scales not recognised')
        else:
            freq_scales = {}

    return freq_scales


def convert_str_to_dict(dict_string, dict_id, int_or_float):
    """
    Parses a dictionary input in string format to a Python dictionary
    """

    parsed_dict = OrderedDict()

    if len(dict_string) > 1:
        if dict_string[0] == '{' and dict_string[-1] == '}' and ':' in dict_string:
            for key_val in dict_string[1:-1].split(','):
                try:
                    key = key_val.split(':')[0]
                    val = key_val.split(':')[1]
                except IndexError:
                    print('Dictionary describing {} not recognised'.format(dict_id))
                    parsed_dict = {}
                    break

                for x in [' ', '"', '\'']:
                    key = key.replace(x, '')
                    val = val.replace(x, '')

                val_error = False
                if int_or_float == int:
                    try:
                        num_val = int(val)
                    except ValueError:
                        val_error = True
                    if str(num_val) != val:
                        val_error = True
                elif int_or_float == float:
                    try:
                        num_val = float(val)
                    except ValueError:
                        val_error = True
                else:
                    raise Exception(
                        'Unrecognised number format {} - please specify as int'
                        ' or float'.format(int_or_float)
                    )

                if val_error is True:
                    print('Non-{} value provided for key {} in {} dictionary '
                          'describing {}'.format(int_or_float, key, dict_id))
                    parsed_dict = {}
                    break
                else:
                    parsed_dict[key] = num_val

        else:
            print('Dictionary describing {} not recognised'.format(dict_id))

    else:
        print('Dictionary describing propensity / frequency scale naming '
              'convention not recognised')

    return parsed_dict


def def_dict_naming_scheme(params):
    """
    Defines order of properties in names of input propensity / frequency
    scales. For now this is fixed in the order below, update as necessary if
    the current naming convention changes.
    """

    try:
        indices = params['dictnameindices']
    except KeyError:
        indices = ''

    indices_dict = convert_str_to_dict(
        indices, 'propensity / frequency scale naming convention', int
    )

    if indices_dict == {}:
        loop = True
    else:
        loop = False

    while loop is True:
        print('Specify dictionary describing propensity / frequency scale '
              'naming convention')
        indices = input(prompt).strip()
        indices_dict = convert_str_to_dict(
            indices, 'propensity / frequency scale naming convention', int
        )
        if indices_dict != {}:
            loop = False
            break

    return indices_dict


def def_prop_freq_scale_weights(params):
    """
    Defines propensity scale weights. Must be run AFTER "propensityscales",
    "frequencyscales" and "dictnameindices" have been defined.
    N.B. Have left this hyperparameter in for now in case I want to use it in
    the future, but in general, and certainly in the case of my initial
    design run, I think this is "an optimisation too far". Will keep all the
    same (i.e. params['scaleweights'] = 'equal') for now.
    """

    scales = (  list(params['propensityscales'].keys())
              + list(params['frequencyscales'].keys()))

    try:
        scale_weights = params['scaleweights'].strip('/')
    except KeyError:
        scale_weights = ''

    if scale_weights.lower() == 'equal':
        scale_weights = {}
        for dict_name in scales:
            try:
                pairindv_index = params['dictnameindices']['pairorindv']
            except KeyError:
                raise KeyError(
                    '"pairorindv" label not included in dictionary naming '
                    'convention - please update your propensity + frequency '
                    'scale names and/or the "dictnameindices" parameter'
                )

            if dict_name.split('_')[pairindv_index] == 'indv':
                scale_weights[dict_name] = 1
            elif dict_name.split('_')[pairindv_index] == 'pair':
                scale_weights[dict_name] = 0.5  # Each pair is counted twice

    elif os.path.isfile('/{}'.format(scale_weights)) and scale_weights.endswith('.pkl'):
        with open('/{}'.format(scale_weights), 'rb') as pickle_file:
            scale_weights_dict = pickle.load(pickle_file)
        if any(type(scale_weights_dict) == x for x in [dict, OrderedDict]):
            scale_weights = scale_weights_dict
        else:
            print('Propensity scale weights dictionary not recognised')
            scale_weights = {}

    elif scale_weights[0] == '{' and scale_weights[-1] == '}' and ':' in scale_weights:
        scale_weights = convert_str_to_dict(
            scale_weights, 'propensity scale weights', float
        )

    else:
        print('Propensity scale weights not provided')
        scale_weights = {}

    for dict_name in scales:
        try:
            scale_weight = scale_weights[dict_name]
        except KeyError:
            print('Weight for {} not provided'.format(dict_name))
            scale_weight = ''

        while not type(scale_weight) in [int, float]:
            print('Weighting for {} not recognised.\n'
                  'Specify weight for {}'.format(dict_name, dict_name))
            scale_weight = input(prompt)

            try:
                scale_weight = float(scale_weight)
                scale_weights[dict_name] = scale_weight
                break
            except ValueError:
                scale_weight = ''

    for scale in list(scale_weights.keys()):
        if not scale in scales:
            raise Exception(
                'Scale {} is not included amongst the input propensity and/or '
                'frequency dictionaries provided'.format(scale))

    return scale_weights


def def_propensity_weight(params):
    """
    Defines weighting between propensity and frequency scales. Must be
    run AFTER "propensityscales" and "frequencyscales" have been defined.
    """

    if params['frequencyscales'] == {} and prop_weight != '':  # This works for
    # both unordered and ordered dictionaries
        raise ValueError('Value provided for "propensityweight", but no input '
                         'frequency scales have been defined.')

    try:
        prop_weight = float(params['propensityweight'])
        if 0 > prop_weight or prop_weight > 1:
            print('Weighting for propensity scales not recognised - please '
                  'enter a value between 0 and 1')
            prop_weight = ''
    except (IndexError, ValueError):
        print('Weighting for propensity scales not recognised - please '
              'enter a value between 0 and 1')
        prop_weight = ''

    if prop_weight == '':
        if params['frequencyscales'] == {}:
            print('No frequency scales defined, so setting propensity weighting'
                  ' to 1')
            prop_weight = 1

        else:
            while not type(prop_weight) == float:
                print('Specify weight for propensity scales:')
                prop_weight = input(prompt)

                try:
                    prop_weight = float(prop_weight)
                    if 0 <= prop_weight <= 1:
                        break
                    else:
                        print('Weighting for propensity scales not recognised '
                              '- please enter a value between 0 and 1')
                        prop_weight = ''
                except ValueError:
                    print('Weighting for propensity scales not recognised - '
                          'please enter a value between 0 and 1')
                    prop_weight = ''

    return prop_weight


def def_phipsi_cluster_coords(params):
    """
    Calculates phi and psi classes if discrete phi / psi dict is input. Must be
    run AFTER "propensityscales", "frequencyscales" and "dictnameindices" have
    been defined.
    """

    scales = (  list(params['propensityscales'].keys())
              + list(params['frequencyscales'].keys()))

    for dict_label in scales:
        dict_label = dict_label.split('_')
        try:
            prop1_index = params['dictnameindices']['prop1']
        except KeyError:
            raise KeyError(
                '"prop1" label not included in dictionary naming convention - '
                'please update your propensity + frequency scale names and/or '
                'the "dictnameindices" parameter'
            )
        try:
            discorcont_index = params['dictnameindices']['discorcont']
        except KeyError:
            raise KeyError(
                '"discorcont" label not included in dictionary naming '
                'convention - please update your propensity + frequency '
                'scale names and/or the "dictnameindices" parameter'
            )

        if dict_label[prop1_index] == 'phipsi' and dict_label[discorcont_index] == 'disc':
            phipsi_coords = ''

            try:
                phipsi_coords = params['phipsiclustercoords']
                if not os.path.isfile(phipsi_coords) or not phipsi_coords.endswith('.pkl'):
                    print('File path to pickled phi/psi voronoi point '
                          'coordinates not recognised')
                    phipsi_coords = ''
                else:
                    with open(phipsi_coords, 'rb') as pickle_file:
                        phipsi_coords = pickle.load(pickle_file)
            except KeyError:
                phipsi_coords = ''

            if phipsi_coords == '':
                while not os.path.isfile(phipsi_coords) or not phipsi_coords.endswith('.pkl'):
                    print('Specify absolute file path of pickled phi / psi '
                          'voronoi point coordinates:')
                    phipsi_coords = '/' + input(prompt).replace('\\', '/').lstrip('/')

                    if os.path.isfile(phipsi_coords) and phipsi_coords.endswith('.pkl'):
                        with open(phipsi_coords, 'rb') as pickle_file:
                            phipsi_coords = pickle.load(pickle_file)
                        break
                    else:
                        print('File path to pickled phi / psi voronoi point '
                              'coordinates not recognised')
            break  # Once phipsi scale has been found, don't need to loop
            # through subsequent scales

    return phipsi_coords


def def_working_directory(params):
    """
    Defines working directory. Note that unlike the file paths to propensity
    dictionaries etc., users can provide a relative rather than an absolute
    directory path if they prefer
    """

    try:
        wd = params['workingdirectory']
        if not os.path.isdir(wd):
            print('File path to working directory not recognised')
            wd = ''
    except KeyError:
        wd = ''

    while not os.path.isdir(wd):
        print('Specify absolute path of working directory')
        wd = input(prompt).replace('\\', '/').rstrip('/') + '/'

        if os.path.isdir(wd):
            break
        else:
            print('File path to working directory not recognised')

    return wd


def def_barrel_or_sandwich(params):
    """
    Defines whether the input structure is a beta-sandwich or a beta-barrel
    backbone
    """

    try:
        barrel = params['barrelorsandwich']
        if barrel == 'barrel':
            barrel = '2.40'
        elif barrel == 'sandwich':
            barrel = '2.60'
        if not barrel in ['2.40', '2.60']:
            print('Backbone structure not recognised')
            barrel = ''
    except KeyError:
        barrel = ''

    if barrel == '':
        while not barrel in ['barrel', '2.40', 'sandwich', '2.60']:
            print('Specify structure type - please enter "barrel" or "sandwich":')
            barrel = input(prompt).lower().replace(' ', '')

            if barrel in ['2.40', '2.60']:
                break
            elif barrel in ['barrel', 'sandwich']:
                if barrel == 'barrel':
                    barrel = '2.40'
                elif barrel == 'sandwich':
                    barrel = '2.60'
                break
            else:
                print('Structure type not recognised')

    return barrel


def def_jobid(params):
    """
    Assigns unique identification code to job
    """

    try:
        job_id = params['job_id']
    except KeyError:
        job_id = None

    if job_id is None:
        print('Specify unique ID (without spaces) for input structure (if you '
              'would like BetaDesigner to assign a random ID, enter "random")):')
        job_id = input(prompt)
        if job_id.lower().replace(' ', '') == 'random':
            job_id = ''.join([random.choice(string.ascii_letters + string.digits)
                              for i in range(6)])

    return job_id


def def_method_initial_seq(params):
    """
    Defines method used to generate initial sequences for backbone structure.
    """

    try:
        initial_method = params['initialseqmethod']
        if not initial_method in ['random', 'rawpropensity', 'rankpropensity']:
            print('Method for determining initial side chain assignments not '
                  'recognised - please select one of "random", "rawpropensity" '
                  'or "rankpropensity"')
            initial_method = ''
    except KeyError:
        initial_method = ''

    if initial_method == '':
        while not initial_method in ['random', 'rawpropensity', 'rankpropensity']:
            print('Specify method for determining initial side chain assignments:')
            initial_method = input(prompt).lower().replace(' ', '')

            if initial_method in ['random', 'rawpropensity', 'rankpropensity']:
                break
            else:
                print('Method not recognised - please select one of "random", '
                      '"rawpropensity" or "rankpropensity"')

    return initial_method


def def_method_fitness_scoring(params):
    """
    Defines method used to measure sequence fitness
    """

    try:
        fit_method = params['fitnessscoremethod']
        if not fit_method in ['propensity', 'allatom', 'alternate', 'split']:
            print('Method for measuring sequence fitness not recognised - '
                  'please select one of "propensity", "allatom", "alternate" '
                  'or "split"')
            fit_method = ''
    except KeyError:
        fit_method = ''

    if fit_method == '':
        while not fit_method in ['propensity', 'allatom', 'alternate', 'split']:
            print('Specify method for measuring sequence fitnesses:')
            fit_method = input(prompt).lower().replace(' ', '')

            if fit_method in ['propensity', 'allatom', 'alternate', 'split']:
                break
            else:
                print('Method not recognised - please select one of '
                      '"propensity", "allatom", "alternate", "split"')

    return fit_method


def def_split_fraction(params):
    """
    Defines fraction of samples to be optimised against propensity in each
    generation of the genetic algorithm.
    N.B. Must be run AFTER "fitnessscoremethod" has been defined.
    N.B. Have left this hyperparameter in for now in case I want to use it in
    the future, but in general, and certainly in the case of my initial
    design run, I think this is "an optimisation too far".
    """

    if params['fitnessscoremethod'] != 'split':
        split_frac = ''

    elif params['fitnessscoremethod'] == 'split':
        try:
            split_frac = params['splitfraction']
            try:
                split_frac = float(split_frac)
                if 0 > split_frac or split_frac > 1:
                    print('Fraction of samples to be optimised against '
                          'propensity not recognised - please enter a value '
                          'between 0 and 1')
                    split_frac = ''
            except ValueError:
                print('Fraction of samples to be optimised against propensity '
                      'not recognised - please enter a value between 0 and 1')
                split_frac = ''
        except KeyError:
            split_frac = ''

        if split_frac == '':
            while not type(split_frac) == float:
                print('Specify fraction of samples to be optimised against '
                      'propensity:')
                split_frac = input(prompt)

                try:
                    split_frac = float(split_frac)
                    if 0 <= split_frac <= 1:
                        break
                    else:
                        print('Fraction of samples to be optimised against '
                              'propensity not recognised - please enter a '
                              'value between 0 and 1')
                        split_frac = ''
                except ValueError:
                    print('Fraction of samples to be optimised against propensity '
                          'not recognised - please enter a value between 0 and 1')
                    split_frac = ''

    return split_frac


def def_method_select_mating_pop(params):
    """
    Defines method used to select a population of individuals for mating
    """

    try:
        mate_method = params['matingpopmethod']
        if not mate_method in ['fittest', 'roulettewheel', 'rankroulettewheel']:
            print('Method for generating mating population not recognised - '
                  'please select one of "fittest", "roulettewheel" or '
                  '"rankroulettewheel"')
            mate_method = ''
    except KeyError:
        mate_method = ''

    if mate_method == '':
        while not mate_method in ['fittest', 'roulettewheel', 'rankroulettewheel']:
            print('Specify method for generating mating population:')
            mate_method = input(prompt).lower().replace(' ', '')

            if mate_method in ['fittest', 'roulettewheel', 'rankroulettewheel']:
                break
            else:
                print('Method not recognised - please select one of "fittest", '
                      '"roulettewheel" or "rankroulettewheel"')

    return mate_method


def def_unfit_fraction(params):
    """
    Defines fraction of unfit sequences to be included in the mating
    population at each generation of the genetic algorithm.
    N.B. Must be run AFTER "matingpopmethod" has been defined.
    """

    if params['matingpopmethod'] != 'fittest':
        unfit_frac = ''

    elif params['matingpopmethod'] == 'fittest':
        try:
            unfit_frac = params['unfitfraction']
            try:
                unfit_frac = float(unfit_frac)
                if 0 > unfit_frac or unfit_frac > 1:
                    print('Fraction of mating population to be comprised of '
                          'unfit samples not recognised - please enter a '
                          'value between 0 and 1')
                    unfit_frac = ''
            except ValueError:
                print('Fraction of mating population to be comprised of unfit '
                      'samples not recognised - please enter a value between '
                      '0 and 1')
                unfit_frac = ''
        except KeyError:
            unfit_frac = ''

        if unfit_frac == '':
            while not type(unfit_frac) == float:
                print('Specify fraction of mating population to be comprised '
                      'of unfit samples:')
                unfit_frac = input(prompt)

                try:
                    unfit_frac = float(unfit_frac)
                    if 0 <= unfit_frac <= 1:
                        break
                    else:
                        print('Fraction of mating population to be comprised '
                              'of unfit samples not recognised - please enter '
                              'a value between 0 and 1')
                        unfit_frac = ''
                except ValueError:
                    print('Fraction of mating population to be comprised of '
                          'unfit samples not recognised - please enter a '
                          'value between 0 and 1')
                    unfit_frac = ''

    return unfit_frac


def def_method_crossover(params):
    """
    Defines method used to crossover parent sequences to generate children
    """

    try:
        cross_method = params['crossovermethod']
        if not cross_method in ['uniform', 'segmented']:
            print('Crossover method not recognised - please select one of '
                  '"uniform" or "segmented"')
            cross_method = ''
    except KeyError:
        cross_method = ''

    if cross_method == '':
        while not cross_method in ['uniform', 'segmented']:
            print('Specify crossover method:')
            cross_method = input(prompt).lower().replace(' ', '')

            if cross_method in ['uniform', 'segmented']:
                break
            else:
                print('Crossover method not recognised - please select one of '
                      '"uniform" or "segmented"')
    return cross_method


def def_crossover_prob(params):
    """
    Defines probability of exchanging amino acid identities for each node in
    the network as part of a uniform crossover.
    N.B. Must be run AFTER "crossovermethod" has been defined.
    """

    if params['crossovermethod'] != 'uniform':
        cross_prob = ''

    elif params['crossovermethod'] == 'uniform':
        try:
            cross_prob = params['crossoverprob']
            try:
                cross_prob = float(cross_prob)
                if 0 > cross_prob or cross_prob > 1:
                    print('Probability of uniform crossover not recognised - '
                          'please enter a value between 0 and 1')
                    cross_prob = ''
            except ValueError:
                print('Probability of uniform crossover not recognised - '
                      'please enter a value between 0 and 1')
                cross_prob = ''
        except KeyError:
            cross_prob = ''

        if cross_prob == '':
            while not type(cross_prob) == float:
                print('Specify probability of uniform crossover:')
                cross_prob = input(prompt)

                try:
                    cross_prob = float(cross_prob)
                    if 0 > cross_prob or cross_prob > 1:
                        break
                    else:
                        print('Probability of uniform crossover not recognised '
                              '- please enter a value between 0 and 1')
                        cross_prob = ''
                except ValueError:
                    print('Probability of uniform crossover not recognised - '
                          'please enter a value between 0 and 1')
                    cross_prob = ''

    return cross_prob


def def_swap_start_prob(params):
    """
    Defines probability of starting a (segmented) crossover.
    N.B. Must be run AFTER "crossovermethod" has been defined.
    """

    if params['crossovermethod'] != 'segmented':
        start_prob = ''

    elif params['crossovermethod'] == 'segmented':
        try:
            start_prob = params['swapstartprob']
            try:
                start_prob = float(start_prob)
                if 0 > start_prob or start_prob > 1:
                    print('Probability of initiating segmented crossover not '
                          'recognised - please enter a value between 0 and 1')
                    start_prob = ''
            except ValueError:
                print('Probability of initiating segmented crossover not '
                      'recognised - please enter a value between 0 and 1')
                start_prob = ''
        except KeyError:
            start_prob = ''

        if start_prob == '':
            while not type(start_prob) == float:
                print('Specify probability of initiating crossover:')
                start_prob = input(prompt)

                try:
                    start_prob = float(start_prob)
                    if 0 <= start_prob <= 1:
                        break
                    else:
                        print('Probability of initiating segmented crossover '
                              'not recognised - please enter a value between '
                              '0 and 1')
                        start_prob = ''
                except ValueError:
                    print('Probability of initiating segmented crossover not '
                          'recognised - please enter a value between 0 and 1')
                    start_prob = ''

    return start_prob


def def_swap_stop_prob(params):
    """
    Defines probability of stopping a (segmented) crossover.
    N.B. Must be run AFTER "crossovermethod" has been defined.
    """

    if params['crossovermethod'] != 'segmented':
        stop_prob = ''

    elif params['crossovermethod'] == 'segmented':
        try:
            stop_prob = params['swapstopprob']
            try:
                stop_prob = float(stop_prob)
                if 0 > stop_prob or stop_prob < 1:
                    print('Probability of ending segmented crossover not '
                          'recognised - please enter a value between 0 and 1')
                    stop_prob = ''
            except ValueError:
                print('Probability of ending segmented crossover not '
                      'recognised - please enter a value between 0 and 1')
                stop_prob = ''
        except KeyError:
            stop_prob = ''

        if stop_prob == '':
            while not type(stop_prob) == float:
                print('Specify probability of ending crossover:')
                stop_prob = input(prompt)

                try:
                    stop_prob = float(stop_prob)
                    if 0 <= stop_prob <= 1:
                        break
                    else:
                        print('Probability of ending segmented crossover not '
                              'recognised - please enter a value between 0 '
                              'and 1')
                        stop_prob = ''
                except ValueError:
                    print('Probability of ending segmented crossover not '
                          'recognised - please enter a value between 0 and 1')
                    stop_prob = ''

    return stop_prob


def def_method_mutation(params):
    """
    Defines method used to mutate children sequences (generated in the
    previous step from parent crossover)
    """

    try:
        mut_method = params['mutationmethod']
        if not mut_method in ['swap', 'scramble']:
            print('Mutation method not recognised - please select one of '
                  '"swap" or "scramble"')
            mut_method = ''
    except KeyError:
        mut_method = ''

    if mut_method == '':
        while not mut_method in ['swap', 'scramble']:
            print('Specify mutation method:')
            mut_method = input(prompt).lower().replace(' ', '')

            if mut_method in ['swap', 'scramble']:
                break
            else:
                print('Mutation method not recognised - please select one of '
                      '"swap" or "scramble"')
    return mut_method


def def_mutation_prob(params):
    """
    Defines probability of mutation of each node in the network
    """

    try:
        mut_prob = params['mutationprob']
        try:
            mut_prob = float(mut_prob)
            if 0 > mut_prob or mut_prob > 1:
                params['mutationprob'] = mut_prob
            else:
                print('Probability of mutation not recognised - please enter '
                      'a value between 0 and 1')
                mut_prob = ''
        except ValueError:
            print('Probability of mutation not recognised - please enter a '
                  'value between 0 and 1')
            mut_prob = ''
    except KeyError:
        mut_prob = ''

    if mut_prob == '':
        while not type(mut_prob) == float:
            print('Specify probability of mutation:')
            mut_prob = input(prompt)

            try:
                mut_prob = float(mut_prob)
                if 0 <= mut_prob <= 1:
                    break
                else:
                    print('Probability of mutation not recognised - please '
                          'enter a value between 0 and 1')
                    mut_prob = ''
            except ValueError:
                print('Probability of mutation not recognised - please enter '
                      'a value between 0 and 1')
                mut_prob = ''

    return mut_prob


def def_pop_size(params):
    """
    Defines the size of the population of sequences to be optimised by the
    genetic algorithm. The population size should be an even number, in order
    that all parent sequences can be paired off for crossover (mating).
    N.B. must be run AFTER "fitnessscoremethod" and "splitfraction" have been
    defined.
    """

    try:
        pop_size = params['populationsize']
    except KeyError:
        pop_size = ''

    error = False
    try:
        pop_size_int = int(pop_size)
        if (
               str(pop_size_int) != pop_size
            or pop_size_int <= 0
            or pop_size_int % 2 != 0
        ):
            error = True
        else:
            if params['fitnessscoremethod'] == 'split':
                pop_frac = pop_size_int * 0.5 * params['splitfraction']
                if not float(pop_frac).is_integer():
                    error = True
    except ValueError:
        error = True

    if error is True:
        if params['fitnessscoremethod'] != 'split':
            print('Population size not recognised - please enter a '
                  'positive even integer')
        else:
            print('Population size not recognised - please enter a '
                  'positive integer that when multiplied by 0.5x the '
                  'fraction of samples to be optimised against propensity '
                  'gives an integer value')
        pop_size = ''

    if pop_size == '':
        while type(pop_size) != int:
            print('Specify number of sequences in population:')
            pop_size = input(prompt).strip()

            try:
                pop_size_int = int(pop_size)
                if (
                        str(pop_size_int) == pop_size
                    and pop_size_int > 0
                    and pop_size_int % 2 == 0
                ):
                    if params['fitnessscoremethod'] != 'split':
                        pop_size = pop_size_int
                        break
                    elif params['fitnessscoremethod'] == 'split':
                        pop_frac = pop_size_int * 0.5 * params['splitfraction']
                        if float(pop_frac).is_integer():
                            pop_size = pop_size_int
                            break
            except KeyError:
                pass

            # No need to set e.g. error = True / False, since if input is
            # correct will break from while loop
            if params['fitnessscoremethod'] != 'split':
                print('Population size not recognised - please enter '
                      'a positive even integer')
            elif params['fitnessscoremethod'] == 'split':
                print('Population size not recognised - please enter a '
                      'positive integer that when multiplied by 0.5x the '
                      'fraction of samples to be optimised against '
                      'propensity gives an integer value')
            pop_size = ''

    return pop_size


def def_num_gens(params):
    """
    Defines the number of generations for which to run the genetic algorithm
    """

    try:
        num_gens = params['maxnumgenerations']
        try:
            num_gens_int = int(num_gens)
            if str(num_gens_int) == num_gens and num_gens_int > 0:
                num_gens = num_gens_int
            else:
                print('Maximum number of generations not recognised - please '
                      'enter a positive integer')
                num_gens = ''
        except ValueError:
            print('Maximum number of generations not recognised - please enter '
                  'a positive integer')
            num_gens = ''
    except KeyError:
        num_gens = ''

    if num_gens == '':
        while type(num_gens) != int:
            print('Specify maximum number of generations for which to run GA:')
            num_gens = input(prompt)

            try:
                num_gens_int = int(num_gens)
                if str(num_gens_int) == num_gens and num_gens_int > 0:
                    break
                else:
                    print('Maximum number of generations not recognised - '
                          'please enter a positive integer')
                    num_gens = ''
            except ValueError:
                print('Maximum number of generations not recognised - please '
                      'enter a positive integer')
                num_gens = ''

    return num_gens

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
                               'frequencyscales', 'scaleweights', 'phipsiclustercoords']:
                        value = value.replace('\\', '/')  # For windows file paths
                        value = '/{}'.format(value.strip('/'))
                    elif key == 'dictnameindices':
                        value = value.strip()
                    elif key in ['workingdirectory']:
                        value = value.replace('\\', '/')  # For windows file paths
                        value = '/{}/'.format(value.strip('/'))
                    elif key  == 'jobid':
                        value = value.replace(' ', '')
                    elif key in [
                        'propensityweight', 'barrelorsandwich', 'initialseqmethod',
                        'fitnessscoremethod', 'splitfraction', 'matingpopmethod',
                        'unfitfraction', 'crossovermethod', 'crossoverprob',
                        'swapstartprob', 'swapstopprob', 'mutationmethod',
                        'mutationprob', 'populationsize', 'maxnumgenerations'
                    ]:
                        value = value.lower().replace(' ', '')

                    params[key] = value

        except FileNotFoundError:
            print('Path to input file not recognised')

    params['inputdataframepath'] = def_input_df_path(params)
    params['inputpdb'] = def_input_pdb(params)
    params['propensityscales'] = def_propensity_scales(params)
    params['frequencyscales'] = def_frequency_scales(params)
    # params['dictnameindices'] = def_dict_naming_scheme(params)
    params['dictnameindices'] = {'intorext': 0,
                                 'edgeorcent': 1,
                                 'prop1': 2,
                                 'interactiontype': 3,
                                 'pairorindv': 4,
                                 'discorcont': 5,
                                 'proporfreq': 6}
    # params['scaleweights'] = def_prop_freq_scale_weights(params)
    params['scaleweights'] = 'equal'  # Currently set to equal weighting by default
    # params['propensityweight'] = def_propensity_weight(params)  # Commented out
    # because this hyperparameter has been selected fo optimisation with hyperopt
    if 'propensityweight' in params.keys():
        params.pop('propensityweight')
    params['phipsiclustercoords'] = def_phipsi_cluster_coords(params)
    params['inputdataframe'] = def_input_df(params)
    params['workingdirectory'] = def_working_directory(params)
    params['barrelorsandwich'] = def_barrel_or_sandwich(params)
    params['jobid'] = def_jobid(params)
    # params['initialseqmethod'] = def_method_initial_seq(params)
    params['initialseqmethod'] = 'random'  # Currently set to "random" by default
    # params['fitnessscoremethod'] = def_method_fitness_scoring(params)
    params['fitnessscoremethod'] = 'split'  # Currently set to "split" by default
    # params['splitfraction'] = def_split_fraction(params)
    params['splitfraction'] = 0.5  # Currently set as 50:50 by default (in my
    # opinion variation of this hyperparameter is probably an "optimisation too far")
    # params['matingpopmethod'] = def_method_select_mating_pop(params)
    params['matingpopmethod'] = 'fittest'  # Currently set as "fittest" (with the
    # value of the hyperparameter "unfitfraction" being optimised with hyperopt)
    # params['unfitfraction'] = def_unfit_fraction(params)  # Commented out
    # because this hyperparameter has been selected to be optimised with hyperopt
    if 'unfitfraction' in params.keys():
        params.pop('unfitfraction')
    # params['crossovermethod'] = def_method_crossover(params)
    params['crossovermethod'] = 'uniform'  # Currently set to uniform crossover
    # (with the value of the hyperparameter "crossoverprob" being optimised with
    # hyperopt)
    # params['crossoverprob'] = def_crossover_prob(params)  # Commented out
    # because this hyperparameter has been selected to be optimised with hyperopt
    if 'crossoverprob' in params.keys():
        params.pop('crossoverprob')
    params['swapstartprob'] = def_swap_stop_prob(params)
    params['swapstopprob'] = def_swap_stop_prob(params)
    # params['mutationmethod'] = def_method_mutation(params)
    params['mutationmethod'] = 'swap'  # Currently set as "swap" by default
    # params['mutationprob'] = def_mutation_prob(params)  # Commented out
    # because this parameter has been selected to be optimised with hyperopt
    if 'mutationprob' in params.keys():
        params.pop('mutationprob')
    params['populationsize'] = def_pop_size(params)
    if params['fitnessscoremethod'] == 'split':
        params['propensitypopsize'] = params['populationsize']*params['splitfraction']
    else:
        params['propensitypopsize'] = ''
    params['maxnumgenerations'] = def_num_gens(params)

    return params


def setup_input_output(params):
    """
    Changes directory to user-specified "working directory", creates directories
    for the input and output data, and copies across necessary input files in
    preparation for running the genetic algorithm
    """

    # Creates working directory
    params['workingdirectory'] = '{}BetaDesigner_results/{}'.format(
        params['workingdirectory'], params['jobid']
    )

    if os.path.isdir(params['workingdirectory']):
        print('Directory {} already exists'.format(params['workingdirectory']))
        delete_dir = ''

        while not delete_dir in ['yes', 'y', 'no', 'n']:
            print('Delete {}?'.format(params['workingdirectory']))
            delete_dir = input(prompt).lower()

            if delete_dir in ['yes', 'y']:
                shutil.rmtree(params['workingdirectory'])
                break
            elif delete_dir in ['no', 'n']:
                raise Exception(
                    'Exiting BetaDesigner - please provide a jobid that is not '
                    'already a directory in {}/ for future '
                    'runs'.format('/'.join(params['workingdirectory'].split('/')[:-1]))
                )
            else:
                print('Input not recognised - please specify "yes" or "no"')
                delete_dir = ''

    if not os.path.isdir(params['workingdirectory']):
        os.makedirs(params['workingdirectory'])

    # Creates directories for input and output data
    os.mkdir('{}/Program_input'.format(params['workingdirectory']))
    os.mkdir('{}/Program_output'.format(params['workingdirectory']))

    # Copies input files into input data directory
    shutil.copy(params['inputdataframepath'],
                '{}/Program_input/Input_DataFrame.pkl'.format(params['workingdirectory']))

    shutil.copy(params['inputpdb'],
                '{}/Program_input/Input_PDB.pdb'.format(params['workingdirectory']))

    with open('{}/Program_input/Propensity_scales.pkl'.format(
        params['workingdirectory']), 'wb') as pickle_file:
        pickle.dump((params['propensityscales']), pickle_file)

    if 'frequencyscales' in params.keys():
        with open('{}/Program_input/Frequency_scales.pkl'.format(
            params['workingdirectory']), 'wb') as pickle_file:
            pickle.dump((params['frequencyscales']), pickle_file)

    if 'phipsiclustercoords' in params.keys():
        with open('{}/Program_input/Ramachandran_voronoi_cluster_coords.pkl'.format(
            params['workingdirectory']), 'wb') as pickle_file:
            pickle.dump((params['phipsiclustercoords']), pickle_file)

    with open('{}/Program_input/Input_parameters.pkl'.format(
        params['workingdirectory']), 'wb') as f:
        pickle.dump((params), f)

    # Writes program params to a txt file for user records
    with open('{}/Program_input/BetaDesigner_params.txt'.format(
        params['workingdirectory']), 'w') as f:
        for key, parameter in params.items():
            f.write('{}: {}\n\n\n\n\n\n'.format(key, parameter))


class initialise_ga_object():

    def __init__(self, params, test=False):
        aa_code_dict = gen_amino_acids_dict()
        if params['barrelorsandwich'] == '2.40':
            aa_code_dict.pop('CYS')
        params['aacodes'] = list(aa_code_dict.values())

        self.input_df = params['inputdataframe']
        self.input_pdb = params['inputpdb']
        self.propensity_dicts = params['propensityscales']
        self.frequency_dicts = params['frequencyscales']
        self.aa_list = params['aacodes']
        self.dict_weights = params['scaleweights']
        self.dict_name_indices = params['dictnameindices']
        # self.propensity_weight = params['propensityweight']  To be optimised with hyperopt
        self.working_directory = params['workingdirectory']
        self.barrel_or_sandwich = params['barrelorsandwich']
        self.job_id = params['jobid']
        self.method_initial_side_chains = params['initialseqmethod']
        self.method_fitness_score = params['fitnessscoremethod']
        if self.method_fitness_score == 'split':
            self.split_fraction = params['splitfraction']
        self.method_select_mating_pop = params['matingpopmethod']
        # self.unfit_fraction = params['unfitfraction']  To be optimised with hyperopt
        self.method_crossover = params['crossovermethod']
        # self.crossover_prob = params['crossoverprob']  To be optimised with hyperopt
        self.swap_start_prob = params['swapstartprob']
        self.swap_stop_prob = params['swapstopprob']
        self.method_mutation = params['mutationmethod']
        # self.mutation_prob = params['mutationprob']  To be optimised with hyperopt
        self.pop_size = params['populationsize']
        self.propensity_pop_size = params['propensitypopsize']
        self.num_gens = params['maxnumgenerations']
        self.test = test
        self.params = params
