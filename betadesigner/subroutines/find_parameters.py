
import os
import pickle
import random
import shutil
import string
import sys
import pandas as pd
from collections import OrderedDict

prompt = '> '

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
                               'propensityscaleweights']:
                        value = value.replace('\\', '/')  # For windows file paths
                        value = '/{}'.format(value.strip('/'))
                    elif key in ['workingdirectory']:
                        value = value.replace('\\', '/')  # For windows file paths
                        value = '/{}/'.format(value.strip('/'))
                    elif key in ['jobid']:
                        value = value.replace(' ', '')
                    elif key in ['barrelorsandwich', 'populationsize',
                                 'numberofgenerations', 'initialseqmethod',
                                 'fitnessscoremethod', 'matingpopmethod',
                                 'crossovermethod', 'mutationmethod']:
                        value = value.replace(' ', '').lower()
        except FileNotFoundError:
            print('Path to input file not recognised')

    # Defines absolute file path to input dataframe, then unpickles dataframe
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

    # Unpickles dataframe
    input_df = pd.read_pickle(input_df_loc)
    parameters['inputdataframe'] = input_df

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

    # Defines absolute file path to pickle file listing propensity scales, then
    # unpickles propensity scales
    if 'propensityscales' in parameters:
        if (
            (not os.path.isfile(parameters['propensityscales']))
            or
            (not parameters['propensityscales'].endswith('.pkl'))
        ):
            print('File path to pickled propensity scales not recognised')
            parameters.pop('propensityscales')

    if not 'propensityscales' in parameters:
        propensity_scales_dict = ''
        while (
            (not os.path.isfile(propensity_scales_dict))
            or
            (not propensity_scales_dict.endswith('.pkl'))
        ):
            print('Specify absolute file path of pickled propensity scales:')
            propensity_scales_dict = input(prompt)

            if (
                (os.path.isfile(propensity_scales_dict))
                and
                (propensity_scales_dict.endswith('.pkl'))
            ):
                parameters['propensityscales'] = propensity_scales_dict
                break
            else:
                print('File path to pickled propensity scales not recognised')

    # Unpickles dictionary of propensity scales
    with open(propensity_dicts_loc, 'rb') as pickle_file:
        propensity_dicts = pickle.load(pickle_file)
    parameters['propensityscales'] = propensity_dicts

    # Defines propensity scale weights
    if 'propensityscaleweights' in parameters:
        if (
            (not os.path.isfile(parameters['propensityscaleweights']))
            or
            (not parameters['propensityscaleweights'].endswith('.pkl'))
        ):
            print('File path to pickled propensity scale weights not recognised')
            parameters.pop('propensityscaleweights')

    if not 'propensityscaleweights' in parameters:
        for propensity_dict in list(parameters['propensityscales'].keys()):
            print('Specify weight for {}')











        propensity_scales_dict = ''
        while (
            (not os.path.isfile(propensity_scales_dict))
            or
            (not propensity_scales_dict.endswith('.pkl'))
        ):
            print('Specify absolute file path of pickled propensity scales:')
            propensity_scales_dict = input(prompt)

            if (
                (os.path.isfile(propensity_scales_dict))
                and
                (propensity_scales_dict.endswith('.pkl'))
            ):
                parameters['propensityscales'] = propensity_scales_dict
                break
            else:
                print('File path to pickled propensity scales not recognised')

    # Unpickles dictionary of propensity scales
    with open(propensity_dicts_loc, 'rb') as pickle_file:
        propensity_dicts = pickle.load(pickle_file)
    parameters['propensityscales'] = propensity_dicts

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
            barrel_or_sandwich = input(prompt)

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

    # Defines the size of the population of sequences to be optimised by the
    # genetic algorithm. The population size should be an even number, in order
    # that all parent sequences can be paired off for crossover (mating).
    if 'populationsize' in parameters:
        try new_population_size = int(parameters['populationsize']):
            if (
                    str(new_population_size) == parameters['populationsize']
                and new_population_size > 0
                and new_population_size % 2 == 0
            ):
                parameters['populationsize'] = new_population_size
            else:
                print('Population size not recognised - please enter a '
                      'positive even integer')
                parameters.pop('populationsize')
        except ValueError:
            print('Population size not recognised - please enter a '
                  'positive even integer')
            parameters.pop('populationsize')

    if not 'populationsize' in parameters:
        population_size = ''
        while type(population_size) != int:
            print('Specify number of sequences in population:')
            population_size = input(prompt).strip()

            try new_population_size = int(population_size):
                if (    str(new_population_size) == population_size
                    and new_population_size > 0
                    and new_population_size % 2 == 0
                ):
                    parameters['populationsize'] = new_population_size
                    break
                else:
                    print('Population size not recognised - please enter a '
                          'positive even integer')
                    population_size = ''
            except ValueError:
                print('Population size not recognised - please enter a '
                      'positive even integer')
                population_size = ''

    # Defines the number of generations for which to run the genetic algorithm
    if 'numberofgenerations' in parameters:
        try new_num_gens = int(parameters['numberofgenerations']):
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

            try new_num_gens = int(num_gens):
                if (    str(new_num_gens) == num_gens
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
            method_initial_side_chains = input(prompt).lower()

            if method_initial_side_chains in [
                'random', 'rawpropensity', 'rankpropensity'
            ]:
                parameters['initialseqmethod'] = method_initial_side_chains
                break
            else:
                print('Method not recognised - please select one of "random", '
                      '"rawpropensity" or "rankpropensity"')

    # Define method used to measure sequence fitness
    if 'fitnessscoremethod' in parameters:
        if not parameters['fitnessscoremethod'] in ['propensity', 'allatom',
                          'alternate', 'split']:
            print('Method for measuring sequence fitness not recognised - '
                  'please select one of "propensity", "all-atom", "alternate" '
                  'or "split"')
            parameters.pop('fitnessscoremethod')

    if not 'fitnessscoremethod' in parameters:
        method_fitness_score = ''
        while not method_fitness_score in ['propensity', 'allatom',
                                           'alternate', 'split']:
            print('Specify method for measuring sequence fitnesses:')
            method_fitness_score = input(prompt).lower()

            if method_fitness_score in ['propensity', 'allatom', 'alternate',
                                        'split']:
                parameters['fitnessscoremethod'] = method_fitness_score
                break
            else:
                print('Method not recognised - please select one of '
                      '"propensity", "all-atom", "alternate", "split"')

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
            method_select_mating_pop = input(prompt).lower()

            if method_select_mating_pop in [
                'fittest', 'roulettewheel', 'rankroulettewheel'
            ]:
                parameters['matingpopmethod'] = method_select_mating_pop
                break
            else:
                print('Method not recognised - please select one of "fittest", '
                      '"roulettewheel" or "rankroulettewheel"')

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
            method_crossover = input(prompt).lower()

            if method_crossover in ['uniform', 'segmented']:
                parameters['crossovermethod'] = method_crossover
                break
            else:
                print('Crossover method not recognised - please select one of '
                      '"uniform" or "segmented"')

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
            method_mutation = input(prompt).lower()

            if method_mutation in ['swap', 'scramble']:
                parameters['mutationmethod'] = method_mutation
                break
            else:
                print('Mutation method not recognised - please select one of '
                      '"swap" or "scramble"')

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
                'Program_input/Input_DataFrame.pkl'.format(working_directory))
    shutil.copy('{}'.format(parameters['propensityscales']),
                'Program_input/Propensity_scales.pkl'.format(working_directory))

    # Writes program parameters to a txt file for user records
    with open('Program_input/BetaDesigner_parameters.txt', 'w') as f:
        for key, parameter in parameters.items():
            f.write('{}: {}\n'.format(key, parameter))

    return parameters

class initialise_class():

    def __init__(self, parameters):
        self.parameters = parameters

        self.input_df = parameters['inputdataframe']
        self.input_pdb = parameters['inputpdb']
        self.propensity_dicts = parameters['propensityscales']
        self.aas = list(self.propensity_dicts['int_z_indv'].keys())
        # self.propensity_dict_weights = parameters['propensityscaleweights']
        self.working_directory = parameters['workingdirectory']
        self.barrel_or_sandwich = parameters['barrelorsandwich']
        self.job_id = parameters['jobid']
        self.pop_size = parameters['populationsize']
        self.num_gens = parameters['numberofgenerations']
        self.method_initial_side_chains = parameters['initialseqmethod']
        self.method_fitness_score = parameters['fitnessscoremethod']
        # self.split_fraction = parameters['splitfraction']
        # self.unfit_fraction = parameters['unfitfraction']
        self.method_select_mating_pop = parameters['matingpopmethod']
        self.method_crossover = parameters['crossovermethod']
        # self.crossover_prob = parameters['crossoverprob']
        # self.swap_start_prob = parameters['swapstartprob']
        # self.swap_stop_prob = parameters['swapstopprob']
        self.method_mutation = parameters['mutationmethod']

        # OVERWRITE ONCE HAVE COMPLETED GENERATION OF PROPENSITY SCALES FROM
        # BETASTATS.
        self.propensity_dicts = OrderedDict({'int_z': {'ARG': np.array([[-50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50], [1.3, 1.3, 0.8, 0.4, 0.4, 0.2, 0.4, 0.4, 0.8, 1.3, 1.3]]),
                                                       'ASP': np.array([[-50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50], [1.3, 1.3, 0.8, 0.4, 0.4, 0.2, 0.4, 0.4, 0.8, 1.3, 1.3]]),
                                                       'GLY': np.array([[-50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50], [1.0, 1.0, 1.2, 1.4, 2.0, 2.5, 2.0, 1.4, 1.2, 1.0, 1.0]]),
                                                       'PHE': np.array([[-50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50], [0.7, 0.7, 0.5, 0.5, 0.3, 0.1, 0.3, 0.5, 0.5, 0.7, 0.7]]),
                                                       'VAL': np.array([[-50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50], [0.9, 0.9, 0.7, 0.7, 0.6, 0.5, 0.6, 0.7, 0.7, 0.9, 0.9]])},
                                             'ext_z': {'ARG': np.array([[-50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50], [1.2, 1.2, 0.6, 0.4, 0.3, 0.2, 0.3, 0.4, 0.6, 1.2, 1.2]]),
                                                       'ASP': np.array([[-50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50], [1.2, 1.2, 0.6, 0.4, 0.3, 0.2, 0.3, 0.4, 0.6, 1.2, 1.2]]),
                                                       'GLY': np.array([[-50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50], [1.0, 1.0, 1.2, 1.2, 1.4, 1.6, 1.4, 1.2, 1.2, 1.0, 1.0]]),
                                                       'PHE': np.array([[-50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50], [0.6, 0.6, 2.5, 2.0, 1.2, 0.8, 1.2, 2.0, 2.5, 0.6, 0.6]]),
                                                       'VAL': np.array([[-50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50], [0.8, 0.8, 1.3, 1.5, 1.7, 1.7, 1.7, 1.5, 1.3, 0.8, 0.8]])}
                                           })
