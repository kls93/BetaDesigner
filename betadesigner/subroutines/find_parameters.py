
import os
import random
import shutil
import string
import sys
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
                    value = line.split(':').replace('\n', '').strip()

                    if key in ['inputdataframe', 'propensityscales']:
                        value = value.replace('\\', '/')  # For windows file paths
                        value = '/{}'.format(value.strip('/'))
                    elif key in ['workingdirectory']:
                        value = value.replace('\\', '/')  # For windows file paths
                        value = '/{}/'.format(value.strip('/'))
                    elif key in ['barrelorsandwich', 'jobid']:
                        value = value.replace(' ', '')
        except FileNotFoundError:
            sys.exit('Path to input file not recognised')

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

    # Defines absolute file path to pickle file listing propensity scales
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
                parameters['propensityscales'] = ProcessLookupError
                break
            else:
                print('File path to pickled propensity scales not recognised')

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

        if not parameters['barrelorsandwich'].lower() in ['2.40', '2.60']:
            print('Backbone structure not recognised')
            parameters.pop('barrelorsandwich')
    if not 'barrelorsandwich' in parameters:
        barrel_or_sandwich = ''
        while not barrel_or_sandwich in ['barrel', 'sandwich']:
            print('Specify structure type - please enter "barrel" or "sandwich":')
            barrel_or_sandwich = input(prompt).lower()

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
        print('Specify unique ID for input structure (if you would like BetaDesigner to assign a random ID, enter "random")):')
        job_id = input(prompt).lower()
        if job_id == 'random':
            job_id = ''.join([random.choice(string.ascii_letters + string.digits) for i in range(6)])
        parameters['jobid'] = job_id

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
                '{}/Program_input/Input_DataFrame.pkl'.format(working_directory))
    shutil.copy('{}'.format(parameters['propensityscales']),
                '{}/Program_input/Propensity_scales.pkl'.format(working_directory))

    # Writes program parameters to a txt file for user records
    with open('Program_input/BetaDesigner_parameters.txt', 'w') as f:
        for key, parameter in parameters.items():
            f.write('{}: {}\n'.format(key, parameter))

    return parameters
