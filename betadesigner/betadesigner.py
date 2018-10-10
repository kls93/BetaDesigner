
import argparse
import sys
import pandas as pd

# Pipeline script to run the BetaDesigner program. The program takes as input a
# PDB file of backbone coordinates, from which it generates a different


def main():
    if name == '__main__':
        from subroutines.find_parameters import find_parameters
        from subroutines.generate_network import generate_network
        from subroutines.run_genetic_algorithm import
        from subroutines.write_output_structures import
    else:
        from betadesigner.subroutines.find_parameters import find_parameters
        from betadesigner.subroutines.generate_network import generate_network
        from betadesigner.subroutines.run_genetic_algorithm import
        from betadesigner.subroutines.write_output_structures

    # Reads in command line inputs
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_file', help='OPTIONAL: Specifies the '
                        'absolute file path of an input file listing run '
                        'parameters')
    args = parser.parse_args()

    # Defines program parameters from input file / user input
    parameters = find_parameters(args)

    input_df_loc = parameters['inputdataframe']
    propensity_dicts_loc = parameters['propensityscales']
    working_directory = parameters['workingdirectory']
    barrel_or_sandwich = parameters['barrelorsandwich']
    job_id = parameters['jobid']

    # Unpickles dataframe and dictionary of propensity scales
    input_df = pd.read_pickle(input_df_loc)
    with open(propensity_dicts_loc, 'rb') as pickle_file:
        propensity_dicts = pickle.load(pickle_file)

    # Checks that only one structure is listed in the input dataframe
    if len(set(input_df['domain_ids'].tolist())) != 1:
        sys.exit('More than one structure listed in input dataframe')

    # Generates networks of interacting residues from input dataframe, in which
    # residues are represented by nodes (labelled with their identity
    # (initially set to 'UNK'), their z-coordinate, and their buried surface
    # area (sandwiches only)), and the interactions between residues are
    # represented by edges (separate edges are constructed for hydrogen bonding
    # interactions, non-hydrogen bonding interactions and +/-2 interactions).
    # Two networks are constructed for a barrel (interior and exterior
    # surfaces), and three networks are constructured for a sandwich (interior
    # and two exterior surfaces).
    networks = generate_network(input_df, propensity_dicts)
    networks.create_network()

    # Adds side-chains onto networks using individual amino acid propensity
    # scales to generate population of starting sequences

    # Optimises sequences for amino acid propensities (considering both
    # individual and pairwise interactions) and side-chain packing using a
    # genetic algorithm.
    # NOTE: make sure genetic algorithm parameter values can vary!

    # Writes PDB files of output sequences
    os.mkdir('Program_output/Output_structures')


# Calls main() function if betadesigner.py is run as a script
if name == '__main__':
    main()
