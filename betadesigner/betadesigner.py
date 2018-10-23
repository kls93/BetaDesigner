
import argparse
import pickle
import sys
import pandas as pd

# Pipeline script to run the BetaDesigner program. The program takes as input a
# PDB file of backbone coordinates. The program optimises an initial dataset of
# possible sequences to fit the structural features of the backbone coordinates
# using a genetic algorithm.


def main():
    if __name__ == '__main__':
        from subroutines.find_parameters import find_parameters
        from subroutines.generate_initial_sequences import gen_ga_input_pipeline
        from subroutines.run_genetic_algorithm import run_ga
        from subroutines.write_output_structures import gen_output
    else:
        from betadesigner.subroutines.find_parameters import find_parameters
        from betadesigner.subroutines.generate_initial_sequences import gen_ga_input_pipeline
        from betadesigner.subroutines.run_genetic_algorithm import run_ga
        from betadesigner.subroutines.write_output_structures import gen_output

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
    pop_size = parameters['populationsize']
    num_gens = parameters['numberofgenerations']
    method_initial_side_chains = parameters['initialseqmethod']
    method_fitness_score = parameters['fitnessscoremethod']
    method_select_mating_pop = parameters['matingpopmethod']
    method_crossover = parameters['crossovermethod']
    method_mutation = parameters['mutationmethod']

    # Unpickles dataframe and dictionary of propensity scales
    input_df = pd.read_pickle(input_df_loc)
    with open(propensity_dicts_loc, 'rb') as pickle_file:
        propensity_dicts = pickle.load(pickle_file)

    # Checks that only one structure is listed in the input dataframe
    if len(set(input_df['domain_ids'].tolist())) != 1:
        sys.exit('More than one structure listed in input dataframe')

    # Generates networks of interacting residues from input dataframe, in which
    # residues are represented by nodes (labelled with their identity
    # (initially set to 'UNK'), their z-coordinate, their buried surface
    # area (sandwiches only), and whether they are an edge or a central strand
    # (sandwiches only)), and the interactions between residues are represented
    # by edges (separate edges are constructed for hydrogen-bonding backbone
    # interactions and non-hydrogen bonding backbone interactions (see
    # Hutchinson et al., 1998), and +/-2 interactions). Two networks are
    # constructed for a barrel (interior and exterior surfaces), and three
    # networks are constructured for a sandwich (interior and two exterior
    # surfaces).
    initial_sequences_object = gen_ga_input_pipeline(
        input_df, propensity_dicts, barrel_or_sandwich, pop_size,
        method_initial_side_chains
    )
    initial_sequences_dict = initial_sequences_object.initial_sequences_pipeline()

    # Optimises sequences for amino acid propensities (considering both
    # individual and pairwise interactions) and side-chain packing using a
    # genetic algorithm.
    # NOTE: make sure genetic algorithm parameter values can vary!
    ga_solutions = run_ga()
    random_initial_sequences_dict = ga_solutions.run_genetic_algorithm(random_initial_sequences_dict)
    raw_propensity_initial_sequences_dict = ga_solutions.run_genetic_algorithm(raw_propensity_initial_sequences_dict)
    rank_propensity_initial_sequences_dict = ga_solutions.run_genetic_algorithm(rank_propensity_initial_sequences_dict)

    # Writes PDB files of output sequences
    """
    os.mkdir('Program_output/Output_structures')
    output = gen_output()
    """


# Calls main() function if betadesigner.py is run as a script
if __name__ == '__main__':
    main()
