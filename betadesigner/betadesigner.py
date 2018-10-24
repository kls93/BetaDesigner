
import argparse
import sys

# Pipeline script to run the BetaDesigner program. The program takes as input a
# PDB file of backbone coordinates. The program optimises an initial dataset of
# possible sequences to fit the structural features of the backbone coordinates
# using a genetic algorithm.


def main():
    if __name__ == '__main__':
        from subroutines.find_parameters import find_parameters
        from subroutines.generate_initial_sequences import gen_ga_input_pipeline
        from subroutines.run_genetic_algorithm import run_ga_pipeline
        from subroutines.write_output_structures import gen_output
    else:
        from betadesigner.subroutines.find_parameters import find_parameters
        from betadesigner.subroutines.generate_initial_sequences import gen_ga_input_pipeline
        from betadesigner.subroutines.run_genetic_algorithm import run_ga_pipeline
        from betadesigner.subroutines.write_output_structures import gen_output

    # Reads in command line inputs
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_file', help='OPTIONAL: Specifies the '
                        'absolute file path of an input file listing run '
                        'parameters')
    args = parser.parse_args()

    # Defines program parameters from input file / user input
    parameters = find_parameters(args)

    # Checks that only one structure is listed in the input dataframe
    if len(set(parameters['inputdataframe']['domain_ids'].tolist())) != 1:
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
    gen_initial_sequences = gen_ga_input_pipeline(parameters)
    initial_sequences_dict = gen_initial_sequences.initial_sequences_pipeline()

    # Optimises sequences for amino acid propensities (considering both
    # individual and pairwise interactions) and side-chain packing using a
    # genetic algorithm.
    genetic_algorithm = run_ga_pipeline(parameters)
    output_sequences_dict = genetic_algorithm.run_genetic_algorithm(
        initial_sequences_dict
    )

    # Uses SCWRL4 within ISAMBARD to pack the output sequences onto the
    # backbone model, and writes the resulting model to a PDB file. Also
    # returns each model's total energy within BUDEFF.
    output = gen_output(parameters)
    output.write_pdb(output_sequences_dict)


# Calls main() function if betadesigner.py is run as a script
if __name__ == '__main__':
    main()
