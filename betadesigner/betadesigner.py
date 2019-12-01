
import argparse
import copy
import math
from hyperopt import fmin, hp, tpe, Trials

# Wrapper script to run the BetaDesigner program. The program takes as input a
# PDB file of backbone coordinates. The program optimises an initial dataset of
# possible sequences to fit the structural features of the backbone coordinates
# using a genetic algorithm.


def main():
    if __name__ == '__main__':
        from subroutines.find_parameters import find_params, setup_input_output
        from subroutines.generate_initial_sequences import gen_ga_input
        from subroutines.run_genetic_algorithm import run_genetic_algorithm
        from subroutines.write_output_structures import gen_output
    else:
        from betadesigner.subroutines.find_parameters import find_params, setup_input_output
        from betadesigner.subroutines.generate_initial_sequences import gen_ga_input
        from betadesigner.subroutines.run_genetic_algorithm import run_genetic_algorithm
        from betadesigner.subroutines.write_output_structures import gen_output

    # Reads in command line inputs
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_file', help='OPTIONAL: Specifies the '
                        'absolute file path of an input file listing run '
                        'params')
    args = parser.parse_args()

    # Defines program params from input file / user input
    params = find_params(args)

    # Deletes parameters to be optimised with hyperopt
    del params['unfitfraction']
    del params['crossoverprob']
    del params['mutationprob']
    del params['propvsfreqweight']
    del params['jobid']

    # Checks that only one structure is listed in the input dataframe
    if len(set(params['inputdataframe']['domain_ids'].tolist())) != 1:
        raise Exception('More than one structure listed in input dataframe')

    # Generates networks of interacting residues from input dataframe, in which
    # residues are represented by nodes (labelled with their identity
    # (initially set to 'UNK'), their z-coordinate, their buried surface
    # area (sandwiches only), and whether they are an edge or a central strand
    # (sandwiches only)), and the interactions between residues are represented
    # by edges (separate edges are constructed for hydrogen-bonding backbone
    # interactions and non-hydrogen bonding backbone interactions (see
    # Hutchinson et al., 1998), +/-2 interactions and van der Waals
    # interactions). Two networks are constructed for a barrel (interior and
    # exterior surfaces), and three networks are constructed for a sandwich
    # (interior and two exterior surfaces).
    gen_initial_sequences = gen_ga_input(params)
    new_sequences_dict = gen_initial_sequences.initial_sequences_pipeline()

    run_ga = True
    run_opt = True
    max_evals = 100
    total_gen = 10
    max_gen = copy.copy(params['maxnumgenerations'])
    params['maxnumgenerations'] = total_gen

    count = 0
    while run_ga is True:
        count += 1
        params['workingdirectory'] = '{}/Optimisation_cycle_{}'.format(
            params['workingdirectory'], count
        )
        setup_input_output(params)

        orig_sequences_dict = copy.deepcopy(new_sequences_dict)
        trials = Trials()

        # Sets the hyperparams unfit_fraction, crossover_probability,
        # mutation_probability, propensity_to_frequency weighting and
        # propensity_weights to uniform ranges between 0 and 1 for optimisation
        # via a tree parzen estimator with hyperopt
        bayes_params = {}
        bayes_params['unfitfraction'] = hp.uniform('unfitfraction', 0, 1)
        bayes_params['crossoverprob'] = hp.uniform('crossoverprob', 0, 1)
        bayes_params['mutationprob'] = hp.uniform('mutationprob', 0, 1)
        bayes_params['propvsfreqweight'] = hp.uniform('propvsfreqweight', 0, 1)
        bayes_params['sequencesdict'] = orig_sequences_dict
        bayes_params = {**copy.deepcopy(params), **bayes_params}

        while run_opt is True:
            best_params = fmin(fn=run_genetic_algorithm, space=bayes_params,
                               algo=tpe.suggest, trials=trials, max_evals=max_evals)
            if max_evals == 100:
                current_best = copy.deepcopy(best_params)
                max_evals = int(max_evals * math.sqrt(10))
            else:
                # If best values are within 1% of previous OR number of trials > 1000000
                similarity_dict = {}
                for key in list(best_params.keys()):
                    if key in ['unfitfraction', 'crossoverprob', 'mutationprob', 'propvsfreqweight']:
                        if (
                               ((0.99*current_best[key]) <= best_params[key] <= (1.01*current_best[key]))
                            or max_evals >= 1000000
                        ):
                            similarity_dict[key] = True
                if all(x is True for x in similarity_dict.values()):
                    run_opt = False
                else:
                    current_best = copy.deepcopy(best_params)
                    max_evals = int(max_evals * math.sqrt(10))

        fitness = run_genetic_algorithm(best_params)
        with open('{}/Program_output/GA_output_sequences_dict.pkl'.format(
            params['workingdirectory']), 'rb') as f:
            new_sequences_dict = pickle.load(f)

        if total_gen == 10:
            current_fitness = copy.deepcopy(fitness)
        else:
            # If updated fitness is within 1% of previous fitness score OR
            # number of generations > user-defined limit
            if (
                   ((0.99*current_fitness) <= fitness <= (1.01*current_fitness))
                or total_gen >= max_gen
            ):
                run_ga = False
            else:
                current_fitness = copy.deepcopy(fitness)
                total_gen += 10

    # Uses SCWRL4 within ISAMBARD to pack the output sequences onto the
    # backbone model, and writes the resulting model to a PDB file. Also
    # returns each model's total energy within BUDEFF.
    with open('{}/Program_output/GA_output_sequences_dict.pkl'.format(
        params['workingdirectory']), 'rb') as f:
        sequences_dict = pickle.load(f)
    best_bayes_params = {'unfitfraction': best_params['unfitfraction'],
                         'crossoverprob': best_params['crossoverprob'],
                         'mutationprob': best_params['mutationprob'],
                         'propvsfreqweight': best_params['propvsfreqweight'],
                         'sequencesdict': sequences_dict}
    output = gen_output(params, best_bayes_params)
    output.write_pdb(sequences_dict)
    output.score_pdb_pyrosetta()
    output.score_pdb_molprobity()


# Calls main() function if betadesigner.py is run as a script
if __name__ == '__main__':
    main()
