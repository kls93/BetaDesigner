
import argparse
import copy
import math
import pickle
from hyperopt import fmin, hp, tpe, Trials

# Wrapper script to run the BetaDesigner program. The program takes as input a
# PDB file of backbone coordinates. The program optimises an initial dataset of
# possible sequences to fit the structural features of the backbone coordinates
# using a genetic algorithm.
# N.B. INPUT BETA-BARREL STRUCTURES MUST BE ORIENTED SUCH THAT THEIR Z-AXIS IS
# ALIGNED WITH THE MEMBRANE NORMAL


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

    # Checks none of the hyperparameters to be optimised with hyperopt have
    # been defined as fixed values
    if any(
        x in params.keys() for x in ['unfitfraction', 'crossoverprob',
        'mutationprob', 'propensityweight']
    ):
        raise Exception('Parameter set aside for Bayesian optimisation has '
                        'been defined in the program input')

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
    (input_sequences_dict, new_sequences_dict
    ) = gen_initial_sequences.initial_sequences_pipeline()
    params['initialsequencesdict'] = input_sequences_dict

    run_ga = True
    run_opt = True
    max_evals = 10  # Number of hyperparameter combinations for hyperopt to try
    sub_gen = 50  # Number of generations to run the GA with a particular
    # combination of hyperparameters
    # max_gen = copy.copy(params['maxnumgenerations'])  # Total number of generations
    # GA can be run for (changing hyperparameters every sub_gen generations)
    max_gen = 50
    params['maxnumgenerations'] = sub_gen

    opt_cycle_count = 0
    # Runs GA in subsets of sub_gen generations, until either the output fitness
    # score has converged (to within 1%) or the number of generations exceeds
    # the user-defined threshold
    while run_ga is True:
        opt_cycle_count += 1

        orig_sequences_dict = copy.deepcopy(new_sequences_dict)

        # Runs hyperparameter optimisation in increments of sqrt(10) evaluations,
        # until either the selected hyperparameter values have converged (to
        # within 1%) or the number of combinations of hyperparameters tested
        # exceeds 1000000
        while run_opt is True:
            # Sets the hyperparams unfit_fraction, crossover_probability,
            # mutation_probability, propensity_to_frequency weighting and
            # propensity_weights to uniform ranges between 0 and 1 for optimisation
            # via a tree parzen estimator with hyperopt
            bayes_params = {}
            bayes_params['crossoverprob'] = hp.uniform('crossoverprob', 0, 1)
            bayes_params['mutationprob'] = hp.uniform('mutationprob', 0, 1)
            bayes_params['propensityweight'] = hp.uniform('propensityweight', 0, 1)
            bayes_params['unfitfraction'] = hp.uniform('unfitfraction', 0, 1)

            # Pickles parameters not being optimised with hyperopt (unfortunately
            # can't feed dataframe (or series or array) data into a function with
            # hyperopt, so am having to pickle the parameters not being optimised
            # with hyperopt
            bayes_params['optimisationcycle'] = opt_cycle_count
            bayes_params['hyperparam_count'] = max_evals
            updated_params = setup_input_output(
                copy.deepcopy(params), opt_cycle_count, bayes_params['hyperparam_count']
            )
            # DEFINITION OF bayes_params['workingdirectory'] MUST FOLLOW
            # DEFINITION OF updated_params AFTER RUNNING setup_input_output
            bayes_params['workingdirectory'] = updated_params['workingdirectory']

            updated_params['sequencesdict'] = copy.deepcopy(orig_sequences_dict)
            with open('{}/Program_input/Input_params.pkl'.format(
                bayes_params['workingdirectory']), 'wb'
            ) as f:
                pickle.dump(updated_params, f)

            trials = Trials()

            # Runs GA with hyperopt
            best_params = fmin(fn=run_genetic_algorithm, space=bayes_params,
                               algo=tpe.suggest, trials=trials, max_evals=max_evals)
            for trial in trials.trials:
                # Record trial_id, hyperparameter values and corresponding loss
                trial_id = '{}_{}_{}'.format(opt_cycle_count, max_evals, trial['tid'])
                crossover_prob = trial['misc']['vals']['crossoverprob'][0]
                mutation_prob = trial['misc']['vals']['mutationprob'][0]
                prop_weight = trial['misc']['vals']['propensityweight'][0]
                unfit_frac = trial['misc']['vals']['unfitfraction'][0]
                out = trial['result']['loss']
                with open('{}/Program_output/Hyperparameter_track.txt'.format(
                    updated_params['workingdirectory']), 'a') as f:
                    if trial['tid'] == 0:
                        f.write('\n\n\nHyperparameter cycle {}\n'.format(max_evals))
                        f.write('trial_id:     crossover_probability, mutation_probability,'
                                ' propensity_weight, unfit_fraction     output_loss\n')
                    f.write('{}:     {}, {}, {}, {}     {}\n'.format(trial_id,
                        crossover_prob, mutation_prob, prop_weight, unfit_frac, out
                    ))
            with open('{}/Program_output/Hyperparameter_track.txt'.format(
                updated_params['workingdirectory']), 'a') as f:
                f.write('\nBest_params:\n')
                f.write('crossover_probability: {}\n'.format(best_params['crossoverprob']))
                f.write('mutation_probability: {}\n'.format(best_params['mutationprob']))
                f.write('propensity_weight: {}\n'.format(best_params['propensityweight']))
                f.write('unfit_fraction: {}\n'.format(best_params['unfitfraction']))

            if max_evals == 1:
                current_best = copy.deepcopy(best_params)
                max_evals = int(max_evals * math.sqrt(10))
            else:
                # If best values are within 5% of previous OR number of trials >= 10000
                similarity_dict = {}
                for key in list(best_params.keys()):
                    if key in ['unfitfraction', 'crossoverprob', 'mutationprob', 'propensityweight']:
                        if (
                               ((0.95*current_best[key]) <= best_params[key]
                                 <= (1.05*current_best[key]))
                            or max_evals >= 10000
                        ):
                            similarity_dict[key] = True
                        else:
                            similarity_dict[key] = False
                if all(x is True for x in similarity_dict.values()):
                    run_opt = False
                else:
                    current_best = copy.deepcopy(best_params)
                    max_evals = int(max_evals * math.sqrt(10))

        best_params['optimisationcycle'] = opt_cycle_count
        best_params['hyperparam_count'] = '{}_final_run'.format(max_evals)
        updated_params = setup_input_output(
            copy.deepcopy(params), opt_cycle_count, best_params['hyperparam_count']
        )
        # DEFINITION OF best_params['workingdirectory'] MUST FOLLOW DEFINITION
        # OF updated_params AFTER RUNNING setup_input_output
        best_params['workingdirectory'] = updated_params['workingdirectory']

        updated_params['sequencesdict'] = copy.deepcopy(orig_sequences_dict)
        with open('{}/Program_input/Input_params.pkl'.format(
            best_params['workingdirectory']), 'wb'
        ) as f:
            pickle.dump(updated_params, f)

        fitness = run_genetic_algorithm(best_params)
        with open(
            '{}/Program_output/GA_output_sequences_dict.pkl'.format(
            best_params['workingdirectory']), 'rb'
        ) as f:
            new_sequences_dict = pickle.load(f)

        if max_gen == sub_gen:
            break

        if sub_gen == 10:  # Optimises hyperparameters for 10 generation cycles
            current_fitness = copy.deepcopy(fitness)
        else:
            # If updated fitness is within 5% of previous fitness score OR
            # number of generations > user-defined limit
            if (
                   ((0.95*current_fitness) <= fitness <= (1.05*current_fitness))
                or sub_gen >= max_gen
            ):
                run_ga = False
                break
            else:
                current_fitness = copy.deepcopy(fitness)
                sub_gen *= 2

    # Uses SCWRL4 within ISAMBARD to pack the output sequences onto the
    # backbone model, and writes the resulting model to a PDB file. Also
    # returns each model's total energy within BUDEFF.
    with open('{}/Program_output/GA_output_sequences_dict.pkl'.format(
        updated_params['workingdirectory']), 'rb') as f:
        sequences_dict = pickle.load(f)
    best_bayes_params = {'unfitfraction': best_params['unfitfraction'],
                         'crossoverprob': best_params['crossoverprob'],
                         'mutationprob': best_params['mutationprob'],
                         'propensityweight': best_params['propensityweight'],
                         'sequencesdict': sequences_dict}
    output = gen_output(updated_params, best_bayes_params)
    (updated_sequences_dict, structures_dict, bude_struct_energies_dict
    ) = output.write_pdb(sequences_dict)
    (rosetta_struct_energies_dict, rosetta_res_energies_dict
    ) = output.score_pdb_rosetta(structures_dict)
    (molp_struct_dict, molp_res_dict
    ) = output.score_pdb_molprobity(structures_dict)

    with open('{}/Program_output/GA_output_sequences_full_name_dict.pkl'.format(
        updated_params['workingdirectory']), 'wb') as f:
        pickle.dump(updated_sequences_dict, f)
    with open('{}/BetaDesigner_results/{}/GA_output_sequences_dict.pkl'.format(
        params['workingdirectory'], params['jobid']), 'wb') as f:
        pickle.dump(updated_sequences_dict, f)

    with open('{}/Program_output/BUDE_struct_energies_dict.pkl'.format(
        updated_params['workingdirectory']), 'wb') as f:
        pickle.dump(bude_struct_energies_dict, f)
    with open('{}/BetaDesigner_results/{}/BUDE_struct_energies_dict.pkl'.format(
        params['workingdirectory'], params['jobid']), 'wb') as f:
        pickle.dump(bude_struct_energies_dict, f)

    with open('{}/Program_output/Rosetta_struct_energies_dict.pkl'.format(
        updated_params['workingdirectory']), 'wb') as f:
        pickle.dump(rosetta_struct_energies_dict, f)
    with open('{}/BetaDesigner_results/{}/Rosetta_struct_energies_dict.pkl'.format(
        params['workingdirectory'], params['jobid']), 'wb') as f:
        pickle.dump(rosetta_struct_energies_dict, f)

    with open('{}/Program_output/Rosetta_res_energies_dict.pkl'.format(
        updated_params['workingdirectory']), 'wb') as f:
        pickle.dump(rosetta_res_energies_dict, f)
    with open('{}/BetaDesigner_results/{}/Rosetta_res_energies_dict.pkl'.format(
        params['workingdirectory'], params['jobid']), 'wb') as f:
        pickle.dump(rosetta_res_energies_dict, f)

    with open('{}/Program_output/MolProbity_struct_score_dict.pkl'.format(
        updated_params['workingdirectory']), 'wb') as f:
        pickle.dump(molp_struct_dict, f)
    with open('{}/BetaDesigner_results/{}/MolProbity_struct_score_dict.pkl'.format(
        params['workingdirectory'], params['jobid']), 'wb') as f:
        pickle.dump(molp_struct_dict, f)

    with open('{}/Program_output/MolProbity_res_score_dict.pkl'.format(
        updated_params['workingdirectory']), 'wb') as f:
        pickle.dump(molp_res_dict, f)
    with open('{}/BetaDesigner_results/{}/MolProbity_res_score_dict.pkl'.format(
        params['workingdirectory'], params['jobid']), 'wb') as f:
        pickle.dump(molp_res_dict, f)

    return (updated_sequences_dict, structures_dict, bude_struct_energies_dict,
            rosetta_struct_energies_dict, rosetta_res_energies_dict,
            molp_struct_dict, molp_res_dict)


# Calls main() function if betadesigner.py is run as a script
if __name__ == '__main__':
    main()
