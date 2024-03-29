
# If want to change run parameters, set param_opt, orig_sub_gen and max_evals
# parameters - all other parameters controlling the GA are set in the input file

import argparse
import copy
import math
import os
import pandas as pd
import pickle
import time
from collections import OrderedDict
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

    start = time.time()
    # Reads in command line inputs
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_file', help='OPTIONAL: Specifies the '
                        'absolute file path of an input file listing run '
                        'params')
    args = parser.parse_args()

    # Determines whether or not automatic parameter optimisation with hyperopt
    # is performed
    param_opt = False

    # Defines program params from input file / user input
    params = find_params(args, param_opt)
    params['paramopt'] = param_opt

    # Checks none of the hyperparameters to be optimised with hyperopt have
    # been defined as fixed values
    if param_opt is True:
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
    # interactions).
    gen_initial_sequences = gen_ga_input(params)
    (initial_network, new_sequences_dict
    ) = gen_initial_sequences.initial_sequences_pipeline()
    params['initialnetwork'] = {'initial_network': initial_network}

    run_ga = True
    max_evals = []  #[10, 30, 100, 300, 1000]  # Number of
    # parameter combinations for hyperopt to try - set to non-empty list if
    # running parameter optimisation
    orig_sub_gen = 10
    sub_gen = copy.deepcopy(orig_sub_gen)  # Number of generations to run the GA
    # with a particular combination of hyperparameters
    max_gen = copy.copy(params['maxnumgenerations'])  # Total number of
    # generations GA can be run for (changing hyperparameters every sub_gen
    # generations)

    opt_cycle_count = 1
    start_gen = 0
    stop_gen = orig_sub_gen
    # Runs GA in subsets of sub_gen generations, until either the output fitness
    # score has converged (to within 1%) or the number of generations exceeds
    # the user-defined threshold
    while run_ga is True:
        orig_sequences_dict = copy.deepcopy(new_sequences_dict)

        # Runs hyperparameter optimisation in increments of sqrt(10) evaluations,
        # until either the selected hyperparameter values have converged (to
        # within 5%) or the number of combinations of hyperparameters tested
        # exceeds 1000
        if param_opt is True:
            hyperparam_count = max_evals[0]
            run_opt = True
        else:
            hyperparam_count = ''
            run_opt = False

        while run_opt is True:
            # Sets the hyperparams unfit_fraction, crossover_probability,
            # mutation_probability, propensity_to_frequency weighting and
            # propensity_weights to uniform ranges between 0 and 1 for optimisation
            # via a tree parzen estimator with hyperopt
            bayes_params = {}
            bayes_params['crossoverprob'] = hp.uniform('crossoverprob', 0, 0.5)
            bayes_params['mutationprob'] = hp.uniform('mutationprob', 0, 1)
            bayes_params['propensityweight'] = hp.uniform('propensityweight', 0, 1)
            bayes_params['unfitfraction'] = hp.uniform('unfitfraction', 0, 1)

            # Pickles parameters not being optimised with hyperopt (unfortunately
            # can't feed dataframe (or series or array) data into a function with
            # hyperopt, so am having to pickle the parameters not being optimised
            # with hyperopt
            bayes_params['optimisationcycle'] = opt_cycle_count
            bayes_params['hyperparam_count'] = hyperparam_count
            updated_params = setup_input_output(
                copy.deepcopy(params), opt_cycle_count, bayes_params['hyperparam_count']
            )
            # DEFINITION OF bayes_params['workingdirectory'] MUST FOLLOW
            # DEFINITION OF updated_params AFTER RUNNING setup_input_output
            bayes_params['workingdirectory'] = updated_params['workingdirectory']

            updated_params['sequencesdict'] = copy.deepcopy(orig_sequences_dict)
            updated_params['startgen'] = start_gen
            updated_params['stopgen'] = stop_gen
            with open('{}/Program_input/Input_params.pkl'.format(
                bayes_params['workingdirectory']), 'wb'
            ) as f:
                pickle.dump(updated_params, f)

            if os.path.isfile('{}/Program_output/Pickled_trials.pkl'.format(
                updated_params['workingdirectory'])):
                pass
            else:
                trials = Trials()
                with open('{}/Program_output/Pickled_trials.pkl'.format(
                    updated_params['workingdirectory']), 'wb') as f:
                    pickle.dump(trials, f)

            # Runs GA with hyperopt
            save_points = range(2, hyperparam_count+1, 1)
            for point in save_points:
                with open('{}/Program_output/Pickled_trials.pkl'.format(
                    updated_params['workingdirectory']), 'rb') as f:
                    trials = pickle.load(f)
                best_params = fmin(fn=run_genetic_algorithm, space=bayes_params,
                                   algo=tpe.suggest, trials=trials, max_evals=point)
                with open('{}/Program_output/Pickled_trials.pkl'.format(
                    updated_params['workingdirectory']), 'wb') as f:
                    pickle.dump(trials, f)

                trial = trials.trials[-1]
                for trial in trials.trials:
                    # Record trial_id, hyperparameter values and corresponding loss
                    trial_id = '{}_{}_{}'.format(opt_cycle_count, hyperparam_count, trial['tid'])
                    crossover_prob = trial['misc']['vals']['crossoverprob'][0]
                    mutation_prob = trial['misc']['vals']['mutationprob'][0]
                    prop_weight = trial['misc']['vals']['propensityweight'][0]
                    unfit_frac = trial['misc']['vals']['unfitfraction'][0]
                    out = trial['result']['loss']
                    if not os.path.isfile('{}/Program_output/Hyperparameter_track.txt'.format(
                        updated_params['workingdirectory']
                    )):
                        with open('{}/Program_output/Hyperparameter_track.txt'.format(
                            updated_params['workingdirectory']), 'w') as f:
                            f.write('')
                    with open('{}/Program_output/Hyperparameter_track.txt'.format(
                        updated_params['workingdirectory']), 'a+') as f:
                        current_lines = f.readlines()
                        if trial['tid'] == 0:
                            f.write('\n\n\nHyperparameter cycle {}\n'.format(hyperparam_count))
                            f.write('trial_id:     crossover_probability, mutation_probability,'
                                    ' propensity_weight, unfit_fraction     output_loss\n')
                        new_line = '{}:     {}, {}, {}, {}     {}\n'.format(trial_id,
                            crossover_prob, mutation_prob, prop_weight, unfit_frac, out
                        )
                        if not new_line in current_lines:
                            f.write(new_line)

            with open('{}/Program_output/Hyperparameter_track.txt'.format(
                updated_params['workingdirectory']), 'a') as f:
                f.write('\nBest_params:\n')
                f.write('crossover_probability: {}\n'.format(best_params['crossoverprob']))
                f.write('mutation_probability: {}\n'.format(best_params['mutationprob']))
                f.write('propensity_weight: {}\n'.format(best_params['propensityweight']))
                f.write('unfit_fraction: {}\n'.format(best_params['unfitfraction']))

            if hyperparam_count == max_evals[0]:
                current_best = copy.deepcopy(best_params)
                hyperparam_count = max_evals[max_evals.index(hyperparam_count)+1]
            else:
                # If best values are within 5% of previous OR number of
                # trials == maximum number specified
                similarity_dict = {}
                for key in list(best_params.keys()):
                    if key in ['unfitfraction', 'crossoverprob', 'mutationprob', 'propensityweight']:
                        if (
                               ((0.95*current_best[key]) <= best_params[key]
                                 <= (1.05*current_best[key]))
                            or hyperparam_count == max_evals[-1]
                        ):
                            similarity_dict[key] = True
                        else:
                            similarity_dict[key] = False
                if all(x is True for x in similarity_dict.values()):
                    run_opt = False
                else:
                    current_best = copy.deepcopy(best_params)
                    hyperparam_count = max_evals[max_evals.index(hyperparam_count)+1]

        if param_opt is False:
            best_params = copy.deepcopy(params)
        best_params['optimisationcycle'] = opt_cycle_count
        best_params['startgen'] = start_gen
        best_params['stopgen'] = stop_gen
        best_params['hyperparam_count'] = '{}final_run'.format(hyperparam_count)
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

        if max_gen <= sub_gen:  # Breaks without comparing fitness scores of
        # current and previous sub-generations (necessary if sub_gens == max_gens)
            run_ga = False
            break

        if sub_gen == orig_sub_gen:  # Optimises hyperparameters for minimum of 2*orig_sub_gen generations
            current_fitness = copy.deepcopy(fitness)
            start_gen = copy.deepcopy(sub_gen)
            sub_gen += orig_sub_gen
            stop_gen = copy.deepcopy(sub_gen)
            opt_cycle_count += 1
        else:
            # If updated fitness is within 0.1% of previous fitness score OR
            # number of generations > user-defined limit
            if (
                   ((0.999*current_fitness) <= fitness <= (1.001*current_fitness))
                or sub_gen >= max_gen
            ):
                run_ga = False
                break
            else:
                current_fitness = copy.deepcopy(fitness)
                start_gen = copy.deepcopy(sub_gen)
                sub_gen += orig_sub_gen
                stop_gen = copy.deepcopy(sub_gen)
                opt_cycle_count += 1

    # Uses SCWRL4 within ISAMBARD to pack the output sequences onto the
    # backbone model, and writes the resulting model to a PDB file. Also
    # returns each model's total energy within BUDEFF. Currently Rosetta
    # fragment picking is not run.
    with open('{}/Program_output/GA_output_sequences_dict.pkl'.format(
        updated_params['workingdirectory']), 'rb') as f:
        sequences_dict = pickle.load(f)
    best_bayes_params = {'unfitfraction': best_params['unfitfraction'],
                         'crossoverprob': best_params['crossoverprob'],
                         'mutationprob': best_params['mutationprob'],
                         'propensityweight': best_params['propensityweight'],
                         'sequencesdict': sequences_dict}
    output = gen_output(updated_params, best_bayes_params)
    (updated_sequences_dict, structures_list, bude_struct_energies_dict
    ) = output.write_pdb(sequences_dict)
    (rosetta_struct_energies_dict, rosetta_res_energies_dict
    ) = output.score_pdb_rosetta(structures_list)
    """
    (worst_best_frag_dict, num_frag_dict, frag_cov_dict
    ) = output.calc_rosetta_frag_coverage(structures_list)
    """
    (molp_struct_df, molp_res_dict
    ) = output.score_pdb_molprobity(structures_list)

    with open('{}/Program_output/GA_output_sequences_program_output.pkl'.format(
        updated_params['workingdirectory']), 'wb') as f:
        pickle.dump(updated_sequences_dict, f)
    with open('{}/BetaDesigner_results/{}/GA_output_sequences_dict.pkl'.format(
        params['workingdirectory'], params['jobid']), 'wb') as f:
        pickle.dump(updated_sequences_dict, f)

    # Saves per-structure scores in a dataframe
    bude_struct_list = []
    rosetta_struct_list = []
    """
    worst_best_frag_list = []
    num_frag_list = []
    frag_cov_list = []
    """
    for struct in molp_struct_df['Structure_id'].tolist():
        bude_struct_list.append(bude_struct_energies_dict[struct])
        rosetta_struct_list.append(rosetta_struct_energies_dict[struct])
        """
        worst_best_frag_list.append(worst_best_frag_dict[struct])
        num_frag_list.append(num_frag_dict[struct])
        frag_cov_list.append(frag_cov_dict[struct])
        """
    per_struct_scores_df = pd.DataFrame(OrderedDict({
        'BUDE_score': bude_struct_list,
        'Rosetta_score': rosetta_struct_list}))
    """
        'Worst_best_frag': worst_best_frag_list,
        'Num_frag': num_frag_list,
        'Frag_cov': frag_cov_list
    }))
    """
    per_struct_scores_df = pd.concat([molp_struct_df, per_struct_scores_df], axis=1)
    per_struct_scores_df.to_pickle('{}/Program_output/Per_struct_scores.pkl'.format(
        updated_params['workingdirectory']))
    per_struct_scores_df.to_pickle('{}/BetaDesigner_results/{}/Per_struct_scores.pkl'.format(
        params['workingdirectory'], params['jobid']))

    # Saves per-residue scores as a dictionary of dataframes (one dataframe per
    # structure)
    per_res_scores_dict = OrderedDict()
    for struct, molp_res_df in molp_res_dict.items():
        rosetta_res_list = []
        for res in molp_res_df['Residue_id'].tolist():
            energy = rosetta_res_energies_dict[struct][res]
            rosetta_res_list.append(energy)
        per_res_scores_df = pd.DataFrame({'Rosetta_score': rosetta_res_list})
        per_res_scores_df = pd.concat([molp_res_df, per_res_scores_df], axis=1)
        per_res_scores_dict[struct] = per_res_scores_df
    with open('{}/Program_output/Per_res_scores.pkl'.format(
        updated_params['workingdirectory']), 'wb') as f:
        pickle.dump(per_res_scores_dict, f)
    with open('{}/BetaDesigner_results/{}/Per_res_scores.pkl'.format(
        params['workingdirectory'], params['jobid']), 'wb') as f:
        pickle.dump(per_res_scores_dict, f)

    end = time.time()
    print('Time for GA to run ({} sequences, {} generations, {} '
          'hyperparameter combinations, split propensity and BUDE '
          'measurements): {}'.format(len(updated_sequences_dict), sub_gen, hyperparam_count, end-start))


# Calls main() function if betadesigner.py is run as a script
if __name__ == '__main__':
    main()
