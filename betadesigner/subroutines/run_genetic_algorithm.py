
# Do I need to worry about genetic algorithm overfitting? NO - GAs, like other
# optimisation techniques, can't be overfit. You can get overfitting if using
# the optimisation technique on something which can be overfit, e.g. for
# hyperpameter selection for an ML algorithm, but this is not the case here.

import copy
import os
import pickle
import random
import string
import multiprocessing as mp
import networkx as nx
import numpy as np
from collections import OrderedDict
from operator import itemgetter

if __name__ == 'subroutines.run_genetic_algorithm':
    from subroutines.calc_propensity_in_parallel import linear_interpolation
    from subroutines.find_parameters import initialise_ga_object
    from subroutines.generate_initial_sequences import (
        random_shuffle, propensity_to_probability_distribution,
        frequency_to_probability_distribution, gen_cumulative_probabilities
    )
    from subroutines.variables import three_to_one_aa_dict
else:
    from betadesigner.subroutines.calc_propensity_in_parallel import linear_interpolation
    from betadesigner.subroutines.find_parameters import initialise_ga_object
    from betadesigner.subroutines.generate_initial_sequences import (
        random_shuffle, propensity_to_probability_distribution,
        frequency_to_probability_distribution, gen_cumulative_probabilities
    )
    from betadesigner.subroutines.variables import three_to_one_aa_dict

aa_code_dict = three_to_one_aa_dict()


class run_ga_calcs(initialise_ga_object):
    """
    Functions required to run each stage of the GA (measuring fitness,
    selecting a mating population, performing crossover and mutation, and
    merging the parent and child generations)
    """

    def __init__(self, params, test=False):
        initialise_ga_object.__init__(self, params, test)
        self.propensity_weight = params['propensityweight']
        self.unfit_fraction = params['unfitfraction']
        self.crossover_prob = params['crossoverprob']
        self.mutation_prob = params['mutationprob']

    def measure_fitness_propensity(self, networks_dict):
        """
        Measures fitness of amino acid sequences from their propensities for
        the structural features of the input backbone structure.
        """

        print('Measuring network fitness')

        dict_list = []
        for label, scale in self.propensity_dicts.items():
            weight = self.dict_weights[label]
            dict_list.append((label, weight, scale))
        for label, scale in self.frequency_dicts.items():
            weight = self.dict_weights[label]
            dict_list.append((label, weight, scale))

        with open('{}/Networks_dict.pkl'.format(self.working_directory), 'wb') as f:
            pickle.dump(networks_dict, f)
        with open('{}/Prop_freq_dicts.pkl'.format(self.working_directory), 'wb') as f:
            pickle.dump(dict_list, f)
        with open('{}/Dict_name_indices.pkl'.format(self.working_directory), 'wb') as f:
            pickle.dump(self.dict_name_indices, f)

        os.system('python -m scoop {}/calc_propensity_in_parallel.py '
                  '-net {}/Networks_dict.pkl -dicts {}/Prop_freq_dicts.pkl '
                  '-indices {}/Dict_name_indices.pkl -bos {} -o {} -aa {}'.format(
                  os.path.dirname(os.path.abspath(__file__)),
                  self.working_directory, self.working_directory,
                  self.working_directory, self.barrel_or_sandwich,
                  self.working_directory, ','.join(self.aa_list)))

        os.remove('{}/Networks_dict.pkl'.format(self.working_directory))
        os.remove('{}/Prop_freq_dicts.pkl'.format(self.working_directory))
        os.remove('{}/Dict_name_indices.pkl'.format(self.working_directory))
        with open('{}/Network_prop_freq_scores.pkl'.format(self.working_directory), 'rb') as f:
            network_prop_scores, network_freq_scores = pickle.load(f)
        os.remove('{}/Network_prop_freq_scores.pkl'.format(self.working_directory))

        return network_prop_scores, network_freq_scores

    def combine_prop_and_freq_scores(
        self, network_prop_scores, network_freq_scores, raw_or_rank
    ):
        """
        Combines propensity and frequency scores
        """

        for index, network_score_dict in enumerate([network_prop_scores, network_freq_scores]):
            # Only need to order propensities if converting to probability
            # scores via their rank
            if index == 0:
                prop_or_freq = 'propensity'
                if raw_or_rank == 'rank':
                    # Have taken -ve logarithm of propensity scores, so lower
                    # scores are more likely
                    network_score_dict = OrderedDict(sorted(
                        network_score_dict.items(), key=itemgetter(1), reverse=True
                    ))
            elif index == 1:
                prop_or_freq = 'frequency'
                # Lower scores are more likely (smaller difference between
                # actual and expected frequency distribution)
                network_score_dict = OrderedDict(sorted(
                    network_score_dict.items(), key=itemgetter(1), reverse=True
                ))

            network_num = np.array(list(network_score_dict.keys()))
            network_scores = np.array(list(network_score_dict.values()))

            if index == 0 and raw_or_rank == 'raw':
                (network_num, network_scores, network_prob
                ) = propensity_to_probability_distribution(
                    network_num, network_scores
                )
            elif index == 0 and raw_or_rank == 'rank':
                (network_num, network_scores, network_prob
                ) = frequency_to_probability_distribution(
                    network_num, network_scores, prop_or_freq
                )
            elif index == 1:
                if self.frequency_dicts != {}:
                    (network_num, network_scores, network_prob
                    ) = frequency_to_probability_distribution(
                        network_num, network_scores, 'propensity'
                    )  # Set to "propensity" to make sure converts frequency scores into rank values
                else:
                    network_prob = np.full(network_num.shape, 0)

            if prop_or_freq == 'propensity':
                propensity_array = np.array([copy.deepcopy(network_num),
                                             copy.deepcopy(network_prob)])
            elif prop_or_freq == 'frequency':
                frequency_array = np.array([copy.deepcopy(network_num),
                                             copy.deepcopy(network_prob)])

        network_fitness_scores = OrderedDict()
        for index_prop, network_num in np.ndenumerate(propensity_array[0]):
            index_prop = index_prop[0]
            index_freq = np.where(frequency_array[0] == network_num)[0][0]

            propensity = float(propensity_array[1][index_prop])
            frequency = float(frequency_array[1][index_freq])

            probability = (  (self.propensity_weight*propensity)
                           + ((1-self.propensity_weight)*frequency))
            network_fitness_scores[network_num] = probability

        return network_fitness_scores

    def measure_fitness_allatom(self, networks_dict):
        """
        Measures fitness of sequences using an all-atom scoring function
        within BUDE. Calls function in separate script in order that these
        calculations can be parallelised using scoop
        """

        print('Measuring network fitness')

        with open('{}/Networks_dict.pkl'.format(self.working_directory), 'wb') as f:
            pickle.dump(networks_dict, f)
        os.system('python -m scoop {}/calc_bude_energy_in_parallel.py -pdb {} '
                  '-net {}/Networks_dict.pkl -o {}'.format(
                  os.path.dirname(os.path.abspath(__file__)), self.input_pdb,
                  self.working_directory, self.working_directory))
        os.remove('{}/Networks_dict.pkl'.format(self.working_directory))
        with open('{}/Network_energies.pkl'.format(self.working_directory), 'rb') as f:
            network_energies = pickle.load(f)
        os.remove('{}/Network_energies.pkl'.format(self.working_directory))

        return network_energies

    def convert_energies_to_probabilities(self, network_energies):
        """
        Converts energy values output from BUDE into probabilities.
        Unfortunately can't use the Boltzmann distribution (e^(ΔE / kT)) because
        values output are too large to be handled by numpy, so instead have to
        compare score rankings
        """

        # Lower scores are more likely
        network_energies = OrderedDict(sorted(
            network_energies.items(), key=itemgetter(1), reverse=True
        ))
        network_num = np.array(list(network_energies.keys()))
        network_scores = np.array(list(network_energies.values()))

        (network_num, network_scores, network_prob
        ) = frequency_to_probability_distribution(
            network_num, network_scores, 'propensity'
        )

        network_fitness_scores = OrderedDict(zip(network_num, network_prob))

        return network_fitness_scores

    def measure_fitness_clashscore(self, networks_dict):
        """
        Measures fitness of sequences via calculating the structures clashscore
        with MolProbity. Calls function in separate script in order that these
        calculations can be parallelised using scoop
        """

        print('Measuring network fitness')

        with open('{}/Networks_dict.pkl'.format(self.working_directory), 'wb') as f:
            pickle.dump(networks_dict, f)
        os.system('python -m scoop {}/calc_clashscore_in_parallel.py -pdb {} '
                  '-net {}/Networks_dict.pkl -o {}'.format(
                  os.path.dirname(os.path.abspath(__file__)), self.input_pdb,
                  self.working_directory, self.working_directory))
        os.remove('{}/Networks_dict.pkl'.format(self.working_directory))
        with open('{}/Network_clashes.pkl'.format(self.working_directory), 'rb') as f:
            network_clashes = pickle.load(f)
        os.remove('{}/Network_clashes.pkl'.format(self.working_directory))

        return network_clashes

    def convert_clashscores_to_probabilities(self, network_clashes):
        """
        Converts MolProbity clashscores into probabilities by comparing their
        rankings
        """

        # Lower scores are more likely
        network_clashes = OrderedDict(sorted(
            network_clashes.items(), key=itemgetter(1), reverse=True
        ))
        network_num = np.array(list(network_clashes.keys()))
        network_scores = np.array(list(network_clashes.values()))

        (network_num, network_scores, network_prob
        ) = frequency_to_probability_distribution(
            network_num, network_scores, 'propensity'
        )

        network_fitness_scores = OrderedDict(zip(network_num, network_prob))

        return network_fitness_scores

    def create_mat_pop_fittest(
        self, networks_dict, network_fitness_scores, pop_size, unfit_fraction,
        test=False, random_nums=[]
    ):
        """
        Creates mating population from the fittest sequences plus a subset of
        less fit sequences (so as to maintain diversity in the mating
        population in order to prevent convergence on a non-global minimum)
        """

        print('Creating mating population')

        # Determines numbers of fittest and random sequences to be added in
        unfit_pop_size = round((pop_size*unfit_fraction), 0)
        pop_size -= unfit_pop_size

        # Initialises dictionary of fittest networks
        mating_pop_dict = OrderedDict()

        # Sorts networks by their fitness values, from most (largest
        # probability) to least (smallest probability) fit
        network_fitness_scores = OrderedDict(sorted(
            network_fitness_scores.items(), key=itemgetter(1), reverse=True
        ))

        # Adds fittest individuals to mating population
        for index, num in enumerate(list(network_fitness_scores.keys())):
            if index < pop_size:
                mating_pop_dict[num] = copy.deepcopy(networks_dict[num])
                network_fitness_scores[num] = ''
            else:
                break

        # Removes fittest networks already included in mating population from
        # dictionary
        unfit_network_indices = [num for num, fitness in
                                 network_fitness_scores.items() if fitness != '']

        # Adds unfit individuals (selected at random) to mating population
        count = 0
        while count < unfit_pop_size:
            if test is False:
                random_index = random.randint(0, (len(unfit_network_indices)-1))
            else:
                random_index = random_nums[count]
            network_num = unfit_network_indices[random_index]
            mating_pop_dict[network_num] = copy.deepcopy(networks_dict[network_num])
            unfit_network_indices = [num for num in unfit_network_indices
                                     if num != network_num]
            count += 1

        return mating_pop_dict

    def create_mat_pop_roulette_wheel(
        self, networks_dict, network_fitness_scores, pop_size, test=False,
        random_nums=[]
    ):
        """
        Creates mating population from individuals, with the likelihood of
        selection of each sequence being weighted by its raw fitness score
        """

        print('Creating mating population')

        # Initialises dictionary of fittest networks
        mating_pop_dict = OrderedDict()

        # Adds individuals (their likelihood of selection weighted by their raw
        # fitness scores) to mating population. Arrays of network numbers,
        # fitness scores and their corresponding probability distribution are
        # updated every cycle to remove the selected network (this prevents the
        # loop from taking a very long time once the highest probability
        # networks have been selected)
        count = 0
        while count < pop_size:
            network_num_array = (
                np.array(list(network_fitness_scores.keys()))
            )
            network_fitness_array = (
                np.array(list(network_fitness_scores.values()))
            )
            network_cumulative_prob = gen_cumulative_probabilities(
                network_fitness_array, ' sequence networks', adjust_scale=True
            )

            if test is False:
                random_number = random.uniform(0, 1)
            else:
                random_number = random_nums[count]
            nearest_index = (np.abs(network_cumulative_prob-random_number)).argmin()

            if network_cumulative_prob[nearest_index] < random_number:
                nearest_index += 1

            selected_network_num = network_num_array[nearest_index]
            mating_pop_dict[selected_network_num] = copy.deepcopy(
                networks_dict[selected_network_num]
            )

            if count != (pop_size - 1):
                network_fitness_scores = OrderedDict(
                    zip(np.delete(network_num_array, nearest_index),
                        np.delete(network_fitness_array, nearest_index))
                )

            count += 1

        return mating_pop_dict

    def uniform_crossover(
        self, mating_pop_dict, test=False, pairs=[], crossover_prob={}
    ):
        """
        Selects pairs of individuals at random from mating population and
        performs uniform crossover
        """

        print('Performing crossovers')

        # Initialises dictionary of child networks
        crossover_pop_dict = OrderedDict()

        if test is False:
            # Selects pairs of networks at random to crossover with each other
            network_num = list(mating_pop_dict.keys())
            random.shuffle(network_num)
            network_num = iter(network_num)  # Do not merge with line below,
            # and do not introduce any lines of code between them!
            network_num = list(zip(network_num, network_num))
        else:
            network_num = pairs

        # Performs uniform crossover
        for index, network_pair in enumerate(network_num):
            network_num_1 = network_pair[0]
            network_num_2 = network_pair[1]
            mate_1 = copy.deepcopy(mating_pop_dict[network_num_1])
            mate_2 = copy.deepcopy(mating_pop_dict[network_num_2])

            for node in list(mate_1.nodes):
                type_1 = mate_1.nodes()[node]['type']
                type_2 = mate_2.nodes()[node]['type']
                if type_1 != type_2:
                    raise TypeError(
                        'Difference between type of {} in {} ({} = {}; {} ='
                        ' {}) - should be identical'.format(node, network_pair,
                        network_num_1, type_1, network_num_2, type_2)
                    )
                if type_1 == 'loop':
                    continue

                if test is False:
                    random_number = random.uniform(0, 1)
                else:
                    random_number = crossover_prob[index][node]

                if random_number <= self.crossover_prob:
                    # Copy to prevent these dictionaries from updating when the
                    # node attributes are updated in the code below (otherwise
                    # both nodes will be assigned the same identity as the node
                    # in mate_1, instead of the node identities being crossed
                    # over)
                    mate_1_node_attributes = copy.deepcopy(mate_1.nodes()[node])
                    mate_2_node_attributes = copy.deepcopy(mate_2.nodes()[node])
                    # mate_1.nodes()[node] = {} does not work, get
                    # TypeError: 'NodeView' object does not support item assignment
                    for attribute in list(mate_1.nodes()[node].keys()):
                        del mate_1.nodes()[node][attribute]
                    for attribute in list(mate_2.nodes()[node].keys()):
                        del mate_2.nodes()[node][attribute]
                    nx.set_node_attributes(mate_1, values={node: mate_2_node_attributes})
                    nx.set_node_attributes(mate_2, values={node: mate_1_node_attributes})

            crossover_pop_dict[network_num_1] = mate_1
            crossover_pop_dict[network_num_2] = mate_2

        return crossover_pop_dict

    def segmented_crossover(
        self, mating_pop_dict, test=False, pairs=[], crossover_prob={}
    ):
        """
        Selects pairs of individuals at random from mating population and
        performs segmented crossover
        """

        print('Performing crossovers')

        # Initialises dictionary of child networks
        crossover_pop_dict = OrderedDict()

        if test is False:
            # Selects pairs of networks at random to crossover with each other
            network_num = list(mating_pop_dict.keys())
            random.shuffle(network_num)
            network_num = iter(network_num)  # Do not merge with line below,
            # and do not introduce any lines of code between them!
            network_num = list(zip(network_num, network_num))
        else:
            network_num = pairs

        # Performs segmented crossover
        for index, network_pair in enumerate(network_num):
            network_num_1 = network_pair[0]
            network_num_2 = network_pair[1]
            mate_1 = copy.deepcopy(mating_pop_dict[network_num_1])
            mate_2 = copy.deepcopy(mating_pop_dict[network_num_2])

            swap = False
            for node in list(mate_1.nodes):
                type_1 = mate_1.nodes()[node]['type']
                type_2 = mate_2.nodes()[node]['type']
                if type_1 != type_2:
                    raise TypeError(
                        'Difference between type of {} in {} ({} = {}; {} ='
                        ' {}) - should be identical'.format(node, network_pair,
                        network_num_1, type_1, network_num_2, type_2)
                    )
                if type_1 == 'loop':
                    continue

                if test is False:
                    random_number = random.uniform(0, 1)
                else:
                    random_number = crossover_prob[index][node]

                if swap is False:
                    if random_number <= self.swap_start_prob:
                        swap = True
                    else:
                        swap = False
                elif swap is True:
                    if random_number <= self.swap_stop_prob:
                        swap = False
                    else:
                        swap = True

                if swap is True:
                    # Copy to prevent these dictionaries from updating when the
                    # node attributes are updated in the code below (otherwise
                    # both nodes will be assigned the same identity as the node
                    # in mate_1, instead of the node identities being crossed
                    # over)
                    mate_1_attributes = copy.deepcopy(mate_1.nodes()[node])
                    mate_2_attributes = copy.deepcopy(mate_2.nodes()[node])
                    # mate_1.nodes()[node] = {} does not work, get
                    # TypeError: 'NodeView' object does not support item assignment
                    for attribute in list(mate_1.nodes()[node].keys()):
                        del mate_1.nodes()[node][attribute]
                    for attribute in list(mate_2.nodes()[node].keys()):
                        del mate_2.nodes()[node][attribute]
                    nx.set_node_attributes(mate_1, values={node: mate_2_attributes})
                    nx.set_node_attributes(mate_2, values={node: mate_1_attributes})

            crossover_pop_dict[network_num_1] = mate_1
            crossover_pop_dict[network_num_2] = mate_2

        return crossover_pop_dict

    def swap_mutate(
        self, crossover_pop_dict, test=False, mutation_prob={}, random_aas=''
    ):
        """
        Performs swap mutations (= mutates randomly selected individual
        network nodes to a randomly selected (different) amino acid identity)
        """

        print('Performing mutations')

        # Initialises dictionary of mutated child networks
        mutated_pop_dict = OrderedDict()

        # Mutates the amino acid identities of randomly selected nodes
        for network_num in list(crossover_pop_dict.keys()):
            G = copy.deepcopy(crossover_pop_dict[network_num])

            for node in list(G.nodes):
                if G.nodes()[node]['type'] == 'loop':
                    continue

                if test is False:
                    random_number = random.uniform(0, 1)
                else:
                    random_number = mutation_prob[network_num][node]
                if random_number <= self.mutation_prob:
                    if test is False:
                        orig_aa = G.nodes()[node]['aa_id']
                        poss_aas = copy.deepcopy(self.aa_list)
                        poss_aas.remove(orig_aa)
                        new_aa = poss_aas[random.randint(0, (len(poss_aas)-1))]
                    else:
                        new_aa = random_aas[0]
                        random_aas = random_aas[1:]

                    nx.set_node_attributes(G, values={node: {'aa_id': new_aa}})

            mutated_pop_dict[network_num] = G

        return mutated_pop_dict

    def scramble_mutate(
        self, crossover_pop_dict, test=False, mutation_prob={}
    ):
        """
        Performs scramble mutations (= scrambles the identities of a subset
        of amino acids selected at random)
        """

        print('Performing mutations')

        # Initialises dictionary of mutated child networks
        mutated_pop_dict = OrderedDict()

        # Scrambles the amino acid identities of randomly selected nodes
        for network_num in list(crossover_pop_dict.keys()):
            G = copy.deepcopy(crossover_pop_dict[network_num])

            scrambled_nodes = []
            aa_ids = []
            for node in list(G.nodes):
                if G.nodes()[node]['type'] == 'loop':
                    continue

                if test is False:
                    random_number = random.uniform(0, 1)
                else:
                    random_number = mutation_prob[network_num][node]
                if random_number <= self.mutation_prob:
                    scrambled_nodes.append(node)
                    aa_ids.append(G.nodes()[node]['aa_id'])

            if test is False:
                random.shuffle(aa_ids)
            else:
                aa_ids = aa_ids[::-1]
            attributes = OrderedDict({
                node: {'aa_id': aa_id} for node, aa_id in zip(scrambled_nodes, aa_ids)
            })
            nx.set_node_attributes(G, values=attributes)

            mutated_pop_dict[network_num] = G

        return mutated_pop_dict

    def add_children_to_parents(self, mutated_pop_dict, mating_pop_dict):
        """
        Combines parent and child generations
        """

        print('Combining parent and child generations')

        merged_networks_dict = OrderedDict()

        for id, G in mutated_pop_dict.items():
            new_id = ''.join(
                [random.choice(string.ascii_letters + string.digits)
                for i in range(10)]
            )
            merged_networks_dict[new_id] = copy.deepcopy(G)
        for id, G in mating_pop_dict.items():
            merged_networks_dict[id] = copy.deepcopy(G)

        return merged_networks_dict


def run_genetic_algorithm(bayes_params):
    """
    Pipeline function to run genetic algorithm
    """

    print('Running genetic algorithm')

    # Unpacks parameters (unfortunately can't feed dataframe (or series or
    # array) data into a function with hyperopt, so am having to pickle the
    # parameters not being optimised with hyperopt
    params_file = '{}/Program_input/Input_params.pkl'.format(
        bayes_params['workingdirectory']
    )
    with open(params_file, 'rb') as f:
        fixed_params = pickle.load(f)
    if not type(fixed_params) in [dict, OrderedDict]:
        raise TypeError('Data in {} is not a pickled dictionary'.format(params_file))
    params = {**bayes_params, **fixed_params}

    # Records sequences and their fitnesses after each generation
    with open('{}/Program_output/Sequence_track.txt'.format(
        bayes_params['workingdirectory']), 'w') as f:
        f.write('Tracking GA optimisation progress\n')

    ga_calcs = run_ga_calcs(params)

    # Defines whether sequences are compared by their raw or rank propensities.
    # Since BUDE scores and frequency values have to be compared by their rank
    # values, have made the decision to also compare propensity values by their
    # rankings.
    """
    if params['matingpopmethod'] in ['fittest', 'roulettewheel']:
        raw_or_rank = 'raw'
    elif params['matingpopmethod'] in ['rankroulettewheel']:
        raw_or_rank = 'rank'
    """
    raw_or_rank = 'rank'

    # Calculates propensity and/or BUDE energy of input structure
    with open('{}/Program_output/Sequence_track.txt'.format(
        bayes_params['workingdirectory']), 'a') as f:
        f.write('Input structure\n')

    if params['fitnessscoremethod'] == 'alternate':
        (network_propensity_scores, network_frequency_scores
        ) = ga_calcs.measure_fitness_propensity(params['initialnetwork'])

        with open('{}/Program_output/Sequence_track.txt'.format(
            bayes_params['workingdirectory']), 'a') as f:
            f.write('network_id, sequence, propensity_score, frequency_score,'
                    ' BUDE energy, clashscore\n')
            for network, G in params['initialnetwork'].items():
                sequence = ''.join([G.nodes()[node]['aa_id'] for node in G.nodes()])
                propensity = network_propensity_scores[network]
                frequency = network_frequency_scores[network]
                f.write('{}, {}, {}, {}, {}, {}\n'.format(
                    network, sequence, propensity, frequency,
                    params['inputpdbenergy'], params['inputpdbclash']
                ))
            f.write('\n')

    if params['fitnessscoremethod'] == 'propensity':
        (network_propensity_scores, network_frequency_scores
        ) = ga_calcs.measure_fitness_propensity(params['initialnetwork'])

        with open('{}/Program_output/Sequence_track.txt'.format(
            bayes_params['workingdirectory']), 'a') as f:
            f.write('network_id, sequence, propensity_score, frequency_score\n')
            for network, G in params['initialnetwork'].items():
                sequence = ''.join([G.nodes()[node]['aa_id'] for node in G.nodes()])
                propensity = network_propensity_scores[network]
                frequency = network_frequency_scores[network]
                f.write('{}, {}, {}, {}\n'.format(
                    network, sequence, propensity, frequency
                ))
            f.write('\n')

    elif params['fitnessscoremethod'] == 'allatom':
        network_energies = ga_calcs.measure_fitness_allatom(params['initialnetwork'])

        with open('{}/Program_output/Sequence_track.txt'.format(
            bayes_params['workingdirectory']), 'a') as f:
            f.write('network_id, sequence, BUDE energy\n')
            for network, G in params['initialnetwork'].items():
                sequence = ''.join([G.nodes()[node]['aa_id'] for node in G.nodes()])
                energy = network_energies[network]
                f.write('{}, {}, {}\n'.format(network, sequence, energy))
            f.write('\n')

    elif params['fitnessscoremethod'] == 'molprobity':
        network_clashes = ga_calcs.measure_fitness_clashscore(params['initialnetwork'])

        with open('{}/Program_output/Sequence_track.txt'.format(
            bayes_params['workingdirectory']), 'a') as f:
            f.write('network_id, sequence, clashscore\n')
            for network, G in params['initialnetwork'].items():
                sequence = ''.join([G.nodes()[node]['aa_id'] for node in G.nodes()])
                clashscore = network_clashes[network]
                f.write('{}, {}, {}\n'.format(network, sequence, clashscore))
            f.write('\n')

    # Runs GA cycles
    gen = params['startgen']
    while gen < params['stopgen']:
        gen += 1
        print('Generation {}'.format(gen))
        with open('{}/Program_output/Sequence_track.txt'.format(
            bayes_params['workingdirectory']), 'a') as f:
            f.write('\n\n\n\n\nGeneration {}\n'.format(gen))


        all_networks_list = [params['sequencesdict']]
        pop_sizes = [params['populationsize']]

        for index, networks_dict in enumerate(all_networks_list):
            # Measures fitness of sequences in starting population.
            if (
                (params['fitnessscoremethod'] == 'propensity')
                or
                (params['fitnessscoremethod'] == 'alternate' and gen % 2 == 1)
            ):
                (network_propensity_scores, network_frequency_scores
                ) = ga_calcs.measure_fitness_propensity(networks_dict)
                network_fitness_scores = ga_calcs.combine_prop_and_freq_scores(
                    network_propensity_scores, network_frequency_scores, raw_or_rank
                )

                # Records sequences output from this generation and their
                # associated fitnesses
                with open('{}/Program_output/Sequence_track.txt'.format(
                    bayes_params['workingdirectory']), 'a') as f:
                    f.write('network, sequence, propensity, frequency, probability\n')
                    for network, G in networks_dict.items():
                        sequence = ''.join([G.nodes()[node]['aa_id'] for node in G.nodes()])
                        propensity = network_propensity_scores[network]
                        frequency = network_frequency_scores[network]
                        probability = network_fitness_scores[network]
                        f.write('{}, {}, {}, {}, {}\n'.format(
                            network, sequence, propensity, frequency, probability
                        ))
                    f.write('Total: {}, {}, {}'.format(
                        sum(network_propensity_scores.values()),
                        sum(network_frequency_scores.values()),
                        sum(network_fitness_scores.values())
                    ))
                    f.write('\n')
            elif (
                (params['fitnessscoremethod'] == 'allatom')
                or
                (params['fitnessscoremethod'] == 'alternate' and gen % 4 == 2)
            ):
                # Runs BUDE energy scoring on parallel processors
                network_energies = ga_calcs.measure_fitness_allatom(networks_dict)
                (network_fitness_scores
                ) = ga_calcs.convert_energies_to_probabilities(network_energies)

                # Records sequences output from this generation and their
                # associated fitnesses
                with open('{}/Program_output/Sequence_track.txt'.format(
                    bayes_params['workingdirectory']), 'a') as f:
                    f.write('network, sequence, BUDE score, probability\n')
                    for network, G in networks_dict.items():
                        sequence = ''.join([G.nodes()[node]['aa_id'] for node in G.nodes()])
                        energy = network_energies[network]
                        probability = network_fitness_scores[network]
                        f.write('{}, {}, {}, {}\n'.format(
                            network, sequence, energy, probability
                        ))
                    f.write('Total: {}, {}'.format(
                        sum(network_energies.values()),
                        sum(network_fitness_scores.values())
                    ))
                    f.write('\n')

            elif (
                (params['fitnessscoremethod'] == 'molprobity')
                or
                (params['fitnessscoremethod'] == 'alternate' and gen % 4 == 0)
            ):
                # Runs MolProbity scoring on parallel processors
                network_clashes = ga_calcs.measure_fitness_clashscore(networks_dict)
                (network_fitness_scores
                ) = ga_calcs.convert_clashscores_to_probabilities(network_clashes)

                # Records sequences output from this generation and their
                # associated fitnesses
                with open('{}/Program_output/Sequence_track.txt'.format(
                    bayes_params['workingdirectory']), 'a') as f:
                    f.write('network, sequence, clashscore, probability\n')
                    for network, G in networks_dict.items():
                        sequence = ''.join([G.nodes()[node]['aa_id'] for node in G.nodes()])
                        clash = network_clashes[network]
                        probability = network_fitness_scores[network]
                        f.write('{}, {}, {}, {}\n'.format(
                            network, sequence, clash, probability
                        ))
                    f.write('Total: {}, {}'.format(
                        sum(network_clashes.values()),
                        sum(network_fitness_scores.values())
                    ))
                    f.write('\n')

            # Selects subpopulation for mating
            if params['matingpopmethod'] == 'fittest':
                mating_pop_dict = ga_calcs.create_mat_pop_fittest(
                    networks_dict, network_fitness_scores, pop_sizes[index],
                    params['unfitfraction']
                )
            elif params['matingpopmethod'] in ['roulettewheel', 'rankroulettewheel']:
                mating_pop_dict = ga_calcs.create_mat_pop_roulette_wheel(
                    networks_dict, network_fitness_scores, pop_sizes[index], params['']
                )

            # Performs crossover of parent sequences to generate child sequences
            if params['crossovermethod'] == 'uniform':
                crossover_pop_dict = ga_calcs.uniform_crossover(mating_pop_dict)
            elif params['crossovermethod'] == 'segmented':
                crossover_pop_dict = ga_calcs.segmented_crossover(mating_pop_dict)

            # Mutates child sequences
            if params['mutationmethod'] == 'swap':
                mutated_pop_dict = ga_calcs.swap_mutate(crossover_pop_dict)
            elif params['mutationmethod'] == 'scramble':
                mutated_pop_dict = ga_calcs.scramble_mutate(crossover_pop_dict)

            # Combines parent and child sequences into single generation
            merged_networks_dict = ga_calcs.add_children_to_parents(
                mutated_pop_dict, mating_pop_dict
            )

            random_order = [n for n in range(len(merged_networks_dict))]
            random.shuffle(random_order)
            shuffled_merged_networks_dict = OrderedDict(
                {list(merged_networks_dict.keys())[n]:
                 list(merged_networks_dict.values())[n] for n in random_order}
            )
            params['sequencesdict'] = shuffled_merged_networks_dict

    # Calculates fitness of output sequences and filters population to maintain
    # the fittest 50%, plus sums the probabilities of the retained sequences and
    # returns this value (to be minimised with hyperopt)
    summed_fitness = 0

    with open('{}/Program_output/Sequence_track.txt'.format(
        bayes_params['workingdirectory']), 'a') as f:
        f.write('\n\n\n\n\nOutput generation\n')

    if params['fitnessscoremethod'] != 'allatom':
        (network_propensity_scores, network_frequency_scores
        ) = ga_calcs.measure_fitness_propensity(params['sequencesdict'])
        network_fitness_scores = ga_calcs.combine_prop_and_freq_scores(
            network_propensity_scores, network_frequency_scores, raw_or_rank
        )
    elif params['fitnessscoremethod'] == 'allatom':
        network_energies = ga_calcs.measure_fitness_allatom(params['sequencesdict'])
        (network_fitness_scores
        ) = ga_calcs.convert_energies_to_probabilities(network_energies)

    # Records sequences output from this generation and their associated
    # fitnesses
    with open('{}/Program_output/Sequence_track.txt'.format(
        bayes_params['workingdirectory']), 'a') as f:
        if params['fitnessscoremethod'] != 'allatom':
            f.write('network, sequence, propensity, frequency\n')
        elif params['fitnessscoremethod'] == 'allatom':
            f.write('network, sequence, BUDE score\n')
        for network, G in params['sequencesdict'].items():
            sequence = ''.join([G.nodes()[node]['aa_id'] for node in G.nodes()])
            if params['fitnessscoremethod'] != 'allatom':
                propensity = network_propensity_scores[network]
                frequency = network_frequency_scores[network]
                f.write('{}, {}, {}, {}\n'.format(
                    network, sequence, propensity, frequency
                ))
            elif params['fitnessscoremethod'] == 'allatom':
                energy = network_energies[network]
                f.write('{}, {}, {}\n'.format(network, sequence, energy))
        if params['fitnessscoremethod'] != 'allatom':
            f.write('Total: {}, {}'.format(
                sum(network_propensity_scores.values()),
                sum(network_frequency_scores.values())
            ))
        elif params['fitnessscoremethod'] == 'allatom':
            f.write('Total: {}'.format(sum(network_energies.values())))
        f.write('\n')

    params['sequencesdict'] = ga_calcs.create_mat_pop_fittest(
        params['sequencesdict'], network_fitness_scores,
        params['populationsize'], unfit_fraction=0
    )

    for network in params['sequencesdict'].keys():
        # Higher propensity is more likely, so add because output from
        # measure_fitness_propensity is sum of -log(propensity) values, and
        # hyperopt minimises output score
        # Can't combine propensity and frequency scores without first converting
        # to a probability, so for calculating output combined fitness can only
        # use combined propensity scores to rank the structures
        if params['fitnessscoremethod'] != 'allatom':
            summed_fitness += network_propensity_scores[network]
        # Lower score is more likely, so add because hyperopt minimises output
        # score
        elif params['fitnessscoremethod'] == 'allatom':
            summed_fitness += network_energies[network]

    with open('{}/Program_output/GA_output_sequences_dict.pkl'.format(
        bayes_params['workingdirectory']), 'wb') as f:
        pickle.dump(params['sequencesdict'], f)

    print(summed_fitness)

    return summed_fitness
