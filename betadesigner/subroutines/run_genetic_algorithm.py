
# Do I need to worry about genetic algorithm overfitting? NO - GAs, like other
# optimisation techniques, can't be overfit. You can get overfitting if using
# the optimisation technique on something which can be overfit, e.g. for
# hyperpameter selection for an ML algorithm, but this is not the case here.

import budeff
import copy
import isambard
import random
import sys
import isambard.modelling as modelling
import networkx as nx
import numpy as np
from collections import OrderedDict
from operator import itemgetter

if __name__ == 'subroutines.run_genetic_algorithm':
    from subroutines.find_parameters import initialise_ga_object
    from subroutines.generate_initial_sequences import (
        linear_interpolation, propensity_to_probability_distribution,
        frequency_to_probability_distribution, gen_cumulative_probabilities
    )
    from subroutines.variables import gen_amino_acids_dict
else:
    from betadesigner.subroutines.find_parameters import initialise_ga_object
    from betadesigner.subroutines.generate_initial_sequences import (
        linear_interpolation, propensity_to_probability_distribution,
        frequency_to_probability_distribution, gen_cumulative_probabilities
    )
    from betadesigner.subroutines.variables import gen_amino_acids_dict


def pack_side_chains(ampal_object, G, rigid_rotamers):
    # Uses SCWRL4 to pack network side chains onto a backbone structure and
    # measures the total energy of the model within BUDE

    aa_dict = gen_amino_acids_dict()

    # Makes FASTA sequence to feed into SCWRL4. BEWARE: if FASTA sequence is
    # shorter than AMPAL object, SCWRL4 will add random amino acids to the end
    # of the sequence until it is the same length.
    fasta_seq = ''
    for res in ampal_object.get_monomers():
        res_id = '{}{}{}{}'.format(
            res.parent.parent.id, res.parent.id, res.id, res.insertion_code
        )

        if res_id in list(G.nodes):
            fasta_seq += G.nodes[res_id]['aa_id']
        else:
            fasta_seq += res.mol_letter

    if len(fasta_seq) != len(list(ampal_object.get_monomers())):
        sys.exit('FASTA sequence and AMPAL object contain different numbers '
                 'of amino acids')

    # Packs side chains with SCWRL4. NOTE that fasta sequence must be provided
    # as a list
    new_ampal_object = modelling.pack_side_chains_scwrl(
        ampal_object, [fasta_seq], rigid_rotamer_model=rigid_rotamers,
        hydrogens=False
    )

    # Calculates total energy of the AMPAL object within BUDE (note that this
    # does not include the interaction of the object with its surrounding
    # environment, hence hydrophobic side chains will not be penalised on the
    # surface of a globular protein and vice versa for membrane proteins)
    energy = budeff.get_internal_energy(new_ampal_object).total_energy
    # Calculates the total energy of the AMPAL object within PyRosetta
    pose = pyrosetta.pose_from_pdb(path_to_pdb)
    score_function = pyrosetta.get_fa_scorefxn()
    energy = score_function(pose)

    return new_ampal_object, energy


class run_ga_calcs(initialise_ga_object):

    def __init__(self, params):
        initialise_ga_object.__init__(self, params)

    def measure_fitness_propensity(self, surface, networks_dict):
        # Measures fitness of amino acid sequences from their propensities for
        # the structural features of the input backbone structure.
        print('Measuring {} network fitness'.format(surface))

        # Initialises dictionaries of network propensity and frequency scores
        network_propensity_scores = OrderedDict()
        network_frequency_scores = OrderedDict()

        # Extracts propensity and frequency scales for the surface (both
        # individual and pairwise)
        sub_indv_dicts = OrderedDict({
            dict_label: aa_dict for dict_label, aa_dict in
            {**self.propensity_dicts, **self.frequency_dicts}.items() if
             (    (dict_label.split('_')[0] in [surface[0:3], '-'])
              and (dict_label.split('_')[5] == 'indv'))
        })
        sub_pair_dicts = OrderedDict({
            dict_label: aa_dict for dict_label, aa_dict in
            {**self.propensity_dicts, **self.frequency_dicts}.items() if
             (    (dict_label.split('_')[0] in [surface[0:3], '-'])
              and (dict_label.split('_')[5] == 'pair'))
        })

        for num in list(networks_dict.keys()):
            G = networks_dict[num]
            # Total propensity count (across all nodes in network)
            propensity_count = 0
            frequency_count = 0

            for node_1 in list(G.nodes):
                # Filters propensity and / or frequency scales depending upon
                # whether the node is in an edge or a central strand
                if self.barrel_or_sandwich == '2.60':
                    eoc_1 = G.nodes[node_1]['eoc']

                    sub_indv_dicts = OrderedDict({
                        dict_label: aa_dict for dict_label, aa_dict
                        in sub_indv_dicts.items()
                        if dict_label.split('_')[1] in [eoc_1, '-']
                    })
                    sub_pair_dicts = OrderedDict({
                        dict_label: aa_dict for dict_label, aa_dict
                        in sub_pair_dicts.items()
                        if dict_label.split('_')[1] in [eoc_1, '-']
                    })

                # Calculates interpolated propensity of each node for all
                # individual structural features considered
                aa_1 = G.nodes[node_1]['aa_id']
                for dict_label, scale_dict in sub_indv_dicts.items():
                    weight = self.dict_weights[dict_label]

                    node_prop_1 = dict_label.split('_')[2]
                    node_prop_2 = dict_label.split('_')[3]
                    node_val_1 = np.nan
                    node_val_2 = np.nan

                    if node_prop_1 != '-':
                        node_val_1 = G.nodes[node_1][node_prop_1]
                    if node_prop_2 != '-':
                        node_val_2 = G.nodes[node_1][node_prop_2]
                    if (
                             node_prop_1 == 'phi'
                         and node_prop_2 == 'psi'
                         and dict_label[6] == 'disc'
                    ):
                        node_val_1 = G.nodes[node]['phipsiclass']
                        node_val_2 = np.nan

                    # Converts non-float values into np.nan
                    if node_val_1 in ['', 'nan', 'NaN', np.nan]:
                        node_val_1 = np.nan
                    if node_val_2 in ['', 'nan', 'NaN', np.nan]:
                        node_val_2 = np.nan

                    value = np.nan
                    if dict_label.split('_')[6] == 'cont' and not np.isnan(node_val_1):
                        # Interpolate dictionary
                        if (   (node_prop_2 == '-')
                            or (node_prop_2 != '-' and not np.isnan(node_val_2))
                        ):
                            value = linear_interpolation(
                                node_val_1, scale_dict[aa_1], dict_label,
                                node_val_2
                            )

                    elif dict_label.split('_')[6] == 'disc':
                        # Filter dataframe
                        scale_dict_copy = scale_dict.set_index('FASTA', drop=True)
                        if node_prop_1 == '-' and node_prop_2 == '-':
                            try:
                                value = scale_dict_copy.iloc[:,0][aa_1]
                            except KeyError:
                                pass
                        elif node_prop_1 == 'phi' and node_prop_2 == 'psi':
                            if not np.isnan(node_val_1):
                                try:
                                    value = scale_dict_copy[node_val_1][aa_1]
                                except KeyError:
                                    pass

                    if not np.isnan(value):
                        if dict_label.split('_')[7] == 'propensity':
                            # NOTE: Must take -ve logarithm of each
                            # individual propensity score before summing
                            # (rather than taking the -ve logarithm of the
                            # summed propensities)
                            value = weight*np.negative(np.log(value))
                            propensity_count += value
                        elif dict_label.split('_')[7] == 'frequency':
                            value *= weight
                            frequency_count += value

                # Loops through each node pair to sum pairwise propensity values
                for node_pair in G.edges(node_1):
                    boolean = random.randint(0, 1)
                    if boolean == 0:
                        node_a = node_pair[0]
                        node_b = node_pair[1]
                    elif boolean == 1:
                        node_a = node_pair[1]
                        node_b = node_pair[0]

                    aa_1 = G.nodes[node_a]['aa_id']
                    aa_2 = G.nodes[node_b]['aa_id']
                    aa_pair = '{}_{}'.format(aa_1, aa_2)

                    # Loops through each interaction between a pair of nodes
                    for edge in G[node_a][node_b]:
                        edge_label = G[node_a][node_b][edge]['interaction']

                        for dict_label, scale_dict in sub_pair_dicts.items():
                            if dict_label.split('_')[4] == edge_label:
                                weight = self.dict_weights[dict_label]

                                node_prop_1 = dict_label.split('_')[2]
                                node_prop_2 = dict_label.split('_')[3]
                                node_val_1 = np.nan
                                node_val_2 = np.nan

                                if node_prop_1 != '-':
                                    node_val_1 = G.nodes[node_a][node_prop_1]
                                if node_prop_2 != '-':
                                    node_val_2 = G.nodes[node_a][node_prop_2]

                                # Converts non-float values into np.nan
                                if node_val_1 in ['', 'nan', 'NaN', np.nan]:
                                    node_val_1 = np.nan
                                if node_val_2 in ['', 'nan', 'NaN', np.nan]:
                                    node_val_2 = np.nan

                                value = np.nan
                                if dict_label.split('_')[6] == 'cont' and not np.isnan(node_val_1):
                                    # Interpolate dictionary
                                    if (   (node_prop_2 == '-')
                                        or (node_prop_2 != '-' and not np.isnan(node_val_2))
                                    ):
                                        try:
                                            aa_scale = scale_dict[aa_pair]
                                            value = linear_interpolation(
                                                node_val_1, aa_scale,
                                                dict_label, node_val_2
                                            )
                                        except KeyError:
                                            pass

                                elif dict_label.split('_')[6] == 'disc':
                                    # Filter dataframe
                                    scale_dict_copy = scale_dict.set_index('FASTA', drop=True)
                                    if node_prop_1 == '-' and node_prop_2 == '-':
                                        try:
                                            value = scale_dict_copy[aa_1][aa_2]
                                        except KeyError:
                                            pass

                                if not np.isnan(value):
                                    if dict_label.split('_')[7] == 'propensity':
                                        # NOTE: Must take -ve logarithm of each
                                        # individual propensity score before
                                        # summing (rather than taking the -ve
                                        # logarithm of the summed propensities)
                                        value = weight*np.negative(np.log(value))
                                        propensity_count += value
                                    elif dict_label.split('_')[7] == 'frequency':
                                        value *= weight
                                        frequency_count += value

            network_propensity_scores[num] = propensity_count
            network_frequency_scores[num] = frequency_count

        return network_propensity_scores, network_frequency_scores

    def combine_propensity_and_frequency_scores(self, network_propensity_scores,
                                                network_frequency_scores,
                                                raw_or_rank):
        # Combines propensity and frequency scores
        network_probabilities = []
        for index, network_score_dict in enumerate(
            [network_propensity_scores, network_frequency_scores]
        ):
            if index == 0:
                prop_or_freq = 'propensity'
                sorted_network_dict = OrderedDict(sorted(
                    network_score_dict.items(), key=itemgetter(1), reverse=True
                ))
            elif index == 1:
                prop_or_freq = 'frequency'
                sorted_network_dict = OrderedDict(sorted(
                    network_score_dict.items(), key=itemgetter(1), reverse=False
                ))

            sorted_network_num = np.array(list(sorted_network_dict.keys()))
            sorted_network_scores = np.array(list(sorted_network_dict.values()))

            if index == 0 and raw_or_rank == 'raw':
                (sorted_network_num, sorted_network_scores,
                 sorted_network_probabilities
                ) = propensity_to_probability_distribution(
                    sorted_network_num, sorted_network_scores
                )
            elif (   (index == 0 and raw_or_rank == 'rank')
                  or (index == 1)
            ):
                (sorted_network, sorted_network_scores,
                 sorted_network_probabilities
                ) = frequency_to_probability_distribution(
                    sorted_network_num, sorted_network_scores,
                    prop_or_freq
                )

            network_array = np.array([sorted_network_num, sorted_network_probabilities])
            network_probabilities.append(network_array)

        propensity_array = network_probabilities[0]
        frequency_array = network_probabilities[1]

        network_fitness_scores = OrderedDict()
        for index_prop, network_num in np.ndenumerate(propensity_array[0]):
            index_prop = index_prop[0]
            index_freq = np.where(frequency_array[0] == network_num)[0][0]

            propensity = propensity_array[1][index_prop]
            frequency = frequency_array[1][index_freq]

            probability = (  (self.propensity_weight['propensity']*propensity)
                           + (self.propensity_weight['frequency']*frequency))
            network_fitness_scores[network_num] = probability

        return network_fitness_scores

    def measure_fitness_all_atom_scoring_function(self, surface, networks_dict):
        # Measures fitness of sequences using an all-atom scoring function
        # within BUDE
        print('Measuring {} network fitness'.format(surface))

        # Initialises dictionary of fitness scores
        network_energies = OrderedDict()

        # Loads backbone model into ISAMBARD. NOTE must have been pre-processed
        # to remove ligands etc. so that only backbone coordinates remain.
        pdb = isambard.ampal.load_pdb(self.input_pdb)

        for num, G in networks_dict.items():
            # Packs network side chains onto the model with SCWRL4 and measures
            # the total model energy within BUDE
            new_pdb, energy = pack_side_chains(pdb, G, True)
            network_energies[num] = energy

        return network_energies

    def convert_energies_to_probabilities(self, network_energies):
        # Converts energy values output from BUDE into probabilities

        for network_num in list(network_energies.keys()):
            energy = network_energies[network_num]  # Energies output from BUDE
            # are in units of kJ/mol
            energy = (energy*1000) / (8.314*293)
            eqm_constant = np.exp(np.negative(energy))
            network_energies[network_num] = eqm_constant
        total = np.sum(list(network_energies.values()))

        network_fitness_scores = OrderedDict()
        for network_num in list(network_energies.keys()):
            eqm_constant = network_energies[network_num]
            network_fitness_scores[network_num] = eqm_constant / total

        return network_fitness_scores

    def create_mating_population_fittest_indv(self, surface, networks_dict,
                                              network_fitness_scores, pop_size,
                                              unfit_fraction):
        # Creates mating population from the fittest sequences plus a subset of
        # less fit sequences (so as to maintain diversity in the mating
        # population in order to prevent convergence on a non-global minimum)
        print('Creating mating population for {}'.format(surface))

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
            random_index = random.randint(0, (len(unfit_network_indices)-1))
            network_num = unfit_network_indices[random_index]
            mating_pop_dict[network_num] = copy.deepcopy(networks_dict[network_num])
            unfit_network_indices = [num for num in unfit_network_indices
                                     if num != network_num]
            count += 1

        return mating_pop_dict

    def create_mating_population_roulette_wheel(self, surface, networks_dict,
                                                network_fitness_scores,
                                                pop_size):
        # Creates mating population from individuals, with the likelihood of
        # selection of each sequence being weighted by its raw fitness score
        print('Creating mating population for {}'.format(surface))

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
            network_cumulative_probabilities = gen_cumulative_probabilities(
                network_fitness_array
            )

            random_number = random.uniform(0, 1)
            nearest_index = (np.abs(network_cumulative_probabilities-random_number)).argmin()

            if network_cumulative_probabilities[nearest_index] < random_number:
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

    def uniform_crossover(self, surface, mating_pop_dict):
        # Selects pairs of individuals at random from mating population and
        # performs uniform crossover
        print('Performing crossovers for {}'.format(surface))

        # Initialises dictionary of child networks
        crossover_pop_dict = OrderedDict()

        # Selects pairs of networks at random to crossover with each other
        network_num = list(mating_pop_dict.keys())
        random.shuffle(network_num)
        network_num = iter(network_num)  # Do not merge with line below,
        # and do not introduce any lines of code between them!
        network_num = list(zip(network_num, network_num))

        # Performs uniform crossover
        for index, network_pair in enumerate(network_num):
            network_num_1 = network_pair[0]
            network_num_2 = network_pair[1]
            mate_1 = copy.deepcopy(mating_pop_dict[network_num_1])
            mate_2 = copy.deepcopy(mating_pop_dict[network_num_2])

            for node in list(mate_1.nodes):
                random_number = random.uniform(0, 1)
                if random_number <= self.crossover_prob:
                    # Copy to prevent these dictionaries from updating when the
                    # node attributes are updated in the code below (otherwise
                    # both nodes will be assigned the same identity as the node
                    # in mate_1, instead of the node identities being crossed
                    # over)
                    mate_1_node_attributes = copy.deepcopy(mate_1.nodes[node])
                    mate_2_node_attributes = copy.deepcopy(mate_2.nodes[node])

                    # Ensures that if labels of nodes don't match between the
                    # two networks (although they should match), non-matched as
                    # well as matched labels are copied across
                    for attribute in list(mate_1.nodes[node].keys()):
                        del mate_1.nodes[node][attribute]
                    for attribute in list(mate_2.nodes[node].keys()):
                        del mate_2.nodes[node][attribute]

                    nx.set_node_attributes(mate_1, values={node: mate_2_node_attributes})
                    nx.set_node_attributes(mate_2, values={node: mate_1_node_attributes})

            crossover_pop_dict[network_num_1] = mate_1
            crossover_pop_dict[network_num_2] = mate_2

        return crossover_pop_dict

    def segmented_crossover(self, surface, mating_pop_dict):
        # Selects pairs of individuals at random from mating population and
        # performs segmented crossover
        print('Performing crossovers for {}'.format(surface))

        # Initialises dictionary of child networks
        crossover_pop_dict = OrderedDict()

        # Selects pairs of networks at random to crossover with each other
        network_num = list(mating_pop_dict.keys())
        random.shuffle(network_num)
        network_num = iter(network_num)  # Do not merge with line below,
        # and do not introduce any lines of code between them!
        network_num = list(zip(network_num, network_num))

        # Performs segmented crossover
        for index, network_pair in enumerate(network_num):
            network_num_1 = network_pair[0]
            network_num_2 = network_pair[1]
            mate_1 = copy.deepcopy(mating_pop_dict[network_num_1])
            mate_2 = copy.deepcopy(mating_pop_dict[network_num_2])

            swap = False
            for node in list(mate_1.nodes):
                random_number = random.uniform(0, 1)

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
                    mate_1_attributes = copy.deepcopy(mate_1.nodes[node])
                    mate_2_attributes = copy.deepcopy(mate_2.nodes[node])

                    # Ensures that if labels of nodes don't match between the
                    # two networks (although they should match), non-matched as
                    # well as matched labels are copied across
                    for attribute in list(mate_1.nodes[node].keys()):
                        del mate_1.nodes[node][attribute]
                    for attribute in list(mate_2.nodes[node].keys()):
                        del mate_2.nodes[node][attribute]

                    nx.set_node_attributes(mate_1, values={node: mate_2_attributes})
                    nx.set_node_attributes(mate_2, values={node: mate_1_attributes})

            crossover_pop_dict[network_num_1] = mate_1
            crossover_pop_dict[network_num_2] = mate_2

        return crossover_pop_dict

    def swap_mutate(self, surface, crossover_pop_dict):
        # Performs swap mutations (= mutates randomly selected individual
        # network nodes to a randomly selected (different) amino acid identity)
        print('Performing mutations for {}'.format(surface))

        # Initialises dictionary of mutated child networks
        mutated_pop_dict = OrderedDict()

        # Mutates the amino acid identities of randomly selected nodes
        for network_num in list(crossover_pop_dict.keys()):
            G = copy.deepcopy(crossover_pop_dict[network_num])

            for node in list(G.nodes):
                random_number = random.uniform(0, 1)
                if random_number <= self.mutation_prob:
                    orig_aa = G.nodes[node]['aa_id']
                    poss_aas = copy.deepcopy(self.aa_list)
                    poss_aas.remove(orig_aa)
                    new_aa = poss_aas[random.randint(0, (len(poss_aas)-1))]

                    nx.set_node_attributes(G, values={node: {'aa_id': new_aa}})

            mutated_pop_dict[network_num] = G

        return mutated_pop_dict

    def scramble_mutate(self, surface, crossover_pop_dict):
        # Performs scramble mutations (= scrambles the identities of a subset
        # of amino acids selected at random)
        print('Performing mutations for {}'.format(surface))

        # Initialises dictionary of mutated child networks
        mutated_pop_dict = OrderedDict()

        # Scrambles the amino acid identities of randomly selected nodes
        for network_num in list(crossover_pop_dict.keys()):
            G = copy.deepcopy(crossover_pop_dict[network_num])

            scrambled_nodes = []
            aa_ids = []
            for node in list(G.nodes):
                random_number = random.uniform(0, 1)
                if random_number <= self.mutation_prob:
                    scrambled_nodes.append(node)
                    aa_ids.append(G.nodes[node]['aa_id'])

            random.shuffle(aa_ids)
            attributes = OrderedDict({
                node: {'aa_id': aa_id} for node, aa_id in
                zip(scrambled_nodes, aa_ids)
            })
            nx.set_node_attributes(G, values=attributes)

            mutated_pop_dict[network_num] = G

        return mutated_pop_dict

    def add_children_to_parents(self, surface, mutated_pop_dict,
                                mating_pop_dict, index):
        # Combines parent and child generations
        print('Combining parent and child generations for {}'.format(surface))

        # Renumbers networks to prevent overlap
        if index == 0:
            count = 0
        elif index == 1:
            count = 2*self.propensity_pop_size  # The number of networks
            # already added to sequences_dict following propensity scoring
        merged_networks_dict = OrderedDict()

        for num, G in mutated_pop_dict.items():
            merged_networks_dict[count] = copy.deepcopy(G)
            count += 1
        for num, G in mating_pop_dict.items():
            merged_networks_dict[count] = copy.deepcopy(G)
            count += 1

        return merged_networks_dict


class run_ga_pipeline(initialise_ga_object):

    def __init__(self, params, sequences_dict):
        initialise_ga_object.__init__(self, params)
        self.sequences_dict = sequences_dict

    def run_genetic_algorithm(self, bayes_params):
        # Pipeline function to run genetic algorithm

        print('Running genetic algorithm')

        self.propensity_weight = bayes_params['propvsfreqweight']
        if self.method_select_mating_pop == 'fittest':
            self.unfit_fraction = bayes_params['unfitfraction']

        ga_calcs = run_ga_calcs(self.params)

        if self.method_select_mating_pop in ['fittest', 'roulettewheel']:
            raw_or_rank = 'raw'
        elif self.method_select_mating_pop == 'rankroulettewheel':
            raw_or_rank = 'rank'

        count = 0
        while count < self.num_gens:
            count += 1
            print('Generation {}'.format(count))

            for surface in list(self.sequences_dict.keys()):
                networks_dict_all = self.sequences_dict[surface]
                networks_list_all = []

                # Splits networks to optimise via different objectives
                # (propensity and side-chain packing) if
                # self.method_fitness_score == 'split'
                if self.method_fitness_score == 'split':
                    networks_dict_propensity = OrderedDict(
                        {key: networks_dict_all[key] for index, key in
                         enumerate(list(networks_dict_all.keys()))
                         if index < 2*self.propensity_pop_size}
                    )
                    networks_dict_all_atom = OrderedDict(
                        {key: networks_dict_all[key] for index, key in
                         enumerate(list(networks_dict_all.keys()))
                         if index >= 2*self.propensity_pop_size}
                    )
                    networks_list_all = [networks_dict_propensity,
                                         networks_dict_all_atom]
                    pop_sizes = [self.propensity_pop_size,
                                 (self.pop_size-self.propensity_pop_size)]

                else:
                    networks_list_all = [networks_dict_all]
                    pop_sizes = [self.pop_size]

                for index, networks_dict in enumerate(networks_list_all):
                    # Measures fitness of sequences in starting population
                    if (
                        (self.method_fitness_score == 'propensity')
                        or
                        (self.method_fitness_score == 'alternate' and count % 2 == 1)
                        or
                        (self.method_fitness_score == 'split' and index == 0)
                    ):
                        (network_propensity_scores, network_frequency_scores
                        ) = ga_calcs.measure_fitness_propensity(
                            surface, networks_dict
                        )
                        (network_fitness_scores
                        ) = ga_calcs.combine_propensity_and_frequency_scores(
                            network_propensity_scores,
                            network_frequency_scores, raw_or_rank
                        )
                    elif (
                        (self.method_fitness_score == 'allatom')
                        or
                        (self.method_fitness_score == 'alternate' and count % 2 == 0)
                        or
                        (self.method_fitness_score == 'split' and index == 1)
                    ):
                        (network_energies
                        ) = ga_calcs.measure_fitness_all_atom_scoring_function(
                            surface, networks_dict
                        )
                        (network_fitness_scores
                        ) = ga_calcs.convert_energies_to_probabilities(
                            network_energies
                        )

                    # Selects subpopulation for mating
                    if self.method_select_mating_pop == 'fittest':
                        mating_pop_dict = ga_calcs.create_mating_population_fittest_indv(
                            surface, networks_dict, network_fitness_scores,
                            pop_sizes[index], self.unfit_fraction
                        )
                    elif self.method_select_mating_pop in [
                        'roulettewheel', 'rankroulettewheel'
                    ]:
                        (mating_pop_dict
                        ) = ga_calcs.create_mating_population_roulette_wheel(
                            surface, networks_dict, network_fitness_scores,
                            pop_sizes[index]
                        )

                    # Performs crossover of parent sequences to generate child
                    # sequences
                    if self.method_crossover == 'uniform':
                        crossover_pop_dict = ga_calcs.uniform_crossover(
                            surface, mating_pop_dict
                        )
                    elif self.method_crossover == 'segmented':
                        crossover_pop_dict = ga_calcs.segmented_crossover(
                            surface, mating_pop_dict
                        )

                    # Mutates child sequences
                    if self.method_mutation == 'swap':
                        mutated_pop_dict = ga_calcs.swap_mutate(
                            surface, crossover_pop_dict
                        )
                    elif self.method_mutation == 'scramble':
                        mutated_pop_dict = ga_calcs.scramble_mutate(
                            surface, crossover_pop_dict
                        )

                    # Combines parent and child sequences into single
                    # generation
                    merged_networks_dict = ga_calcs.add_children_to_parents(
                        surface, mutated_pop_dict, mating_pop_dict, index
                    )

                    # Shuffles metworks dictionary so that in the case of a
                    # split optimisation a mixture of parent and child networks
                    # are combined into the sub-classes whose fitnesses are
                    # measured by different methods in the following round of
                    # optimisation
                    if self.method_fitness_score == 'split' and index == 1:
                        merged_networks_dict = OrderedDict(
                            {**self.sequences_dict[surface], **merged_networks_dict}
                        )

                    merged_networks_num = list(merged_networks_dict.keys())
                    merged_networks = list(merged_networks_dict.values())
                    random.shuffle(merged_networks)
                    shuffled_merged_networks_dict = OrderedDict(
                        {merged_networks_num[i]: merged_networks[i]
                         for i in range(len(merged_networks_num))}
                    )
                    self.sequences_dict[surface] = shuffled_merged_networks_dict

        # Calculates fitness of output sequences and filters population based
        # upon fitness
        for surface in list(self.sequences_dict.keys()):
            networks_dict = self.sequences_dict[surface]

            if self.method_fitness_score != 'allatom':
                (network_propensity_scores, network_frequency_scores
                ) = ga_calcs.measure_fitness_propensity(
                    surface, networks_dict
                )
                (network_fitness_scores
                ) = ga_calcs.combine_propensity_and_frequency_scores(
                    network_propensity_scores, network_frequency_scores,
                    raw_or_rank
                )
            elif self.method_fitness_score == 'allatom':
                (network_energies
                ) = ga_calcs.measure_fitness_all_atom_scoring_function(
                    surface, networks_dict
                )
                (network_fitness_scores
                ) = ga_calcs.convert_energies_to_probabilities(network_energies)

            mating_pop_dict = ga_calcs.create_mating_population_fittest_indv(
                surface, networks_dict, network_fitness_scores,
                self.pop_size, unfit_fraction=0
            )
            self.sequences_dict[surface] = mating_pop_dict

        return self.sequences_dict
