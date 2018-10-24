
# Do I need to worry about genetic algorithm overfitting?
import budeff
import copy
import isambard
import isambard.modelling as modelling
import math
import random
import networkx as nx
import numpy as np
from collections import OrderedDict
from operator import itemgetter

if __name__ == 'subroutines.run_genetic_algorithm':
    from subroutines.find_parameters import initialise_class
    from subroutines.generate_initial_sequences import (
        interpolate_propensities, propensity_to_probability_distribution
    )
    from subroutines.variables import gen_amino_acids_dict
else:
    from betadesigner.subroutines.find_parameters import initialise_class
    from betadesigner.subroutines.generate_initial_sequences import (
        interpolate_propensities, propensity_to_probability_distribution
    )
    from betadesigner.subroutines.variables import gen_amino_acids_dict


def pack_side_chains(ampal_object, G, rigid_rotamers):
    # Uses SCWRL4 to pack network side chains onto a backbone structure and
    # measures the total energy of the model within BUDE

    aa_dict = gen_amino_acids_dict()

    # Makes FASTA sequence to feed into SCWRL4
    fasta_seq = ''
    for res in ampal_object.get_monomers():
        res_id = '{}{}{}'.format(res.parent.id, res.id, res.insertion_code)
        if res_id in list(G.nodes):
            fasta_seq += aa_dict[G.nodes[res_id]['aa_id']]
        else:
            fasta_seq += res.mol_letter

    # Packs side chains with SCWRL4. NOTE that fasta sequence must be provided
    # as a list
    new_ampal_object = modelling.pack_side_chains_scwrl(
        ampal_object, [fasta_seq], rigid_rotamer_model=rigid_rotamers,
        hydrogens=False
    )

    # Calculate total energy
    energy = budeff.get_internal_energy(new_ampal_object).total_energy

    return new_ampal_object, energy


class run_ga_calcs(initialise_class):

    def __init__(self, parameters):
        initialise_class.__init__(self, parameters)

    def measure_fitness_propensity(self, surface, networks_dict):
        # Measures fitness of amino acid sequences from their propensities for
        # the structural features of the input backbone structure.
        print('Measuring {} network fitness'.format(surface))

        # Initialises dictionary of fitness scores
        network_fitness_scores = OrderedDict()

        # Extracts propensity scales for the surface (both individual and
        # pairwise)
        sub_indv_propensity_dicts = OrderedDict({
            dict_label: propensity_dict for dict_label, propensity_dict in
            self.propensity_dicts.items() if
            ((dict_label.split('_')[0] == surface[0:3])
             and (dict_label.split('_')[-1] == 'indv'))
        })

        for num in list(networks_dict.keys()):
            G = networks_dict[num]
            # Total propensity count (across all nodes in network)
            propensity_count = 0
            # Per node propensity count (not currently in use for further
            # calculations)
            node_propensities_dict = OrderedDict(
                {node: {'propensity_sum': 0} for node in list(G.nodes)}
            )

            for node_1 in list(G.nodes):
                # Calculates interpolated propensity of each node for all
                # individual structural features considered
                aa_1 = G.nodes[node_1]['aa_id']

                for dict_label, propensity_dict in sub_indv_propensity_dicts.items():
                    node_prop = G.nodes[node_1][dict_label.split('_')[1]]
                    prop_weight = self.propensity_dict_weights[dict_label]
                    aa_propensity_scale = propensity_dict[aa_1]

                    propensity = interpolate_propensities(
                        node_prop, aa_propensity_scale, dict_label
                    )
                    propensity = prop_weight*np.negative(np.log(propensity))
                    propensity_count += propensity
                    node_propensities_dict[node_1]['propensity_sum'] += propensity

                # Loops through each node pair to sum pairwise propensity values
                for node_pair in G.edges(node_1):
                    if node_pair[0] == node_1:
                        node_2 = node_pair[1]
                    elif node_pair[1] == node_1:
                        node_2 == node_pair[0]
                    aa_2 = G.nodes[node_2]['aa_id']

                    # Loops through each interaction between a pair of nodes
                    for edge in G[node_1][node_2]:
                        edge_label = G[node_1][node_2][edge]['interaction']

                        # Loops through each property of node_1
                        for prop, node_prop in G.node[node_1].items():
                            if not prop in ['aa_id', 'int_ext', 'propensity_sum']:
                                dict_label = '{}_{}_{}_pairwise'.format(
                                    surface, prop, edge_label
                                )
                                try:
                                    propensity_dict = self.propensity_dicts[dict_label]
                                    prop_weight = self.propensity_dict_weights[dict_label]
                                    propensity_scale = propensity_dict['{}_{}'.format(aa_1, aa_2)]

                                    propensity = interpolate_propensities(
                                        node_prop, propensity_scale, dict_label
                                    )
                                    propensity = prop_weight*np.negative(np.log(propensity))
                                    propensity_count += propensity
                                    node_propensities_dict[node_1] += propensity
                                except KeyError:
                                    pass

            network_fitness_scores[num] = propensity_count

            nx.set_node_attributes(G, node_propensities_dict)
            networks_dict[num] = G

        return networks_dict, network_fitness_scores

    def measure_fitness_all_atom_scoring_function(self, surface, networks_dict):
        # Measures fitness of sequences using an all-atom scoring function
        # within BUDE
        print('Measuring {} network fitness'.format(surface))

        # Initialises dictionary of fitness scores
        network_fitness_scores = OrderedDict()

        # Loads backbone model into ISAMBARD. NOTE must have been pre-processed
        # to remove ligands etc. so that only backbone coordinates remain.
        pdb = isambard.ampal.load_pdb(self.input_pdb)


        for num, G in networks_dict.items():
            # Packs network side chains onto the model with SCWRL4 and measures
            # the total model energy within BUDE
            new_pdb, energy = pack_side_chains_swrl(
                pdb, sequence, G, True
            )
            network_fitness_scores[num] = energy

        return network_fitness_scores

    def create_mating_population_fittest_indv(self, surface, networks_dict,
                                              network_fitness_scores,
                                              unfit_fraction):
        # Creates mating population from the fittest sequences plus a subset of
        # less fit sequences (so as to maintain diversty in the mating
        # population in order to prevent convergence on a non-global minimum)
        print('Creating mating population for {}'.format(surface))

        # Determines numbers of fittest and random sequences to be added in
        if 0 <= unfit_fraction <= 1:
            unfit_pop_size = round((self.pop_size*unfit_fraction), 0)
            pop_size = self.pop_size - unfit_pop_size
        elif unfit_fraction > 1:
            unfit_pop_size = unfit_fraction
            pop_size = self.pop_size - unfit_pop_size

        # Initialises dictionary of fittest networks
        mating_pop_dict = OrderedDict()

        # Sorts networks by their fitness values, from most (-ve) to least
        # (+ve) fit
        network_fitness_scores = OrderedDict(sorted(
            network_fitness_scores.items(), key=itemgetter(1), reverse=False)
        )

        # Adds fittest individuals to mating population
        for index, num in enumerate(list(network_fitness_scores.keys())):
            if index < pop_size:
                mating_pop_dict[num] = networks_dict[num]
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
            mating_pop_dict[network_num] = networks_dict[network_num]
            unfit_network_indices = [num for num in unfit_network_indices
                                     if num != network_num]
            count += 1

        return mating_pop_dict

    def create_mating_population_roulette_wheel(self, surface, networks_dict,
                                                network_fitness_scores,
                                                raw_or_rank):
        # Creates mating population from individuals, with the likelihood of
        # selection of each sequence being weighted by its raw fitness score
        print('Creating mating population for {}'.format(surface))

        # Initialises dictionary of fittest networks
        mating_pop_dict = OrderedDict()

        # Sorts networks by their fitness values, from least (+ve) to most
        # (-ve) fit
        sorted_network_num = np.argsort(np.array(list(network_fitness_scores.keys())))[::-1]
        sorted_network_fitness_scores = np.sort(np.array(list(network_fitness_scores.values())))[::-1]

        network_cumulative_probabilities = propensity_to_probability_distribution(
            sorted_network_fitness_scores, raw_or_rank
        )

        # Adds individuals (their likelihood of selection weighted by their raw
        # fitness scores) to mating population
        count = 0
        while count < self.pop_size:
            random_number = random.uniform(0, 1)
            nearest_index = (np.abs(network_cumulative_probabilities)).argmin()

            if network_cumulative_probabilities[nearest_index] >= random_number:
                selected_network = sorted_network_num[nearest_index]
            else:
                selected_network = sorted_network_num[nearest_index+1]

            if not selected_network in list(mates.keys()):
                # Ensures a network is only added to the mating population once
                # NOTE that the networks dictionaries are not updated after a
                # network is selected, in order to avoid the need to update
                # the corresponding probability values
                mating_pop_dict[selected_network] = networks_dict[selected_network]
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
            mate_1 = mating_pop_dict[network_num_1]
            mate_2 = mating_pop_dict[network_num_2]

            for node in list(mate_1.nodes):
                random_number = random.uniform(0, 1)
                if random_number <= self.crossover_prob:
                    mate_1_attributes = mate_1.nodes[node]
                    mate_2_attributes = mate_2.nodes[node]

                    nx.set_node_attributes(mate_1, values={})  # Ensures that
                    # if labels of nodes don't match between the two networks
                    # (although they should match), non-matched as well as
                    # matched labels are copied across
                    nx.set_node_attributes(mate_1, values={node: mate_2_attributes})
                    nx.set_node_attributes(mate_2, values={})
                    nx.set_node_attributes(mate_2, values={node: mate_1_attributes})

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
            mate_1 = mating_pop_dict[network_num_1]
            mate_2 = mating_pop_dict[network_num_2]

            count = 0
            swap = False
            for node in list(mate_1.nodes):
                count += 1
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
                    mate_1_attributes = mate_1.nodes[node]
                    mate_2_attributes = mate_2.nodes[node]

                    nx.set_node_attributes(mate_1, values={})
                    nx.set_node_attributes(mate_1, values={node: mate_2_attributes})
                    nx.set_node_attributes(mate_2, values={})
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
            G = crossover_pop_dict[network_num]

            for node in list(G.nodes):
                random_number = random.uniform(0, 1)
                if random_number <= self.mutation_prob:
                    orig_aa = G.nodes[node]['aa_id']
                    poss_aas = copy.deepcopy(self.aas)
                    poss_aas.remove(orig_aa)
                    new_aa = poss_aas[random.randint(0, (len(poss_aas)-1))]

                    G.nodes[node]['aa_id'] = new_aa

            mutated_pop_dict[network_num] = G

        return mutated_pop_dict

    def scramble_mutate(self, surface, crossover_pop_dict):
        # Performs scramble mutations (= scrambles the identities of a subset
        # of amino acids selected at random)
        print('Performing mutations for {}'.format(surface))

        # Initialises dictionary of mutated child networks
        mutated_pop_dict = OrderedDict()

        # Scrambles the amino acid identities of randomly selected nodes
        for network_num in list(mutated_pop_dict.keys()):
            G = mutated_pop_dict[network_num]

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

    def add_children_to_parents(self, surface, mating_pop_dict,
                                mutated_pop_dict, index, split_number):
        # Combines parent and child generations
        print('Combining parent and child generations for {}'.format(surface))

        for node in list(mutated_pop_dict[0].nodes):
            print(mutated_pop_dict[0].nodes[node]['z'],
                  mutated_pop_dict[0].nodes[node]['aa_id'])

        # Renumbers networks to prevent overlap
        if index == 0:
            count = 0
        elif index == 1:
            count = split_number
        merged_networks_dict = OrderedDict()

        for num, G in mating_pop_dict.items():
            merged_networks_dict[count] = G
            count += 1
        for num, G in mutated_pop_dict.items():
            merged_networks_dict[count] = G
            count += 1

        return merged_networks_dict


class run_ga_pipeline(initialise_class):

    def __init__(self, parameters):
        initialise_class.__init__(self, parameters)

    def run_genetic_algorithm(self, sequences_dict):
        # Pipeline function to run genetic algorithm

        print('Running genetic algorithm')
        ga_calcs = run_ga_calcs(self.parameters)

        count = 0
        while count < self.num_gens:
            count += 1
            print('Generation {}'.format(count))

            for surface in list(sequences_dict.keys()):
                networks_dict_all = sequences_dict[surface]
                networks_list_all = []

                # Splits networks to optimise via different objectives
                # (propensity and side-chain packing) if
                # self.method_fitness_score == 'split'
                if self.method_fitness_score == 'split':
                    split_number = math.ceil(
                        self.split_fraction*len(list(networks_dict_all.keys()))
                    )
                    networks_dict_propensity = OrderedDict(
                        {key: networks_dict_all[key] for index, key in
                         enumerate(list(networks_dict_all.keys()))
                         if index < split_number}
                    )
                    networks_dict_all_atom = OrderedDict(
                        {key: networks_dict_all[key] for index, key in
                         enumerate(list(networks_dict_all.keys()))
                         if index >= split_number}
                    )
                    networks_list_all = [networks_dict_propensity,
                                         networks_dict_all_atom]
                else:
                    split_number = ''
                    networks_list_all = [networks_dict_all]

                for index, networks_dict in enumerate(networks_list_all):
                    # Measures fitness of sequences in starting population
                    if (
                        (self.method_fitness_score == 'propensity')
                        or
                        (self.method_fitness_score == 'alternate' and count % 2 == 1)
                        or
                        (self.method_fitness_score == 'split' and index == 0)
                    ):
                        (networks_dict, network_fitness_scores
                        ) = ga_calcs.measure_fitness_propensity(
                            surface, networks_dict
                        )
                    elif (
                        (self.method_fitness_score == 'allatom')
                        or
                        (self.method_fitness_score == 'alternate' and count % 2 == 0)
                        or
                        (self.method_fitness_score == 'split' and index == 1)
                    ):
                        (networks_dict, network_fitness_scores
                        ) = ga_calcs.measure_fitness_all_atom_scoring_function(
                            surface, networks_dict
                        )

                    # Selects subpopulation for mating
                    if self.method_select_mating_pop == 'fittest':
                        mating_pop_dict = ga_calcs.create_mating_population_fittest_indv(
                            surface, networks_dict, network_fitness_scores,
                            self.unfit_fraction
                        )
                    elif self.method_select_mating_pop == 'roulettewheel':
                        mating_pop_dict = ga_calcs.create_mating_population_roulette_wheel(
                            surface, networks_dict, network_fitness_scores, 'raw'
                        )
                    elif self.method_select_mating_pop == 'rankroulettewheel':
                        mating_pop_dict = ga_calcs.create_mating_population_roulette_wheel(
                            surface, networks_dict, network_fitness_scores, 'rank'
                        )

                    # Performs crossover of parent sequences to generate child
                    # sequences
                    if self.method_crossover == 'uniform':
                        crossover_output_dict = ga_calcs.uniform_crossover(
                            surface, mating_pop_dict
                        )
                    elif self.method_crossover == 'segmented':
                        crossover_output_dict = ga_calcs.egmented_crossover(
                            surface, mating_pop_dict
                        )

                    # Mutates child sequences
                    if self.method_mutation == 'swap':
                        mutation_output_dict = ga_calcs.swap_mutate(
                            surface, crossover_output_dict
                        )
                    elif self.method_mutation == 'scramble':
                        mutation_output_dict = ga_calcs.scramble_mutate(
                            surface, crossover_output_dict
                        )

                    # Combines parent and child sequences into single
                    # generation
                    merged_networks_dict = ga_calcs.add_children_to_parents(
                        surface, mating_pop_dict, mutation_output_dict, index,
                        split_number
                    )

                    if self.method_fitness_score != 'split':
                        sequences_dict[surface] = merged_networks_dict
                    elif self.method_fitness_score == 'split':
                        if index == 0:
                            sequences_dict[surface] = merged_networks_dict
                        elif index == 1:
                            sequences_dict[surface] = {**sequences_dict[surface],
                                                       **merged_networks_dict}

        # Calculates fitness of output sequences and filters population based
        # upon fitness
        for surface in list(sequences_dict.keys()):
            networks_dict = sequences_dict[surface]

            if self.method_fitness_score != 'allatom':
                networks_dict, network_fitness_scores = ga_calcs.measure_fitness_propensity(
                    surface, networks_dict
                )
            elif self.method_fitness_score == 'allatom':
                network_fitness_scores = ga_calcs.measure_fitness_all_atom_scoring_function(
                    surface, networks_dict
                )

            mating_pop_dict = ga_calcs.create_mating_population_fittest_indv(
                surface, networks_dict, network_fitness_scores, unfit_fraction=0
            )
            sequences_dict[surface] = mating_pop_dict

        return sequences_dict
