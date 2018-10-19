
# Do I need to worry about genetic algorithm overfitting?
import budeff
import isambard
import random
import networkx as nx
import numpy as np
from collections import OrderedDict

class run_ga():

    def __init__(self, propensity_dicts):
        self.propensity_dicts = propensity_dicts
        self.aas = list(self.propensity_dicts['int_z_indv'].keys())

    def measure_fitness_propensity(sequences_dict, propensity_dict_weights):
        # Measures fitness of amino acid sequences from their propensities for
        # the structural features of the input backbone structure
        fitness_scores = OrderedDict()

        for surface_label, networks_dict in sequences_dict.items():
            sub_indv_propensity_dicts = OrderedDict({
                dict_label: propensity_dict for dict_label, propensity_dict in
                self.propensity_dicts.items() if
                ((dict_label.split('_')[0] == surface_label[0:3])
                 and (dict_label.split('_')[2] == 'indv'))
            })

            network_fitnesses = OrderedDict()
            for num, G in networks_dict.items():
                propensity_count = 0
                node_propensities_dict = OrderedDict(node: 0 for node in list(G.nodes))

                for node_1 in list(G.nodes):
                    aa_1 = G.nodes[node_1]['aa_id']

                    for dict_label, propensity_dict in sub_indv_propensity_dicts.items():
                        node_prop = G.nodes[node_1][dict_label.split('_')[1]]
                        prop_weight = propensity_dict_weights[dict_label]
                        aa_propensity_scale = propensity_dict[aa_1]

                        index_1 = (np.abs(aa_propensity_scale[0]-node_prop)).argmin()
                        prop_val_1 = aa_propensity_scale[0][index_1]
                        propensity_1 = aa_propensity_scale[1][index_1]

                        index_2 = ''
                        if prop_val_1 < node_prop:
                            index_2 = index_1 + 1
                        elif prop_val_1 > node_prop:
                            index_2 = index_1 - 1

                        if index_2 == '':
                            propensity == aa_propensity_scale[1][index_1]
                        else:
                            prop_val_2 = aa_propensity_scale[0][index_2]
                            propensity_2 = aa_propensity_scale[1][index_2]

                            weight_1 = abs(prop_val_2 - node_prop)
                            weight_2 = abs(prop_val_1 - node_prop)
                            propensity = (((propensity_1*weight_1) + (propensity_2*weight_2))
                                          / abs(prop_val_2 - prop_val_1))

                        propensity = prop_weight*np.negative(np.log(propensity))
                        propensity_count += propensity
                        node_propensities_dict[node_1] += propensity

                    # Loops through each node pair
                    for node_pair in G.edges(node_1):
                        node_pair = ''
                        if node_pair[0] == node_1:
                            node_2 = node_pair[1]
                        elif node_pair[1] == node_1:
                            node_2 == node_pair[0]
                        aa_2 = G.nodes[node_2]['aa_id']

                        # Loops through each interaction between a pair of nodes
                        for edge in G[node_1][node_2]:
                            edge_label = G[node_1][node_2][edge]['label']

                            # Loops through each property of node_1
                            for prop, node_prop in G.node[node_1].items():
                                propensity_dict = self.propensity_dicts[
                                    '{}_{}_{}_pairwise'.format(surface_label, prop, edge_label)
                                ]
                                prop_weight = propensity_dict_weights[dict_label]
                                propensity_scale = propensity_dict['{}_{}'.format(aa_1, aa_2)]

                                index_1 = (np.abs(propensity_scale[0]-node_prop)).argmin()
                                prop_val_1 = propensity_scale[0][index_1]
                                propensity_1 = propensity_scale[1][index_1]

                                index_2 = ''
                                if prop_val_1 < node_prop:
                                    index_2 = index_1 + 1
                                elif prop_val_1 > node_prop:
                                    index_2 = index_1 - 1

                                if index_2 == '':
                                    propensity == propensity_scale[1][index_1]
                                else:
                                    prop_val_2 = propensity_scale[0][index_2]
                                    propensity_2 = propensity_scale[1][index_2]

                                    weight_1 = abs(prop_val_2 - node_prop)
                                    weight_2 = abs(prop_val_1 - node_prop)
                                    propensity = (((propensity_1*weight_1) + (propensity_2*weight_2))
                                                  / abs(prop_val_2 - prop_val_1))

                                propensity = prop_weight*np.negative(np.log(propensity))
                                propensity_count += propensity
                                node_propensities_dict[node_1] += propensity

                network_fitnesses[num] = propensity_count

            fitness_scores[surface_label] = network_fitnesses

        return fitness_scores

    def measure_fitness_all_atom_scoring_function(
        sequences_dict, pdb_file_lines, orig_pdb_seq, aa_to_fasta
    ):
        # Measures fitness of sequences using an all-atom scoring function
        # within BUDE
        fitness_scores = OrderedDict()

        # Load backbone structure into ampal
        pdb = isambard.ampal.load_pdb(pdb_file_lines)

        for surface_label, networks_dict in sequences_dict.items():
            network_fitnesses = OrderedDict()

            for num, G in networks_dict.items():
                # Add loop amino acids into amino acid sequence
                sequence = ''
                for res_id, res_name in orig_pdb_seq.items():
                    if res_id in list(G.nodes):
                        sequence += aa_to_fasta[G.nodes[node]['aa_id']]
                    else:
                        sequence += aa_to_fasta[res_name]

                # Packs side chains with SCWRL4
                pdb = pack_side_chains_swrl(
                    pdb, sequence, rigid_rotamer_model=True, hydrogens=False
                )

                # Calculate all-atom scoring function
                energy = budeff.get_internal_energy(pdb).total_energy

                network_fitnesses[num] = energy

            fitness_scores[surface_label] = network_fitnesses

        return fitness_scores

    def create_mating_population_fittest_indv(
        sequences_dict, fitness_scores, pop_size, unfit_fraction=0
    ):
        # Creates mating population from fittest individuals
        mating_pop_dict = OrderedDict()

        if 0 < unfit_fraction < 1:
            unfit_pop_size = round((pop_size*unfit_fraction), 0)
            pop_size = pop_size - unfit_pop_size
        elif unfit_fraction >= 1:
            unfit_pop_size = unfit_fraction
            pop_size = pop_size - unfit_pop_size

        for surface_label, networks_dict in sequences_dict.items():
            network_fitnesses = fitness_scores[surface_label]

            # Sorts networks by their fitness values, from most to least fit
            network_fitnesses = OrderedDict(sorted(
                network_fitnesses.items(), key=itemgetter(1), reverse=False)
            )

            mates = OrderedDict()
            for index, num in enumerate(list(network_fitnesses.keys())):
                while index < pop_size:
                    mates[num] = sequences_dict[num]
                    network_fitnesses[num] = ''

            network_fitnesses = OrderedDict({
                num: fitness for num, fitness in network_fitnesses.items()
                if fitness != ''
            })

            count = 0
            while count < unfit_pop_size:
                random_index = random.randint(0, (len(list(network_fitnesses.keys()))-1))
                network_num = list(network_fitnesses.keys())[random_index]
                mates[network_num] = sequences_dict[network_num]

                network_fitnesses[network_num] = ''
                network_fitnesses = OrderedDict({
                    num: fitness for num, fitness in network_fitnesses.items()
                    if fitness != ''
                })
                count += 1

            mating_pop_dict[surface_label] = mates

        return mating_pop_dict

    def create_mating_population_roulette_wheel(sequences_dict, fitness_scores, pop_size):
        # Creates mating population from individuals weighted by their fitness
        mating_pop_dict = OrderedDict()

        for surface_label, networks_dict in sequences_dict.items():
            network_fitnesses = fitness_scores[surface_label]
            # Sorts networks from least to most fit
            network_fitnesses = OrderedDict(sorted(
                network_fitnesses.items(), key=itemgetter(1), reverse=True)
            )

            propensity_diff_sum = 1
            for index, propensity in enumerate(list(network_fitnesses.values())):
                propensity = network_fitnesses[network_label]
                if index == 0:
                    ref_propensity = propensity
                elif index > 0:
                    propensity_diff = abs(ref_propensity - propensity)
                    propensity_diff_sum += (propensity_diff + 1)

            network_cumulative_probabilities_dict = OrderedDict()
            cumulative_probability = 0
            for index, propensity in enumerate(list(network_fitnesses.values())):
                network_label = list(network_fitnesses.keys())[index]
                if index == 0:
                    ref_propensity = propensity
                    probability = 1 / propensity_diff
                elif index > 1:
                    probability = (abs(ref_propensity-propensity)+1) / propensity_diff_sum
                cumulative_probability += probability
                network_cumulative_probabilities_dict[network_label] = cumulative_probability

            cumulative_probabilities_array = np.array(list(network_cumulative_probabilities_dict.values()))
            if round(cumulative_probabilities_array[-1], 4) != 1.0:
                sys.exit('ERROR {}: Cumulative probability = {}'.format(
                    surface_label, cumulative_probabilities_array[-1])
                )

            mates = OrderedDict()
            count = 0
            while count < pop_size:
                # Selects amino acid weighted by its probability
                random_number = random.uniform(0, 1)
                nearest_index = (np.abs(cumulative_probabilities_array)).argmin()

                if cumulative_probabilities_array[nearest_index] >= random_number:
                    selected_network_num = list(network_cumulative_probabilities_dict.keys())[nearest_index]
                else:
                    selected_network_num = list(network_cumulative_probabilities_dict.keys())[nearest_index+1]

                if not selected_network_num in list(mates.keys()):
                    mates[selected_network_num] = networks_dict[selected_network_num]
                    count += 1

            mating_pop_dict[surface_label] = mates

        return mating_pop_dict

    def create_mating_population_rank_roulette_wheel(sequences_dict, fitness_scores, pop_size):
        # Creates mating population from individuals weighted by their fitness rank

        # ADD IN TEST TO ENSURE THAT POP_SIZE IS AN EVEN NUMBER
        mating_pop_dict = OrderedDict()

        for surface_label, networks_dict in sequences_dict.items():
            network_fitnesses = fitness_scores[surface_label]
            # Sorts networks from least to most fit
            network_fitnesses = OrderedDict(sorted(
                network_fitnesses.items(), key=itemgetter(1), reverse=True)
            )
            # Replaces network fitnesses by their ranks
            count = 0
            for network_num in list(network_fitnesses.keys()):
                count += 1
                network_fitnesses[network_num] = count

            propensity_diff_sum = 1
            for index, propensity in enumerate(list(network_fitnesses.values())):
                propensity = network_fitnesses[network_label]
                if index == 0:
                    ref_propensity = propensity
                elif index > 0:
                    propensity_diff = abs(ref_propensity - propensity)
                    propensity_diff_sum += (propensity_diff + 1)

            network_cumulative_probabilities_dict = OrderedDict()
            cumulative_probability = 0
            for index, propensity in enumerate(list(network_fitnesses.values())):
                network_label = list(network_fitnesses.keys())[index]
                if index == 0:
                    ref_propensity = propensity
                    probability = 1 / propensity_diff
                elif index > 1:
                    probability = (abs(ref_propensity-propensity)+1) / propensity_diff_sum
                cumulative_probability += probability
                network_cumulative_probabilities_dict[network_label] = cumulative_probability

            cumulative_probabilities_array = np.array(list(network_cumulative_probabilities_dict.values()))
            if round(cumulative_probabilities_array[-1], 4) != 1.0:
                sys.exit('ERROR {}: Cumulative probability = {}'.format(
                    surface_label, cumulative_probabilities_array[-1])
                )

            mates = OrderedDict()
            count = 0
            while count < pop_size:
                # Selects amino acid weighted by its probability
                random_number = random.uniform(0, 1)
                nearest_index = (np.abs(cumulative_probabilities_array)).argmin()

                if cumulative_probabilities_array[nearest_index] >= random_number:
                    selected_network_num = list(network_cumulative_probabilities_dict.keys())[nearest_index]
                else:
                    selected_network_num = list(network_cumulative_probabilities_dict.keys())[nearest_index+1]

                if not selected_network_num in list(mates.keys()):
                    mates[selected_network_num] = networks_dict[selected_network_num]
                    count += 1

            mating_pop_dict[surface_label] = mates

        return mating_pop_dict

    def uniform_crossover(mating_pop_dict, crossover_prob):
        # Selects pairs of individuals at random from mating population,
        # generates uniform crossover
        crossover_pop_dict = OrderedDict()
        for surface_label, networks_dict in mating_pop_dict.items():
            crossover_networks = OrderedDict()

            network_num = list(networks_dict.keys())
            random.shuffle(network_num)
            network_num = iter(network_num)  # Do not merge with line below,
            # and do not introduce any lines of code between them!
            network_num = list(zip(network_num, network_num))

            for index, network_pair in enumerate(network_num):
                network_num_1 = network_pair[0]
                network_num_2 = network_pair[1]
                mate_1 = networks_dict[network_num_1]
                mate_2 = networks_dict[network_num_2]

                for node in list(mate_1.nodes):
                    random_number = random.uniform(0, 1)
                    if random_number <= crossover_prob:
                        mate_1_attributes = mate_1.nodes[node]
                        mate_2_attributes = mate_2.nodes[node]

                        nx.set_node_attributes(mate_1, values={node: mate_2_attributes})
                        nx.set_node_attributes(mate_2, values={node: mate_1_attributes})

                crossover_networks[network_num_1] = mate_1
                crossover_networks[network_num_2] = mate_2

            crossover_pop_dict[surface_label] = crossover_networks

        return crossover_pop_dict

    def segmented_crossover(mating_pop_dict, swap_start_prob, swap_stop_prob):
        # Selects pairs of individuals at random from mating population,
        # generates segmented crossover
        crossover_pop_dict = OrderedDict()
        for surface_label, networks_dict in mating_pop_dict.items():
            crossover_networks = OrderedDict()

            network_num = list(networks_dict.keys())
            random.shuffle(network_num)
            network_num = iter(network_num)  # Do not merge with line below,
            # and do not introduce any lines of code between them!
            network_num = list(zip(network_num, network_num))

            for index, network_pair in enumerate(network_num):
                network_num_1 = network_pair[0]
                network_num_2 = network_pair[1]
                mate_1 = networks_dict[network_num_1]
                mate_2 = networks_dict[network_num_2]

                mutate_current = False
                for node in list(mate_1.nodes):
                    count += 1
                    random_number = random.uniform(0, 1)

                    if mutate_current is False:
                        if random_number <= swap_start_prob:
                            mutate_next = True
                        else:
                            mutate_next = False
                    elif mutate_current is True:
                        if random_number <= swap_stop_prob:
                            mutate_next = False
                        else:
                            mutate_next = True

                    mutate_current = mutate_next
                    if mutate_current is True:
                        mate_1_attributes = mate_1.nodes[node]
                        mate_2_attributes = mate_2.nodes[node]

                        nx.set_node_attributes(mate_1, values={node: mate_2_attributes})
                        nx.set_node_attributes(mate_2, values={node: mate_1_attributes})

                crossover_networks[network_num_1] = mate_1
                crossover_networks[network_num_2] = mate_2

            crossover_pop_dict[surface_label] = crossover_networks

        return crossover_pop_dict

    def swap_mutate(crossover_pop_dict, mutation_prob):
        # Mutate individual network nodes with a fixed probability
        mutated_pop_dict = OrderedDict()

        for surface_label, networks_dict in crossover_pop_dict.items():
            mutated_networks = OrderedDict()

            for network_num in list(networks_dict.keys()):
                network = networks_dict[network_num]

                for node in list(network.nodes):
                    random_number = random.uniform(0, 1)
                    if random_number <= mutation_prob:
                        orig_aa = network.nodes[node]['aa_id']
                        poss_aas = self.aas.remove(orig_aa)
                        new_aa = poss_aas[random.randint(0, (len(poss_aas)-1))]

                        network.nodes[node]['aa_id'] = new_aa

                mutated_networks[network_num] = network

            mutated_pop_dict[surface_label] = mutated_networks

        return mutated_pop_dict

    def scramble_mutate(crossover_pop_dict, mutation_prob):
        # Scramble the identities of a subset of amino acids selected at random
        mutated_pop_dict = OrderedDict()

        for surface_label, networks_dict in crossover_pop_dict.items():
            mutated_networks = OrderedDict()

            for network_num in list(networks_dict.keys()):
                network = networks_dict[network_num]

                scrambled_nodes = []
                aa_ids = []
                for node in list(network.nodes):
                    random_number = random.uniform(0, 1)
                    if random_number <= mutation_prob:
                        scrambled_nodes.append(node)
                        aa_ids.aapend(network.nodes[node]['aa_id'])

                random.shuffle(aa_ids)
                attributes = OrderedDict({
                    node: {'aa_id': aa_id} for node, aa_id in
                    zip(scrambled_nodes, aa_ids)
                })
                nx.set_node_attributes(network, values=attributes)

                mutated_networks[network_num] = network

            mutated_pop_dict[surface_label] = mutated_networks

        return mutated_pop_dict

    def add_children_to_parents(sequences_dict, mutated_pop_dict):
        # Combines parent and child generations
        merged_gen_dict = OrderedDict()

        for surface_label in list(sequences_dict.keys()):
            parent_networks = sequences_dict[surface_label]
            child_networks = mutated_pop_dict[surface_label]

            merged_networks = OrderedDict({**parent_networks, **child_networks})
            merged_gen_dict[surface_label] = merged_networks

        return merged_gen_dict

    def run_genetic_algorithm(num_generations, sequences_dict):
        # Pipeline function to run genetic algorithm
        count = 0
        while count < num_generations:
            count += 1
            fitness_scores = measure_fitness_propensity(sequences_dict, propensity_dict_weights)
            # fitness_scores = measure_fitness_all_atom_scoring_function(sequences_dict)

            mating_pop_dict = create_mating_population_fittest_indv(
                sequences_dict, fitness_scores, pop_size, unfit_fraction
            )
            # mating_pop_dict = create_mating_population_roulette_wheel(
            #     sequences_dict, fitness_scores, pop_size
            # )
            # mating_pop_dict = create_mating_population_rank_roulette_wheel(
            #     sequences_dict, fitness_scores, pop_size
            # )

            crossover_output_dict = uniform_crossover(
                mating_pop_dict, crossover_prob
            )
            # crossover_output_dict = segmented_crossover(
            #    mating_pop_dict, swap_start_prob, swap_stop_prob
            # )

            mutation_output_dict = swap_mutate(
                crossover_output_dict, mutation_prob
            )
            # mutation_output_dict = scramble_mutate(
            #     crossover_output_dict, mutation_prob
            # )

            sequences_dict = add_children_to_parents(
                sequences_dict, mutation_output_dict
            )

        return sequences_dict
