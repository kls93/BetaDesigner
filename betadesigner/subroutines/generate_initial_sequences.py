
import copy
import random
import sys
import networkx as nx
import numpy as np
import pandas as pd
from collections import OrderedDict
from operator import itemgetter

# Initially, I should exclude contacts outside of the beta-strands of interest.

# PROPENSITY SCALE DICTIONARIES NAMES MUST ALWAYS START WITH "int" OR "ext",
# AND END WITH "z" OR "bsa" AS MUST NETWORK NAMES


class generate_ga_input():
    # Creates dataframes of residues on interior and exterior surfaces

    def __init__(self, input_df, propensity_dicts, barrel_or_sandwich):
        self.input_df = input_df
        self.propensity_dicts = propensity_dicts
        self.barrel_or_sandwich = barrel_or_sandwich

        # OVERWRITE ONCE HAVE COMPLETED GENERATION OF PROPENSITY SCALES FROM
        # BETASTATS.
        self.propensity_dicts = OrderedDict({'int_z': {'ARG': np.array([[-50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50], [1.3, 1.3, 0.8, 0.4, 0.4, 0.2, 0.4, 0.4, 0.8, 1.3, 1.3]]),
                                                       'ASP': np.array([[-50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50], [1.3, 1.3, 0.8, 0.4, 0.4, 0.2, 0.4, 0.4, 0.8, 1.3, 1.3]]),
                                                       'GLY': np.array([[-50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50], [1.0, 1.0, 1.2, 1.4, 2.0, 2.5, 2.0, 1.4, 1.2, 1.0, 1.0]]),
                                                       'PHE': np.array([[-50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50], [0.7, 0.7, 0.5, 0.5, 0.3, 0.1, 0.3, 0.5, 0.5, 0.7, 0.7]]),
                                                       'VAL': np.array([[-50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50], [0.9, 0.9, 0.7, 0.7, 0.6, 0.5, 0.6, 0.7, 0.7, 0.9, 0.9]])},
                                             'ext_z': {'ARG': np.array([[-50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50], [1.2, 1.2, 0.6, 0.4, 0.3, 0.2, 0.3, 0.4, 0.6, 1.2, 1.2]]),
                                                       'ASP': np.array([[-50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50], [1.2, 1.2, 0.6, 0.4, 0.3, 0.2, 0.3, 0.4, 0.6, 1.2, 1.2]]),
                                                       'GLY': np.array([[-50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50], [1.0, 1.0, 1.2, 1.2, 1.4, 1.6, 1.4, 1.2, 1.2, 1.0, 1.0]]),
                                                       'PHE': np.array([[-50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50], [0.6, 0.6, 2.5, 2.0, 1.2, 0.8, 1.2, 2.0, 2.5, 0.6, 0.6]]),
                                                       'VAL': np.array([[-50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50], [0.8, 0.8, 1.3, 1.5, 1.7, 1.7, 1.7, 1.5, 1.3, 0.8, 0.8]])}
                                           })

    def filter_input_df(self):
        # Slices input dataframe into sub-dataframes of residues on the same
        # surface of the structure

        surface_dfs = OrderedDict()

        if self.barrel_or_sandwich == '2.40':
            int_surf_df = self.input_df[self.input_df['int_ext'] == 'interior']
            int_surf_df = int_surf_df.reset_index(drop=True)
            surface_dfs['int'] = int_surf_df  # Label must be "int"

            ext_surf_df = self.input_df[self.input_df['int_ext'] == 'exterior']
            ext_surf_df = ext_surf_df.reset_index(drop=True)
            surface_dfs['ext'] = ext_surf_df  # Label must be "ext"

        elif self.barrel_or_sandwich == '2.60':
            int_surf_df = self.input_df[self.input_df['int_ext'] == 'interior']
            int_surf_df = int_surf_df.reset_index(drop=True)
            surface_dfs['int'] = int_surf_df  # Label must be "int"

            sheet_ids = list(set(self.input_df['sheet_number'].tolist()))
            if len(sheet_ids) != 2:
                sys.exit('Incorrect number of sheets in input beta-sandwich structure')

            ext_surf_1_df = self.input_df[(self.input_df['int_ext'] == 'exterior')
                                          & (self.input_df['sheet_number'] == sheet_ids[0])]
            ext_surf_1_df = ext_surf_1_df.reset_index(drop=True)
            surface_dfs['ext1'] = ext_surf_1_df  # Label must be "ext1"

            ext_surf_2_df = self.input_df[(self.input_df['int_ext'] == 'exterior')
                                          & (self.input_df['sheet_number'] == sheet_ids[-1])]
            ext_surf_2_df = ext_surf_2_df.reset_index(drop=True)
            surface_dfs['ext2'] = ext_surf_2_df  # Label must be "ext2"

        return surface_dfs

    def generate_networks(self, surface_dfs):
        # Generates 2 (beta-barrels - interior and exterior surfaces) / 3
        # (beta-sandwiches - interior and (x2) exterior surfaces) networks of
        # interacting residues.

        # Defines dictionary of residue interaction types to include as network
        # edges.
        interactions_dict = {'HB': 'hb_pairs',
                             'NHB': 'nhb_pairs',
                             'Plus Minus 2': 'minus_2',
                             'Plus Minus 2': 'plus_2'}

        # Creates networks of interacting residues on each surface
        networks = OrderedDict()

        for surface_label, sub_df in surface_dfs.items():
            # Initialises MultiGraph (= undirected graph with self loops and
            # parallel edges)
            G = nx.MultiGraph()

            # Adds nodes (= residues) to MultiGraph, labelled with their
            # side-chain identity (initially set to unknown), z-coordinate,
            # buried surface area (sandwiches only) and whether they are edge
            # or central strands (sandwiches only).
            if self.barrel_or_sandwich == '2.40':
                for num in range(sub_df.shape[0]):
                    node = sub_df['res_ids'][num]
                    z_coord = sub_df['z_coords'][num]
                    G.add_node(node, aa_id='UNK', int_ext=surface_label, z=z_coord)
            elif self.barrel_or_sandwich == '2.60':
                for num in range(sub_df.shape[0]):
                    node = sub_df['res_ids'][num]
                    z_coord = sub_df['sandwich_z_coords'][num]
                    buried_surface_area = sub_df['buried_surface_area'][num]
                    edge_or_central = sub_df['edge_or_central']
                    G.add_node(node, aa_id='UNK', int_ext=surface_label, z=z_coord,
                               bsa=buried_surface_area, eoc=edge_or_central)

            # Adds edges (= residue interactions) to MultiGraph, labelled by
            # interaction type. The interactions considered are defined in
            # interactions_dict.
            for edge_label, interaction_type in interactions_dict.items():
                for num in range(sub_df.shape[0]):
                    res_1 = sub_df['res_ids'][num]

                    res_list = sub_df[interaction_type][num]
                    if type(res_list) != list:
                        res_list = [res_list]

                    if len(res_list) > 1:
                        print(res_list)
                        sys.exit('Res list error {} {}'.format(res_1, interaction_type))
                    elif len(res_list) == 1 and res_list != ['']:
                        res_2 = res_list[0]
                        # Only interactions between residues within sub_df are
                        # considered.
                        if not res_2 in list(G.nodes()):
                            pass
                        else:
                            # Ensures interactions are only added to the
                            # network once.
                            if G.has_edge(res_1, res_2) is False:
                                G.add_edge(res_1, res_2, label=edge_label)
                            elif G.has_edge(res_1, res_2) is True:
                                attributes = [val for edge_label, sub_dict in
                                              G[res_1][res_2].items() for key,
                                              val in sub_dict.items()]
                                if not edge_label in attributes:
                                    G.add_edge(res_1, res_2, label=edge_label)

            networks[surface_label] = G

        return networks

    def add_random_initial_side_chains(self, networks, pop_size):
        # Generates initial population of random sequences

        # Initialises dictionary of sequence populations for all networks
        sequences = OrderedDict()

        # Generates list of possible amino acid identities
        aas = list(self.propensity_dicts['int_z'].keys())

        # For each network, assigns a random amino acid to each node in the
        # network to generate an initial sequence. Repeats pop_size times to
        # generate a starting population of sequences to be fed into the
        # genetic algorithm.
        for network_label in list(networks.keys()):
            G = networks[network_label]
            # Initialises dictionary of starting sequences
            initial_networks = OrderedDict()

            for num in range(pop_size):
                H = copy.deepcopy(G)

                new_node_aa_ids = OrderedDict()
                for node in list(H.nodes):
                    random_aa = aas[random.randint(0, (len(aas)-1))]
                    new_node_aa_ids[node] = {'aa_id': random_aa}
                nx.set_node_attributes(H, new_node_aa_ids)

                initial_networks[num] = H

            sequences[network_label] = initial_networks

        return sequences

    def add_initial_side_chains_from_propensities(self, networks, pop_size,
                                                  raw_or_rank):
        # Uses user-specified propensity scales of the amino acids with
        # z-coordinate, buried surface area (sandwiches only) and edge vs.
        # central strands (sandwiches only).

        # Initialises dictionary of sequence populations for all networks
        sequences = OrderedDict()

        for network_label in list(networks.keys()):
            G = networks[network_label]

            # Initialises dictionary of starting sequences
            initial_networks = OrderedDict()

            for num in range(pop_size):
                # Extracts propensity scales for the surface the residues in
                # the network are on
                sub_propensity_dicts = OrderedDict({
                    dict_label: propensity_dict for dict_label, propensity_dict in
                    self.propensity_dicts.items() if
                    dict_label.split('_')[0] == network_label.split('_')[0][0:3]
                })

                # Creates copy of network
                H = copy.deepcopy(G)

                for node in list(H.nodes):
                    # Calculates summed propensity for each amino acid across
                    # all structural features considered in the design process
                    node_indv_propensities_dict = OrderedDict()
                    for aa in list(self.propensity_dicts['int_z'].keys()):
                        node_indv_propensities_dict[aa] = np.zeros((1, len(sub_propensity_dicts)))

                    print(node_indv_propensities_dict)

                    count = 0
                    for dict_label, propensity_dict in sub_propensity_dicts.items():
                        node_prop = H.nodes[node][dict_label.split('_')[-1]]
                        print(node)
                        print(node_prop)

                        for aa, aa_propensity_scale in propensity_dict.items():
                            # Calculates interpolated propensity value
                            index_1 = (np.abs(aa_propensity_scale[0]-node_prop)).argmin()
                            prop_val_1 = aa_propensity_scale[0][index_1]
                            print(prop_val_1)
                            propensity_1 = aa_propensity_scale[1][index_1]
                            print(propensity_1)

                            index_2 = ''
                            if prop_val_1 < node_prop:
                                index_2 = index_1 + 1
                            elif prop_val_1 > node_prop:
                                index_2 = index_1 - 1

                            if index_2 == '':
                                propensity == aa_propensity_scale[1][index_1]
                                print(propensity)
                            else:
                                prop_val_2 = aa_propensity_scale[0][index_2]
                                propensity_2 = aa_propensity_scale[1][index_2]
                                print(prop_val_2)
                                print(propensity_2)

                                weight_1 = abs(prop_val_2 - node_prop)
                                weight_2 = abs(prop_val_1 - node_prop)
                                propensity = (((propensity_1*weight_1) + (propensity_2*weight_2))
                                              / abs(prop_val_2 - prop_val_1))
                                print(propensity)

                            node_indv_propensities_dict[aa][0][count] = propensity

                        count += 1

                    # Sums propensities across structural features considered
                    for aa in list(node_indv_propensities_dict.keys()):
                        propensity_array = node_indv_propensities_dict[aa]
                        propensity_sum = np.sum(np.negative(np.log(propensity_array)))
                        node_indv_propensities_dict[aa] = propensity_sum


                    # Orders amino acids by their propensity values from least
                    # to most favourable
                    node_indv_propensities_dict = OrderedDict(sorted(
                        node_indv_propensities_dict.items(), key=itemgetter(1),
                        reverse=True
                    ))
                    print(node_indv_propensities_dict)

                    # Generates cumulative probability distribution from
                    # -ln(propensity) (~ free energy) differences
                    if raw_or_rank == 'raw':
                        propensity_diff_sum = 0
                        for index, propensity in enumerate(list(node_indv_propensities_dict.values())):
                            if index == 0:
                                ref_propensity = propensity
                            elif index > 0:
                                propensity_diff = abs(ref_propensity - propensity)
                                propensity_diff_sum += propensity_diff
                        print(propensity_diff_sum)

                        node_cumulative_probabilities_dict = OrderedDict()
                        cumulative_probability = 0
                        for index, propensity in enumerate(list(node_indv_propensities_dict.values())):
                            aa = list(node_indv_propensities_dict.keys())[index]

                            if index == 0:
                                ref_propensity = propensity
                                probability = 1 / propensity_diff_sum
                            elif index > 0:
                                probability = abs(ref_propensity-propensity) / propensity_diff_sum
                            cumulative_probability += probability
                            node_cumulative_probabilities_dict[aa] = cumulative_probability
                        print(node_cumulative_probabilities_dict)

                    # Generates cumulative probability distribution from ranks
                    # of -ln()propensity (~ free energy) values
                    elif raw_or_rank == 'rank':
                        largest_rank = len(node_indv_propensities_dict)
                        rank_sum = (largest_rank*(largest_rank+1)) / 2

                        node_cumulative_probabilities_dict = OrderedDict()
                        cumulative_probability = 0
                        for index, aa in enumerate(list(node_indv_propensities_dict.keys())):
                            probability = index / rank_sum
                            cumulative_probability += probability
                            node_cumulative_probabilities_dict[aa] = cumulative_probability
                        print(node_cumulative_probabilities_dict)

                    cumulative_probabilities_array = np.array(list(node_cumulative_probabilities_dict.values()))
                    if round(cumulative_probabilities_array[-1], 4) != 1.0:
                        sys.exit('ERROR {} {}_z: Cumulative probability = {}'.format(
                            node, network_label, cumulative_probabilities_array[-1])
                        )

                    # Selects amino acid weighted by its probability
                    random_number = random.uniform(0, 1)
                    nearest_index = (np.abs(cumulative_probabilities_array)).argmin()

                    if cumulative_probabilities_array[nearest_index] >= random_number:
                        selected_aa = list(node_cumulative_probabilities_dict.keys())[nearest_index]
                    else:
                        selected_aa = list(node_cumulative_probabilities_dict.keys())[nearest_index+1]

                    nx.set_node_attributes(H, {'{}'.format(node): {'aa_id': '{}'.format(selected_aa)}})

                initial_networks[num] = H
                print(nx.get_node_attributes(H, 'aa_id'))

            sequences[network_label] = initial_networks

        return sequences
