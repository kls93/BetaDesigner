
import copy
import random
import sys
import networkx as nx
import numpy as np
import pandas as pd
from collections import OrderedDict
from operator import itemgetter

if __name__ == 'subroutines.generate_initial_sequences':
    from subroutines.find_parameters import initialise_class
else:
    from betadesigner.subroutines.find_parameters import initialise_class

# Initially, I should exclude contacts outside of the beta-strands of interest.
# PROPENSITY SCALE DICTIONARIES NAMES MUST ALWAYS START WITH "int" OR "ext",
# AND END WITH "z" OR "bsa" AS MUST NETWORK NAMES. All propensity dict names of
# the format surface_structuralfeature_individualorcombined (e.g. int_z_indv)


def interpolate_propensities(node_prop, aa_propensity_scale, dict_label):
    # Interpolates propensity value for node property value
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
        try:
            prop_val_2 = aa_propensity_scale[0][index_2]
            propensity_2 = aa_propensity_scale[1][index_2]

            weight_1 = abs(prop_val_2 - node_prop)
            weight_2 = abs(prop_val_1 - node_prop)
            propensity = (((propensity_1*weight_1) + (propensity_2*weight_2))
                          / abs(prop_val_2 - prop_val_1))
        except IndexError:
            sys.exit('Parameter values of input backbone coordinate model '
                     'structural features ({}) are outside of the range of '
                     'parameter values used to construct propensity '
                     'scales'.format(dict_label))

    return propensity


def propensity_to_probability_distribution(sorted_network_num,
                                           sorted_node_indv_propensities,
                                           raw_or_rank):
    # Generates cumulative probability distribution from -ln(propensity)
    # (~ free energy) differences
    if raw_or_rank == 'rank':
        sorted_node_indv_propensities = np.array(
            range(1, (sorted_node_indv_propensities.shape[0]+1))
        )

    # Converts fitness scores into probabilities
    total = 0
    for index, propensity in np.ndenumerate(sorted_node_indv_propensities):
        if index[0] == 0:
            ref_propensity = propensity
            total += 1
        elif index[0] > 0:
            propensity_diff = abs(ref_propensity-propensity)
            total += (propensity_diff+1)

    node_probabilities = np.array([])
    for index, propensity in np.ndenumerate(sorted_node_indv_propensities):
        if index[0] == 0:
            ref_propensity = propensity
            probability = 1 / total
        elif index[0] > 0:
            probability = (abs(ref_propensity-propensity)+1) / total
        node_probabilities = np.append(
            node_probabilities, probability
        )

    # Randomly shuffles probability array before creating cumulative
    # probability distribution
    probability_array = np.transpose(np.array([sorted_network_num,
                                               sorted_node_indv_propensities,
                                               node_probabilities]))
    np.random.shuffle(probability_array)
    probability_array = np.transpose(probability_array)
    sorted_network_num = probability_array[0]
    sorted_node_indv_propensities = probability_array[1]
    node_probabilities = probability_array[2]

    node_cumulative_probabilities = np.array([])
    cumulative_probability = 0
    for index, probability in np.ndenumerate(node_probabilities):
        cumulative_probability += probability
        node_cumulative_probabilities = np.append(
            node_cumulative_probabilities, cumulative_probability
        )


    if round(node_cumulative_probabilities[-1], 4) != 1.0:
        sys.exit('ERROR {} {}: Cumulative probability = {}'.format(
            node, network_label, cumulative_probabilities_array[-1])
        )

    return (sorted_network_num, sorted_node_indv_propensities,
            node_cumulative_probabilities)


class gen_ga_input_calcs(initialise_class):

    def __init__(self, parameters):
        initialise_class.__init__(self, parameters)

    def slice_input_df(self):
        # Slices input dataframe into sub-dataframes of residues on the same
        # surface of the structure
        surface_dfs = OrderedDict()

        if self.barrel_or_sandwich == '2.40':
            int_surf_df = self.input_df[self.input_df['int_ext'] == 'interior']
            int_surf_df = int_surf_df.reset_index(drop=True)
            surface_dfs['int'] = int_surf_df

            ext_surf_df = self.input_df[self.input_df['int_ext'] == 'exterior']
            ext_surf_df = ext_surf_df.reset_index(drop=True)
            surface_dfs['ext'] = ext_surf_df

        elif self.barrel_or_sandwich == '2.60':
            int_surf_df = self.input_df[self.input_df['int_ext'] == 'interior']
            int_surf_df = int_surf_df.reset_index(drop=True)
            surface_dfs['int'] = int_surf_df

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
                    G.add_node(node, aa_id='UNK', int_ext=surface_label,
                               z=z_coord)
            elif self.barrel_or_sandwich == '2.60':
                for num in range(sub_df.shape[0]):
                    node = sub_df['res_ids'][num]
                    z_coord = sub_df['sandwich_z_coords'][num]
                    buried_surface_area = sub_df['buried_surface_area'][num]
                    edge_or_central = sub_df['edge_or_central']
                    G.add_node(node, aa_id='UNK', int_ext=surface_label,
                               z=z_coord, bsa=buried_surface_area,
                               eoc=edge_or_central)

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
                        sys.exit('Res list error {} {} {}'.format(
                            res_1, res_list, interaction_type
                        ))
                    elif len(res_list) == 1 and res_list not in [[''], [np.nan]]:
                        res_2 = res_list[0]
                        # Only interactions between residues within sub_df are
                        # considered.
                        if not res_2 in list(G.nodes()):
                            pass
                        else:
                            # Ensures interactions are only added to the
                            # network once.
                            if G.has_edge(res_1, res_2) is False:
                                G.add_edge(res_1, res_2, interaction=edge_label)
                            elif G.has_edge(res_1, res_2) is True:
                                attributes = [val for edge_label, sub_dict in
                                              G[res_1][res_2].items() for key,
                                              val in sub_dict.items()]
                                if not edge_label in attributes:
                                    G.add_edge(res_1, res_2, interaction=edge_label)

            networks[surface_label] = G

        return networks

    def add_random_initial_side_chains(self, initial_sequences_dict,
                                       network_label, G):
        # For each network, assigns a random amino acid to each node in the
        # network to generate an initial sequence. Repeats pop_size times to
        # generate a starting population of sequences to be fed into the
        # genetic algorithm.

        # Initialises dictionary of starting sequences
        initial_networks = OrderedDict()

        for num in range(self.pop_size):
            H = copy.deepcopy(G)

            new_node_aa_ids = OrderedDict()
            for node in list(H.nodes):
                random_aa = self.aas[random.randint(0, (len(self.aas)-1))]
                new_node_aa_ids[node] = {'aa_id': random_aa}
            nx.set_node_attributes(H, new_node_aa_ids)

            initial_networks[num] = H

        initial_sequences_dict[network_label] = initial_networks

        return initial_sequences_dict

    def add_initial_side_chains_from_propensities(self, initial_sequences_dict,
                                                  network_label, G, raw_or_rank):
        # For each network, assigns an amino acid to each node in the
        # network to generate an initial sequence. The likelihood of selection
        # of an amino acid for a particular node is weighted by the raw / rank
        # propensity of the amino acid for the structural features of that
        # node. Repeats pop_size times to generate a starting population of
        # sequences to be fed into the genetic algorithm.

        # Initialises dictionary of starting sequences
        initial_networks = OrderedDict()
        for num in range(self.pop_size):
            initial_networks[num] = copy.deepcopy(G)

        # Extracts individual amino acid propensity scales for the surface
        sub_propensity_dicts = OrderedDict({
            dict_label: propensity_dict for dict_label, propensity_dict in
            self.propensity_dicts.items() if
            (dict_label.split('_')[0] == network_label[0:3]
             and dict_label.split('_')[-1] == 'indv')
        })

        for node in list(G.nodes):
            # Calculates summed propensity for each amino acid across all
            # structural features considered in the design process
            node_indv_propensities = np.zeros((len(self.aas), len(sub_propensity_dicts)))

            count = 0
            for dict_label, propensity_dict in sub_propensity_dicts.items():
                node_prop = G.nodes[node][dict_label.split('_')[1]]

                for aa, aa_propensity_scale in propensity_dict.items():
                    propensity = interpolate_propensities(
                        node_prop, aa_propensity_scale, dict_label
                    )
                    node_indv_propensities[self.aas.index(aa)][count] = propensity

                count += 1

            # Sums propensities across structural features considered
            node_indv_propensities = np.sum(np.negative(np.log(node_indv_propensities)), axis=1)

            # Orders amino acids by their propensity values from least
            # to most favourable
            sorted_node_aa_ids = np.argsort(node_indv_propensities)[::-1]
            sorted_node_indv_propensities = np.sort(node_indv_propensities)[::-1]

            # Converts propensity values into probability distribution
            node_cumulative_probabilities = propensity_to_probability_distribution(
                sorted_node_indv_propensities, raw_or_rank
            )

            # Selects amino acid weighted by its probability
            for num in range(self.pop_size):
                random_number = random.uniform(0, 1)
                nearest_index = (np.abs(node_cumulative_probabilities-random_number)).argmin()

                if node_cumulative_probabilities[nearest_index] >= random_number:
                    selected_aa = sorted_node_aa_ids[nearest_index]
                else:
                    selected_aa = sorted_node_aa_ids[nearest_index+1]

                nx.set_node_attributes(
                    initial_networks[num],
                    {'{}'.format(node): {'aa_id': '{}'.format(selected_aa)}}
                )

        return initial_sequences_dict


class gen_ga_input_pipeline(initialise_class):

    def __init__(self, parameters):
        initialise_class.__init__(self, parameters)

    def initial_sequences_pipeline(self):
        # Pipeline function to generate initial population of side chains for
        # input into genetic algorithm.

        input_calcs = gen_ga_input_calcs(self.parameters)

        # Creates networks of interacting residues from input dataframe
        surface_dfs_dict = input_calcs.slice_input_df()
        networks_dict = input_calcs.generate_networks(surface_dfs_dict)

        # Adds side-chains onto networks using individual amino acid propensity
        # scales, in order to generate a population of starting sequences

        # Initialises dictionary of sequence populations for all networks
        initial_sequences_dict = OrderedDict()
        for network_label, G in networks_dict.items():
            print('Generating initial sequence population for {} surface of '
                  'backbone model'.format(network_label.split('_')[0]))
            if self.method_initial_side_chains == 'random':
                initial_sequences_dict = (
                    input_calcs.add_random_initial_side_chains(
                        initial_sequences_dict, network_label, G
                    )
                )
            elif self.method_initial_side_chains in ['rawpropensity', 'rankpropensity']:
                initial_sequences_dict = (
                    input_calcs.add_initial_side_chains_from_propensities(
                        initial_sequences_dict, network_label, G,
                        self.method_initial_side_chains
                    )
                )

        return initial_sequences_dict
