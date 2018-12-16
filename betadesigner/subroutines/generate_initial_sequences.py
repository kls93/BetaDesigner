
import copy
import random
import sys
import networkx as nx
import numpy as np
import pandas as pd  # Required to slice input dataframe
from collections import OrderedDict

if __name__ == 'subroutines.generate_initial_sequences':
    from subroutines.find_parameters import initialise_class
else:
    from betadesigner.subroutines.find_parameters import initialise_class

# Initially, I should exclude contacts outside of the beta-strands of interest.
# All propensity / frequency dict names should be of the format:
# intorext_edgeorcentral_contprop1_contprop2_interactiontype_pairorindv_discorcont_propensityorfrequency
# (e.g. int_-_z_-_hb_indv_cont_propensity, ext_edg_-_-_vdw_pair_disc_frequency)


def find_indices(index_1, prop_val_1, node_val_1, scale):
    # Finds indices of propensity values in continuous scale for use in
    # interpolation calculation
    if index_1 < (scale.shape[0] - 1):
        if prop_val_1 <= node_val_1:
            index_2 = index_1 + 1
        elif prop_val_1 > node_val_1:
            index_2 = index_1 - 1
    else:
        index_2 = index_1 - 1

    return index_2


def linear_interpolation(node_val_1, aa_propensity_scale, dict_label,
                         node_val_2):
    # Interpolates propensity value for node property value
    if aa_propensity_scale.shape[0] == 2:
        if aa_propensity_scale[0][0] <= node_val_1 <= aa_propensity_scale[0][-1]:
            index_1 = (np.abs(aa_propensity_scale[0]-node_val_1)).argmin()
            prop_val_1 = aa_propensity_scale[0][index_1]
            propensity_1 = aa_propensity_scale[1][index_1]

            index_2 = find_indices(
                index_1, prop_val_1, node_val_1, aa_propensity_scale
            )

            # Linear interpolation
            prop_val_2 = aa_propensity_scale[0][index_2]
            propensity_2 = aa_propensity_scale[1][index_2]

            weight_1 = abs(prop_val_2 - node_val_1) / abs(prop_val_2 - prop_val_1)
            weight_2 = abs(prop_val_1 - node_val_1) / abs(prop_val_2 - prop_val_1)
            propensity = (propensity_1*weight_1) + (propensity_2*weight_2)

        else:
            propensity = np.nan
            print('Parameter values of input backbone coordinate model '
                  'structural features ({}) are outside of the range of '
                  'parameter values used to construct propensity scales - node '
                  'value = {}, parameter range = {}:{}'.format(
                      dict_label, node_val_1, aa_propensity_scale[0][0],
                      aa_propensity_scale[0][-1]
                ))

    elif aa_propensity_scale.shape[0] == 3:
        if (    aa_propensity_scale[0][0][0] <= node_val_1 <= aa_propensity_scale[0][0][-1]
            and aa_propensity_scale[1][0][0] <= node_val_2 <= aa_propensity_scale[1][-1][0]
        ):
            x_dimension = aa_propensity_scale[0][0]
            y_dimension = aa_propensity_scale[1][:,0]
            propensities = aa_propensity_scale[2]

            index_x1 = (np.abs(x_dimension-node_val_1)).argmin()
            prop_val_x1 = x_dimension[index_x1]
            index_x2 = find_indices(index_x1, prop_val_x1, node_val_1, x_dimension)
            prop_val_x2 = x_dimension[index_x2]

            index_y1 = (np.abs(y_dimension-node_val_2)).argmin()
            prop_val_y1 = y_dimension[index_y1]
            index_y2 = find_indices(index_y1, prop_val_y1, node_val_2, y_dimension)
            prop_val_y2 = y_dimension[index_y2]

            # Bilinear interpolation
            propensity_x1y1 = propensities[index_y1][index_x1]
            propensity_x1y2 = propensities[index_y2][index_x1]
            propensity_x2y1 = propensities[index_y1][index_x2]
            propensity_x2y2 = propensities[index_y2][index_x2]

            x1_weight = abs(prop_val_x2 - node_val_1) / abs(prop_val_x2 - prop_val_x1)
            x2_weight = abs(prop_val_x1 - node_val_1) / abs(prop_val_x2 - prop_val_x1)
            y1_weight = abs(prop_val_y2 - node_val_2) / abs(prop_val_y2 - prop_val_y1)
            y2_weight = abs(prop_val_y1 - node_val_2) / abs(prop_val_y2 - prop_val_y1)

            propensity_xy1 = (propensity_x1y1*x1_weight) + (propensity_x2y1*x2_weight)
            propensity_xy2 = (propensity_x1y2*x1_weight) + (propensity_x2y2*x2_weight)
            propensity = (propensity_xy1*y1_weight) + (propensity_xy2*y2_weight)

        else:
            propensity = np.nan
            print('Parameter values of input backbone coordinate model '
                  'structural features ({}) are outside of the range of '
                  'parameter values used to construct propensity scales - node '
                  'value 1 = {}, parameter range 1 = {}:{}, node value 2 = {}, '
                  'parameter range 2 = {}:{}'.format(
                      dict_label, node_val_1, aa_propensity_scale[0][0][0],
                      aa_propensity_scale[0][0][-1], node_val_2,
                      aa_propensity_scale[1][0][0], aa_propensity_scale[1][-1][0]
                ))

    return propensity


def random_shuffle(array_1, array_2, array_3):
    # Randomly shuffles arrays of network numbers, propensity and probability
    # values
    probability_array = np.transpose(np.array([array_1, array_2, array_3]))
    np.random.shuffle(probability_array)
    probability_array = np.transpose(probability_array)
    array_1 = probability_array[0]
    array_2 = probability_array[1]
    array_3 = probability_array[2]

    return array_1, array_2, array_3


def propensity_to_probability_distribution(sorted_network_num,
                                           sorted_node_indv_propensities):
    # Generates probability distribution from -ln(propensity) (~ free energy)
    # differences

    # Converts fitness scores into probabilities
    node_eqm_constant_values = np.exp(np.negative(sorted_node_indv_propensities))
    total = np.sum(node_eqm_constant_values)

    node_probabilities = np.full(sorted_node_indv_propensities.shape, np.nan)
    for index, eqm_constant in np.ndenumerate(node_eqm_constant_values):
        index = index[0]  # Numpy array indices are tuples
        node_probabilities[index] = eqm_constant / total

    # Randomly shuffles probability array, in order to avoid smallest
    # probabilities being grouped together at the beginning of the range
    (sorted_network_num, sorted_node_indv_propensities, node_probabilities
    ) = random_shuffle(
        sorted_network_num, sorted_node_indv_propensities, node_probabilities
    )

    return (sorted_network_num, sorted_node_indv_propensities,
            node_probabilities)


def frequency_to_probability_distribution(sorted_network_num,
                                          sorted_node_indv_propensities,
                                          propensity_or_frequency):
    # Generates probability distribution in which networks are weighted in
    # proportion to their rank propensity / frequency values

    if propensity_or_frequency == 'propensity':
        sorted_node_indv_propensities = np.array(
            range(1, (sorted_node_indv_propensities.shape[0]+1))
        )

    # Converts fitness scores into probabilities
    total = sorted_node_indv_propensities.sum()

    node_probabilities = np.full(sorted_node_indv_propensities.shape, np.nan)
    for index, propensity in np.ndenumerate(sorted_node_indv_propensities):
        index = index[0]  # Numpy array indices are tuples
        node_probabilities[index] = propensity / total

    # Randomly shuffles probability array, in order to avoid smallest
    # probabilities being grouped together at the beginning of the range
    (sorted_network_num, sorted_node_indv_propensities, node_probabilities
    ) = random_shuffle(
        sorted_network_num, sorted_node_indv_propensities, node_probabilities
    )

    return (sorted_network_num, sorted_node_indv_propensities,
            node_probabilities)


def calc_probability_distribution(sub_dicts, prop_freq_array, prop_or_freq,
                                  raw_or_rank):
    # Converts array of propensity / frequency values into a probability
    # distribution

    # Orders amino acids by their propensity values from least (+ve)
    # to most (-ve) favourable / by their frequency values from least (smaller)
    # to most (higher) favourable
    if prop_or_freq == 'propensity':
        sorted_prop_freq_array_indices = np.argsort(prop_freq_array)[::-1]
        sorted_prop_freq_array = np.sort(prop_freq_array)[::-1]
    elif prop_or_freq == 'frequency':
        sorted_prop_freq_array_indices = np.argsort(prop_freq_array)
        sorted_prop_freq_array = np.sort(prop_freq_array)

    # Converts propensity values into probability distribution
    if prop_or_freq == 'propensity' and raw_or_rank == 'raw':
        (sorted_prop_freq_array_indices, sorted_prop_freq_array,
         sorted_node_probabilities
        ) = propensity_to_probability_distribution(
            sorted_prop_freq_array_indices, sorted_prop_freq_array
        )
    elif (   (prop_or_freq == 'propensity' and raw_or_rank == 'rank')
          or (prop_or_freq == 'frequency')
    ):
        (sorted_prop_freq_array_indices, sorted_prop_freq_array,
         sorted_node_probabilities
        ) = frequency_to_probability_distribution(
            sorted_prop_freq_array_indices, sorted_prop_freq_array,
            prop_or_freq
        )

    return (sorted_prop_freq_array_indices, sorted_prop_freq_array,
            sorted_node_probabilities)


def gen_cumulative_probabilities(node_probabilities):
    # Converts raw probability values into a cumulative probability
    # distribution
    node_cumulative_probabilities = np.full(node_probabilities.shape, np.nan)
    cumulative_probability = 0
    for index, probability in np.ndenumerate(node_probabilities):
        index = index[0]  # Numpy array indices are tuples
        cumulative_probability += probability
        node_cumulative_probabilities[index] = cumulative_probability

    if round(node_cumulative_probabilities[-1], 4) != 1.0:
        sys.exit('ERROR: Cumulative probability = {}'.format(
            node_cumulative_probabilities[-1])
        )

    return node_cumulative_probabilities


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

            ext_surf_1_df = self.input_df[  (self.input_df['int_ext'] == 'exterior')
                                          & (self.input_df['sheet_number'] == sheet_ids[0])]
            ext_surf_1_df = ext_surf_1_df.reset_index(drop=True)
            surface_dfs['ext1'] = ext_surf_1_df  # Label must be "ext1"

            ext_surf_2_df = self.input_df[  (self.input_df['int_ext'] == 'exterior')
                                          & (self.input_df['sheet_number'] == sheet_ids[-1])]
            ext_surf_2_df = ext_surf_2_df.reset_index(drop=True)
            surface_dfs['ext2'] = ext_surf_2_df  # Label must be "ext2"

        return surface_dfs

    def generate_networks(self, surface_dfs):
        # Generates 2 (beta-barrels - interior and exterior surfaces) / 3
        # (beta-sandwiches - interior and (x2) exterior surfaces) networks of
        # interacting residues.

        # Defines dictionary of residue interaction types to include as network
        # edges. NOTE might want to provide these interactions as a program
        # input?
        interactions_dict = {'hb': 'hb_pairs',
                             'nhb': 'nhb_pairs',
                             'plusminus2': 'minus_2',
                             'vdw': 'van_der_waals'}

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
                    node = sub_df['domain_ids'][num] + sub_df['res_ids'][num]
                    int_or_ext = sub_df['int_ext'][num]
                    z_coord = sub_df['z_coords'][num]
                    phi = sub_df['phi'][num]
                    psi = sub_df['psi'][num]
                    phi_psi_class = sub_df['phi_psi_class'][num]
                    G.add_node(node, aa_id='UNK', int_ext=int_or_ext,
                               z=z_coord, phi=phi, psi=psi,
                               phipsiclass=phi_psi_class)
            elif self.barrel_or_sandwich == '2.60':
                for num in range(sub_df.shape[0]):
                    node = sub_df['domain_ids'][num] + sub_df['res_ids'][num]
                    int_or_ext = sub_df['int_ext'][num]
                    z_sandwich_coord = sub_df['sandwich_z_coords'][num]
                    z_strand_coord = sub_df['strand_z_coords'][num]
                    buried_surface_area = sub_df['buried_surface_area'][num]
                    edge_or_central = sub_df['edge_or_central'][num]
                    phi = sub_df['phi'][num]
                    psi = sub_df['psi'][num]
                    phi_psi_class = sub_df['phi_psi_class'][num]
                    G.add_node(node, aa_id='UNK', int_ext=int_or_ext,
                               zsandwich=z_sandwich_coord,
                               zstrand=z_strand_coord,
                               bsa=buried_surface_area, eoc=edge_or_central,
                               phi=phi, psi=psi, phipsiclass=phi_psi_class)

            domain_res_ids = list(G.nodes)

            # Adds edges (= residue interactions) to MultiGraph, labelled by
            # interaction type. The interactions considered are defined in
            # interactions_dict.
            for edge_label, interaction_type in interactions_dict.items():
                for num in range(sub_df.shape[0]):
                    res_1 = sub_df['domain_ids'][num] + sub_df['res_ids'][num]
                    res_list = sub_df[interaction_type][num]
                    if type(res_list) != list:
                        res_list = [res_list]

                    for res_2 in res_list:
                        res_2 = sub_df['domain_ids'][num] + res_2
                        # Only interactions between residues within sub_df are
                        # considered.
                        if not res_2 in domain_res_ids:
                            pass
                        else:
                            # Ensures interactions are only added to the
                            # network once.
                            if G.has_edge(res_1, res_2) is False:
                                G.add_edge(res_1, res_2, interaction=edge_label)
                            elif G.has_edge(res_1, res_2) is True:
                                attributes = [val for label, sub_dict in
                                              G[res_1][res_2].items() for key,
                                              val in sub_dict.items()]
                                if not edge_label in attributes:
                                    G.add_edge(res_1, res_2, interaction=edge_label)

            networks[surface_label] = G

        return networks

    def add_random_initial_side_chains(self, initial_sequences_dict,
                                       network_label, G):
        # For each network, assigns a random amino acid to each node in the
        # network to generate an initial sequence. Repeats 2*pop_size times to
        # generate a starting population of sequences to be fed into the
        # genetic algorithm.

        # Initialises dictionary of starting sequences
        initial_networks = OrderedDict()

        for num in range(2*self.pop_size):
            H = copy.deepcopy(G)

            new_node_aa_ids = OrderedDict()
            for node in list(H.nodes):
                random_aa = self.aa_list[random.randint(0, (len(self.aa_list)-1))]
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
        # propensity and / or frequency of the amino acid for the structural
        # features of that node. Repeats 2*pop_size times to generate a
        # starting population of sequences to be fed into the genetic algorithm.

        # Initialises dictionary of starting sequences
        initial_networks = OrderedDict()
        for num in range(2*self.pop_size):
            initial_networks[num] = copy.deepcopy(G)

        # Extracts individual amino acid propensity scales for the surface
        sub_dicts = OrderedDict({
            dict_label: scale_dict for dict_label, scale_dict in
            {**self.propensity_dicts, **self.frequency_dicts}.items() if
             (dict_label.split('_')[0] in [network_label[0:3], '-']
              and dict_label.split('_')[5] == 'indv')
        })

        for node in list(G.nodes):
            if self.barrel_or_sandwich == '2.60':
                # Filters propensity and frequency scales depending upon
                # whether the node is in an edge or a central strand
                eoc = G.nodes[node]['eoc']
                sub_dicts = OrderedDict(
                    {dict_label: scale_dict for dict_label,
                     scale_dict in sub_dicts.items() if
                     dict_label.split('_')[1] in [eoc, '-']}
                )

            # Calculates summed propensity for each amino acid across all
            # structural features considered in the design process
            node_indv_propensities = np.full(
                (len(self.aa_list), len(sub_dicts)), np.nan
            )
            node_indv_frequencies = np.full(
                (len(self.aa_list), len(sub_dicts)), np.nan
            )

            count = 0
            for dict_label, scale_dict in sub_dicts.items():
                node_prop_1 = dict_label.split('_')[2]
                node_prop_2 = dict_label.split('_')[3]
                node_val_1 = np.nan
                node_val_2 = np.nan

                if node_prop_1 != '-':
                    node_val_1 = G.nodes[node][node_prop_1]
                if node_prop_2 != '-':
                    node_val_2 = G.nodes[node][node_prop_2]
                if (
                         node_prop_1 == 'phi'
                     and node_prop_2 == 'psi'
                     and dict_label.split('_')[6] == 'disc'
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
                        for aa, aa_scale in scale_dict.items():
                            value = linear_interpolation(
                                node_val_1, aa_scale, dict_label, node_val_2
                            )

                            if dict_label.split('_')[7] == 'propensity':
                                node_indv_propensities[self.aa_list.index(aa)][count] = value
                            elif dict_label.split('_')[7] == 'frequency':
                                node_indv_frequencies[self.aa_list.index(aa)][count] = value

                elif dict_label.split('_')[6] == 'disc':
                    # Filter dataframe
                    scale_dict_copy = scale_dict.set_index('FASTA', drop=True)
                    for aa in self.aa_list:
                        if node_prop_1 == '-' and node_prop_2 == '-':
                            try:
                                value = scale_dict_copy.iloc[:,0][aa]
                            except KeyError:
                                pass
                        elif node_prop_1 == 'phi' and node_prop_2 == 'psi':
                            if not np.isnan(node_val_1):
                                try:
                                    value = scale_dict_copy[node_val_1][aa]
                                except KeyError:
                                    pass

                        if dict_label.split('_')[7] == 'propensity':
                            node_indv_propensities[self.aa_list.index(aa)][count] = value
                        elif dict_label.split('_')[7] == 'frequency':
                            node_indv_frequencies[self.aa_list.index(aa)][count] = value

                count += 1

            # Sums weighted propensities across structural features considered
            # NOTE: must take -ve logarithm of each individual propensity
            # score before summing (rather than taking the -ve logarithm of
            # the summed propensities)
            node_indv_propensities = np.negative(np.log(node_indv_propensities))

            for index, dict_label in enumerate(list(sub_dicts.keys())):
                dict_weight = self.dict_weights[dict_label]
                node_indv_propensities[:,index] *= dict_weight
                node_indv_frequencies[:,index] *= dict_weight

            node_indv_propensities = np.nanmean(node_indv_propensities, axis=1)
            node_indv_frequencies = np.nanmean(node_indv_frequencies, axis=1)

            # Removes NaN values
            nan_propensity = np.isnan(node_indv_propensities)
            node_indv_propensities = node_indv_propensities[~nan_propensity]
            node_indv_frequencies = node_indv_frequencies[~nan_propensity]

            nan_frequency = np.isnan(node_indv_frequencies)
            node_indv_propensities = node_indv_propensities[~nan_frequency]
            node_indv_frequencies = node_indv_frequencies[~nan_frequency]

            if node_indv_propensities.shape[0] == 0:
                sys.exit('Cannot select side chain identity for node {}'.format(node))

            # Converts propensities and frequencies into probability
            # distributions
            (node_indv_aa_index_propensity, node_indv_propensities,
             node_propensity_probabilities) = calc_probability_distribution(
                sub_dicts, node_indv_propensities, 'propensity', raw_or_rank
            )
            (node_indv_aa_index_frequency, node_indv_frequencies,
             node_frequency_probabilities) = calc_probability_distribution(
                 sub_dicts, node_indv_frequencies, 'frequency', raw_or_rank
            )

            node_probabilities = np.full(node_propensity_probabilities.shape, np.nan)
            for index in copy.deepcopy(node_indv_aa_index_propensity):
                prop_index = np.where(node_indv_aa_index_propensity == index)[0][0]
                freq_index = np.where(node_indv_aa_index_frequency == index)[0][0]
                propensity = node_propensity_probabilities[prop_index]
                frequency = node_frequency_probabilities[freq_index]

                probability = (  (propensity * self.propensity_weight['propensity'])
                               + (frequency * self.propensity_weight['frequency']))
                node_probabilities[prop_index] = probability

            node_cumulative_probabilities = gen_cumulative_probabilities(node_probabilities)

            # Selects amino acid weighted by its probability
            for num in range(2*self.pop_size):
                random_number = random.uniform(0, 1)
                nearest_index = (np.abs(node_cumulative_probabilities-random_number)).argmin()

                if node_cumulative_probabilities[nearest_index] >= random_number:
                    selected_aa = self.aa_list[node_indv_aa_index_propensity[nearest_index]]
                else:
                    selected_aa = self.aa_list[node_indv_aa_index_propensity[nearest_index+1]]

                nx.set_node_attributes(
                    initial_networks[num],
                    values={'{}'.format(node): {'aa_id': '{}'.format(selected_aa)}}
                )

        initial_sequences_dict[network_label] = initial_networks

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

        # Adds side-chains in order to generate a population of starting
        # sequences to be fed into the genetic algorithm
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
                raw_or_rank = self.method_initial_side_chains.replace('propensity', '')
                initial_sequences_dict = (
                    input_calcs.add_initial_side_chains_from_propensities(
                        initial_sequences_dict, network_label, G, raw_or_rank
                    )
                )

        return initial_sequences_dict
