
import copy
import random
import networkx as nx
import numpy as np
import pandas as pd
from collections import OrderedDict

if __name__ == 'subroutines.generate_initial_sequences':
    from subroutines.find_parameters import initialise_ga_object
else:
    from betadesigner.subroutines.find_parameters import initialise_ga_object

# Initially, I should exclude contacts outside of the beta-strands of interest.
# All propensity / frequency dict names should be of the format:
# intorext_edgeorcentral_contprop1_contprop2_interactiontype_pairorindv_discorcont_propensityorfrequency
# (e.g. int_-_z_-_-_indv_cont_propensity, ext_edg_-_-_vdw_pair_disc_frequency)


def nansum_axis_1(input_array):
    """
    Overwrites behaviour of np.nansum to return np.nan instead of 0.0 when
    summing an array along axis=1 (i.e. along a row, or the x-axis if you like)
    consisting entirely of NaN values
    """

    output_array = []
    for row in input_array:
        if np.isnan(row).all():
            output_array.append(np.nan)
        else:
            output_array.append(np.nansum(row))
    output_array = np.array(output_array)

    return output_array


def find_indices(index_1, prop_val_1, node_val, scale):
    """
    Finds indices of propensity values in continuous scale for use in
    interpolation calculation
    """

    if index_1 < (scale.shape[0] - 1):
        if prop_val_1 <= node_val:
            index_2 = index_1 + 1
        elif prop_val_1 > node_val:
            index_2 = index_1 - 1
    else:
        index_2 = index_1 - 1

    return index_2


def linear_interpolation(node_val, aa_propensity_scale, dict_label):
    """
    Interpolates propensity value for node property value
    """

    if type(aa_propensity_scale) != np.ndarray:
        propensity = np.nan

    else:
        node_val_scale = aa_propensity_scale[0]
        propensity_scale = aa_propensity_scale[1]
        if node_val_scale[0] <= node_val <= node_val_scale[-1]:
            index_1 = (np.abs(node_val_scale-node_val)).argmin()
            prop_val_1 = node_val_scale[index_1]
            propensity_1 = propensity_scale[index_1]

            index_2 = find_indices(index_1, prop_val_1, node_val, node_val_scale)
            prop_val_2 = node_val_scale[index_2]
            propensity_2 = propensity_scale[index_2]

            # Linear interpolation
            weight_1 = abs(prop_val_2 - node_val) / abs(prop_val_2 - prop_val_1)
            weight_2 = abs(prop_val_1 - node_val) / abs(prop_val_2 - prop_val_1)
            propensity = (propensity_1*weight_1) + (propensity_2*weight_2)

        else:
            propensity = np.nan
            print('Parameter values of input backbone coordinate model '
                  'structural features ({}) are outside of the range of '
                  'parameter values used to construct propensity scales - node '
                  'value = {}, parameter range = ({}, {})'.format(
                      dict_label, node_val, aa_propensity_scale[0][0],
                      aa_propensity_scale[0][-1]
                ))

    return propensity


def combine_propensities(node_indv_propensities, node_indv_frequencies,
                         sub_dicts, dict_weights, aa_list):
    """
    Sums weighted propensities across structural features considered
    NOTE: must take -ve logarithm of each individual propensity
    score before summing (rather than taking the -ve logarithm of
    the summed propensities)
    """

    node_indv_propensities = np.negative(np.log(node_indv_propensities))

    for index, dict_label in enumerate(list(sub_dicts.keys())):
        dict_weight = dict_weights[dict_label]
        node_indv_propensities[:,index] *= dict_weight
        node_indv_frequencies[:,index] *= dict_weight

    node_indv_propensities = nansum_axis_1(node_indv_propensities, axis=1)
    node_indv_frequencies = nansum_axis_1(node_indv_frequencies, axis=1)

    # Removes NaN values
    filtered_aa_list = np.array(copy.deepcopy(aa_list))

    nan_propensity = np.isnan(node_indv_propensities)
    node_indv_propensities = node_indv_propensities[~nan_propensity]
    node_indv_frequencies = node_indv_frequencies[~nan_propensity]
    filtered_aa_list = filtered_aa_list[~nan_propensity]

    nan_frequency = np.isnan(node_indv_frequencies)
    node_indv_propensities = node_indv_propensities[~nan_frequency]
    node_indv_frequencies = node_indv_frequencies[~nan_frequency]
    filtered_aa_list = filtered_aa_list[~nan_frequency]

    return node_indv_propensities, node_indv_frequencies, filtered_aa_list

def random_shuffle(array_1, array_2, array_3):
    """
    Randomly shuffles arrays of network numbers, propensity and probability
    values
    """

    probability_array = np.transpose(np.array([array_1, array_2, array_3]))
    np.random.shuffle(probability_array)
    probability_array = np.transpose(probability_array)
    array_1 = probability_array[0]
    array_2 = probability_array[1]
    array_3 = probability_array[2]

    return array_1, array_2, array_3


def propensity_to_probability_distribution(index_num, node_indv_propensities):
    """
    Generates probability distribution from -ln(propensity) (~ free energy)
    differences
    """

    # Converts fitness scores into probabilities
    node_eqm_constant_values = np.exp(np.negative(node_indv_propensities))
    total = np.sum(node_eqm_constant_values)

    node_probabilities = np.full(node_indv_propensities.shape, np.nan)
    for index, eqm_constant in np.ndenumerate(node_eqm_constant_values):
        index = index[0]  # Numpy array indices are tuples
        node_probabilities[index] = eqm_constant / total

    # Randomly shuffles probability array, in order to avoid smallest
    # probabilities being grouped together at the beginning of the range
    (index_num, node_indv_propensities, node_probabilities
    ) = random_shuffle(
        index_num, node_indv_propensities, node_probabilities
    )
    index_num = index_num.astype(int)

    return index_num, node_indv_propensities, node_probabilities


def frequency_to_probability_distribution(index_num, node_indv_vals,
                                          propensity_or_frequency):
    """
    Generates probability distribution in which networks are weighted in
    proportion to their rank propensity / frequency values
    """

    if propensity_or_frequency == 'propensity':
        node_indv_vals = np.array(range(1, (node_indv_vals.shape[0]+1)))

    # Converts fitness scores into probabilities
    total = node_indv_vals.sum()

    node_probabilities = np.full(node_indv_vals.shape, np.nan)
    for index, val in np.ndenumerate(node_indv_vals):
        index = index[0]  # Numpy array indices are tuples
        node_probabilities[index] = val / total

    # Randomly shuffles probability array, in order to avoid smallest
    # probabilities being grouped together at the beginning of the range
    (index_num, node_indv_vals, node_probabilities) = random_shuffle(
        index_num, node_indv_vals, node_probabilities
    )
    index_num = index_num.astype(int)

    return index_num, node_indv_vals, node_probabilities


def calc_probability_distribution(sub_dicts, prop_freq_array, prop_or_freq,
                                  raw_or_rank):
    """
    Converts array of propensity / frequency values into a probability
    distribution
    """

    # Orders amino acids by their propensity values from least (+ve)
    # to most (-ve) favourable / by their frequency values from least (smaller)
    # to most (higher) favourable
    if prop_or_freq == 'propensity' and raw_or_rank == 'rank':
        prop_freq_array_indices = np.argsort(prop_freq_array)[::-1]
        prop_freq_array = np.sort(prop_freq_array)[::-1]
    else:
        prop_freq_array_indices = np.arange(prop_freq_array.shape[0])

    # Converts values into probability distribution
    if prop_or_freq == 'propensity' and raw_or_rank == 'raw':
        (prop_freq_array_indices, prop_freq_array, node_probabilities
        ) = propensity_to_probability_distribution(
            prop_freq_array_indices, prop_freq_array
        )
    elif (   (prop_or_freq == 'propensity' and raw_or_rank == 'rank')
          or (prop_or_freq == 'frequency')
    ):
        (prop_freq_array_indices, prop_freq_array, node_probabilities
        ) = frequency_to_probability_distribution(
            prop_freq_array_indices, prop_freq_array, prop_or_freq
        )

    return prop_freq_array_indices, prop_freq_array, node_probabilities


def gen_cumulative_probabilities(node_probabilities, node):
    """
    Converts raw probability values into a cumulative probability
    distribution
    """

    if node_probabilities.size == 0:
        raise Exception('No probability values calculated for {}'.format(node))

    node_cumulative_probabilities = np.full(node_probabilities.shape, np.nan)
    cumulative_probability = 0
    for index, probability in np.ndenumerate(node_probabilities):
        index = index[0]  # Numpy array indices are tuples
        cumulative_probability += probability
        node_cumulative_probabilities[index] = cumulative_probability

    if round(node_cumulative_probabilities[-1], 4) != 1.0:
        raise Exception('ERROR: Cumulative probability = {}'.format(
            node_cumulative_probabilities[-1])
        )

    return node_cumulative_probabilities


class gen_ga_input_calcs(initialise_ga_object):

    def __init__(self, params):
        initialise_ga_object.__init__(self, params)

    def slice_input_df(self):
        """
        Slices input dataframe into sub-dataframes of residues on the same
        surface of the structure
        """

        surface_dfs = OrderedDict()

        if self.barrel_or_sandwich == '2.40':
            int_surf_df = copy.deepcopy(self.input_df[self.input_df['int_ext'] == 'interior'])
            int_surf_df = int_surf_df.reset_index(drop=True)
            surface_dfs['int'] = int_surf_df

            ext_surf_df = copy.deepcopy(self.input_df[self.input_df['int_ext'] == 'exterior'])
            ext_surf_df = ext_surf_df.reset_index(drop=True)
            surface_dfs['ext'] = ext_surf_df

        elif self.barrel_or_sandwich == '2.60':
            int_surf_df = copy.deepcopy(self.input_df[self.input_df['int_ext'] == 'interior'])
            int_surf_df = int_surf_df.reset_index(drop=True)
            surface_dfs['int'] = int_surf_df

            sheet_ids = list(set(self.input_df['sheet_number'].tolist()))
            if len(sheet_ids) != 2:
                raise Exception('Incorrect number of sheets in input beta-sandwich structure')

            ext_surf_1_df = copy.deepcopy(self.input_df[  (self.input_df['int_ext'] == 'exterior')
                                                        & (self.input_df['sheet_number'] == sheet_ids[0])])
            ext_surf_1_df = ext_surf_1_df.reset_index(drop=True)
            surface_dfs['ext1'] = ext_surf_1_df  # Label must be "ext1"

            ext_surf_2_df = copy.deepcopy(self.input_df[  (self.input_df['int_ext'] == 'exterior')
                                                        & (self.input_df['sheet_number'] == sheet_ids[1])])
            ext_surf_2_df = ext_surf_2_df.reset_index(drop=True)
            surface_dfs['ext2'] = ext_surf_2_df  # Label must be "ext2"

        return surface_dfs

    def generate_networks(self, surface_dfs):
        """
        Generates 2 (beta-barrels - interior and exterior surfaces) / 3
        (beta-sandwiches - interior and (x2) exterior surfaces) networks of
        interacting residues.
        """

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

    def add_random_initial_side_chains(
        self, initial_sequences_dict, network_label, G
    ):
        """
        For each network, assigns a random amino acid to each node in the
        network to generate an initial sequence. Repeats 2*pop_size times to
        generate a starting population of sequences to be fed into the
        genetic algorithm.
        """

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

    def add_initial_side_chains_from_propensities(
        self, initial_sequences_dict, network_label, G, raw_or_rank, test, input_num
    ):
        """
        For each network, assigns an amino acid to each node in the
        network to generate an initial sequence. The likelihood of selection
        of an amino acid for a particular node is weighted by the raw / rank
        propensity and / or frequency of the amino acid for the structural
        features of that node. Repeats 2*pop_size times to generate a
        starting population of sequences to be fed into the genetic algorithm.
        """

        # TODO: Remove fixed numbers for dict name properties, and instead have
        # the user define

        # Initialises dictionary of starting sequences
        initial_networks = OrderedDict()
        for num in range(2*self.pop_size):
            initial_networks[num] = copy.deepcopy(G)

        # Extracts individual amino acid propensity scales for the surface
        intext_index = self.dict_name_indices['intorext']
        pairindv_index = self.dict_name_indices['pairorindv']
        prop_index = self.dict_name_indices['prop1']
        discorcont_index = self.dict_name_indices['discorcont']
        proporfreq_index = self.dict_name_indices['proporfreq']
        dicts = OrderedDict({
            dict_label: scale_dict for dict_label, scale_dict in
            {**self.propensity_dicts, **self.frequency_dicts}.items()
            if (dict_label.split('_')[intext_index] in [network_label[0:3], '-']
            and dict_label.split('_')[pairindv_index] == 'indv')
        })

        for node in list(G.nodes):
            if self.barrel_or_sandwich == '2.60':
                # Filters propensity and frequency scales depending upon
                # whether the node is in an edge or a central strand
                eoc = G.nodes[node]['eoc']
                eoc_index = self.dict_name_indices['edgeorcent']
                sub_dicts = OrderedDict(
                    {dict_label: scale_dict for dict_label,
                     scale_dict in dicts.items() if
                     dict_label.split('_')[eoc_index] in [eoc, '-']}
                )
            else:
                sub_dicts = dicts

            # Calculates summed propensity for each amino acid across all
            # structural features considered in the design process
            node_indv_propensities = np.full((len(self.aa_list), len(sub_dicts)), np.nan)
            node_indv_frequencies = np.full((len(self.aa_list), len(sub_dicts)), np.nan)

            dict_index = 0
            for dict_label, scale_dict in sub_dicts.items():
                node_prop = dict_label.split('_')[prop_index]
                node_val = np.nan

                if node_prop != '-':
                    try:
                        node_val = G.nodes[node][node_prop]
                    except KeyError:
                        raise KeyError('{} not defined for node {}'.format(node_prop, node))
                # Converts non-float values into np.nan
                if node_val in ['', 'nan', 'NaN', np.nan]:
                    node_val = np.nan

                value = np.nan
                if dict_label.split('_')[discorcont_index] == 'cont' and not np.isnan(node_val):
                    # Interpolate dictionary
                    for aa, aa_scale in scale_dict.items():
                        aa_index = self.aa_list.index(aa)
                        value = linear_interpolation(node_val, aa_scale, dict_label)

                        if dict_label.split('_')[proporfreq_index] == 'propensity':
                            node_indv_propensities[aa_index][dict_index] = value
                        elif dict_label.split('_')[proporfreq_index] == 'frequency':
                            node_indv_frequencies[aa_index][dict_index] = value

                elif dict_label.split('_')[discorcont_index] == 'disc':
                    # Filter dataframe
                    scale_dict_copy = copy.deepcopy(scale_dict).set_index('FASTA', drop=True)
                    for aa_index, aa in enumerate(self.aa_list):
                        if node_prop == '-':
                            try:
                                value = scale_dict_copy.iloc[:,0][aa]
                            except KeyError:
                                raise Exception('{} not defined in {}'.format(aa, dict_label))
                        elif node_prop == 'phipsi':
                            if not np.isnan(node_val):
                                try:
                                    value = scale_dict_copy[node_val][aa]
                                except KeyError:
                                    raise Exception('{} not defined in {}'.format(aa, dict_label))

                        if dict_label.split('_')[proporfreq_index] == 'propensity':
                            node_indv_propensities[aa_index][dict_index] = value
                        elif dict_label.split('_')[proporfreq_index] == 'frequency':
                            node_indv_frequencies[aa_index][dict_index] = value

                dict_index += 1

            (node_indv_propensities, node_indv_frequencies, filtered_aa_list
            ) = combine_propensities(
                node_indv_propensities, node_indv_frequencies, sub_dicts,
                self.dict_weights, self.aa_list
            )

            if node_indv_propensities.size == 0:
                raise Exception('Cannot select side chain identity for node {}'.format(node))

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
            for prop_index in copy.deepcopy(node_indv_aa_index_propensity):
                freq_index = np.where(node_indv_aa_index_frequency == index)[0][0]
                propensity = node_propensity_probabilities[prop_index]
                frequency = node_frequency_probabilities[freq_index]

                # Since propensity_weight is a hyperparameter to be optimised
                # with hyperopt, for initial sequence generation the propensity
                # and frequency scales are weighted equally
                probability = propensity + frequency
                node_probabilities[prop_index] = probability
            filtered_aa_list = filtered_aa_list[node_indv_aa_index_propensity]
            node_cumulative_probabilities = gen_cumulative_probabilities(
                node_probabilities, node
            )

            # Selects amino acid weighted by its probability
            for num in range(2*self.pop_size):
                if test is False:
                    random_number = random.uniform(0, 1)
                elif test is True:
                    random_number = input_num
                nearest_index = (np.abs(node_cumulative_probabilities-random_number)).argmin()

                if node_cumulative_probabilities[nearest_index] >= random_number:
                    selected_aa = filtered_aa_list[nearest_index]
                else:
                    selected_aa = filtered_aa_list[nearest_index+1]

                nx.set_node_attributes(
                    initial_networks[num],
                    values={'{}'.format(node): {'aa_id': '{}'.format(selected_aa)}}
                )

        initial_sequences_dict[network_label] = initial_networks

        return initial_sequences_dict


class gen_ga_input(initialise_ga_object):

    def __init__(self, params):
        initialise_ga_object.__init__(self, params)

    def initial_sequences_pipeline(self):
        """
        Wrapper function to generate initial population of side chains for
        input into genetic algorithm.
        """

        input_calcs = gen_ga_input_calcs(self.params)

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
                        initial_sequences_dict, network_label, G, raw_or_rank, False, ''
                    )
                )

        return initial_sequences_dict
