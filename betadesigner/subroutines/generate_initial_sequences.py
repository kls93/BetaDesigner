
import copy
import random
import string
import networkx as nx
import numpy as np
import pandas as pd
from collections import OrderedDict

if __name__ == 'subroutines.generate_initial_sequences':
    from subroutines.find_parameters import initialise_ga_object
    from subroutines.calc_propensity_in_parallel import linear_interpolation
else:
    from betadesigner.subroutines.find_parameters import initialise_ga_object
    from betadesigner.subroutines.calc_propensity_in_parallel import linear_interpolation

# Initially, I should exclude contacts outside of the beta-strands of interest.
# All propensity / frequency dict names should be of the format:
# intorext_edgeorcentral_contprop1_contprop2_interactiontype_pairorindv_discorcont_propensityorfrequency
# (e.g. int_-_z_-_-_indv_cont_propensity, ext_edg_-_-_vdw_pair_disc_frequency)


def random_shuffle(array_1, array_2, array_3):
    """
    Randomly shuffles arrays of network numbers, propensity and probability
    values
    """

    type_array_1 = array_1.dtype
    type_array_2 = array_2.dtype
    type_array_3 = array_3.dtype

    probability_array = np.transpose(np.array([array_1, array_2, array_3]))
    np.random.shuffle(probability_array)
    probability_array = np.transpose(probability_array)
    array_1 = probability_array[0].astype(type_array_1)
    array_2 = probability_array[1].astype(type_array_2)
    array_3 = probability_array[2].astype(type_array_3)

    return array_1, array_2, array_3


def propensity_to_probability_distribution(
    index_num, node_indv_propensities, test=False
):
    """
    Generates probability distribution from -ln(propensity) (~ free energy)
    differences
    """

    # Checks for NaN values
    if np.isnan(node_indv_propensities).any():
        raise Exception('NaN value encountered in propensity array')

    # Converts fitness scores into probabilities
    node_eqm_constant_values = np.exp(np.negative(node_indv_propensities))
    total = np.sum(node_eqm_constant_values)

    node_probabilities = np.full(node_indv_propensities.shape, np.nan)
    for index, eqm_constant in np.ndenumerate(node_eqm_constant_values):
        index = index[0]  # Numpy array indices are tuples
        node_probabilities[index] = eqm_constant / total

    # Randomly shuffles probability array, in order to avoid smallest
    # probabilities being grouped together at the beginning of the range
    if test is False:
        (index_num, node_indv_propensities, node_probabilities
        ) = random_shuffle(
            index_num, node_indv_propensities, node_probabilities
        )

    return index_num, node_indv_propensities, node_probabilities


def frequency_to_probability_distribution(
    index_num, node_indv_vals, prop_or_freq, test=False
):
    """
    Generates probability distribution in which networks are weighted in
    proportion to their rank propensity / frequency values
    """

    # Checks for NaN values
    if np.isnan(node_indv_vals).any():
        raise Exception('NaN value encountered in {} array'.format(prop_or_freq))

    if prop_or_freq == 'propensity':
        node_indv_vals = np.array(range(1, (node_indv_vals.shape[0]+1)))

    # Converts fitness scores into probabilities
    total = node_indv_vals.sum()

    node_probabilities = np.full(node_indv_vals.shape, np.nan)
    for index, val in np.ndenumerate(node_indv_vals):
        index = index[0]  # Numpy array indices are tuples
        node_probabilities[index] = val / total

    # Randomly shuffles probability array, in order to avoid smallest
    # probabilities being grouped together at the beginning of the range
    if test is False:
        (index_num, node_indv_vals, node_probabilities) = random_shuffle(
            index_num, node_indv_vals, node_probabilities
        )

    return index_num, node_indv_vals, node_probabilities


def calc_probability_distribution(prop_freq_array, prop_or_freq, raw_or_rank,
                                  test=False):
    """
    Converts array of propensity / frequency values into a probability
    distribution
    """

    # If scoring by rank, orders amino acids by their propensity values from
    # least (+ve) to most (-ve) favourable
    if prop_or_freq == 'propensity' and raw_or_rank == 'rank':
        prop_freq_array_indices = np.argsort(prop_freq_array)[::-1]
        prop_freq_array = np.sort(prop_freq_array)[::-1]
    else:
        prop_freq_array_indices = np.arange(prop_freq_array.shape[0])

    # Converts values into probability distribution
    if prop_or_freq == 'propensity' and raw_or_rank == 'raw':
        (prop_freq_array_indices, prop_freq_array, node_probabilities
        ) = propensity_to_probability_distribution(
            prop_freq_array_indices, prop_freq_array, test
        )
    elif (   (prop_or_freq == 'propensity' and raw_or_rank == 'rank')
          or (prop_or_freq == 'frequency')
    ):
        (prop_freq_array_indices, prop_freq_array, node_probabilities
        ) = frequency_to_probability_distribution(
            prop_freq_array_indices, prop_freq_array, prop_or_freq, test
        )

    return prop_freq_array_indices, prop_freq_array, node_probabilities


def gen_cumulative_probabilities(node_probabilities, node, adjust_scale=False):
    """
    Converts raw probability values into a cumulative probability
    distribution
    """

    if node_probabilities.size == 0:
        raise Exception('No probability values calculated for {}'.format(node))

    if not all(type(val) == np.float64 for val in node_probabilities):
        raise TypeError(
            'Unexpected type encountered in node probability distribution '
            '{}'.format(node_probabilities)
        )

    if np.isnan(np.sum(node_probabilities)):
        raise ValueError(
            'NaN encountered in node probability distribution '
            '{}'.format(node_probabilities)
        )

    if adjust_scale is True:
        total = np.sum(node_probabilities)
        node_probabilities = node_probabilities / total

    node_cumulative_probabilities = np.full(node_probabilities.shape, np.nan)
    cumulative_probability = 0
    for index, probability in np.ndenumerate(node_probabilities):
        index = index[0]  # Numpy array indices are tuples
        cumulative_probability += probability
        node_cumulative_probabilities[index] = cumulative_probability

    if round(node_cumulative_probabilities[-1], 4) != 1.0:
        raise OSError.errno(
            'ERROR: Cumulative probability = {}'.format(node_cumulative_probabilities[-1])
        )

    return node_cumulative_probabilities


class gen_ga_input_calcs(initialise_ga_object):

    def __init__(self, params, test=False):
        initialise_ga_object.__init__(self, params, test)

    def generate_networks(self):
        """
        Generates networks of interacting residues
        """

        # Defines dictionary of residue interaction types to include as network
        # edges.
        #**N.B.** Might want to provide these interactions as a program input?
        # **N.B.** 'intra' in the interaction names dict refers to interactions
        # between residues in the same chain
        interactions = [['hb', 'hb_pairs', 'hb_pairs_fasta_intra'],
                        ['nhb', 'nhb_pairs', 'nhb_pairs_fasta_intra'],
                        ['plusminus2', 'minus_2', 'minus_2_fasta'],
                        ['plusminus2', 'plus_2', 'plus_2_fasta'],
                        ['plusminus1', 'minus_1', 'minus_1_fasta'],
                        ['plusminus1', 'plus_1', 'plus_1_fasta'],
                        ['vdw', 'van_der_waals', 'van_der_waals_fasta_intra']]

        # Initialises MultiGraph (= undirected graph with self loops and
        # parallel edges) network of interacting residues
        G = nx.MultiGraph()

        # Adds nodes (= residues) to MultiGraph, labelled with their side-chain
        # identity (initially set to unknown), z-coordinate, buried surface area
        # (sandwiches only) and whether they are edge or central strands
        # (sandwiches only).
        if self.barrel_or_sandwich == '2.40':
            for num in range(self.input_df.shape[0]):
                node = self.input_df['domain_ids'][num] + self.input_df['res_ids'][num]
                aa_id = self.input_df['fasta_seq'][num]
                int_or_ext = self.input_df['int_ext'][num][0:3]
                z_coord = self.input_df['z_coords'][num]
                try:
                    phi_psi_class = self.input_df['phi_psi_class'][num]
                except KeyError:
                    phi_psi_class = '-'
                if not int_or_ext in ['int', 'ext']:
                    raise ValueError('Residue {} has not been assigned to the '
                                     'interior or exterior surface of the input'
                                     ' beta-barrel structure'.format(node))
                G.add_node(node, type='strand', aa_id=aa_id, int_ext=int_or_ext,
                           eoc='-', z=z_coord, phipsi=phi_psi_class)
        elif self.barrel_or_sandwich == '2.60':
            for num in range(self.input_df.shape[0]):
                node = self.input_df['domain_ids'][num] + self.input_df['res_ids'][num]
                aa_id = self.input_df['fasta_seq'][num]
                int_or_ext = self.input_df['int_ext'][num][0:3]
                z_sandwich_coord = self.input_df['sandwich_z_coords'][num]
                #z_strand_coord = self.input_df['strand_z_coords'][num]
                #buried_surface_area = self.input_df['buried_surface_area'][num]
                edge_or_central = self.input_df['edge_or_central'][num]
                try:
                    phi_psi_class = self.input_df['phi_psi_class'][num]
                except KeyError:
                    phi_psi_class = '-'
                if not int_or_ext in ['int', 'ext']:
                    raise ValueError('Residue {} has not been assigned to the '
                                     'interior or exterior surface of the input'
                                     ' beta-barrel structure'.format(node))
                G.add_node(node, type='strand', aa_id=aa_id, int_ext=int_or_ext,
                           z=z_sandwich_coord,
                           #zstrand=z_strand_coord, bsa=buried_surface_area,
                           eoc=edge_or_central,
                           phipsi=phi_psi_class)

        domain_res_ids = list(G.nodes())

        # Adds edges (= residue interactions) to MultiGraph, labelled by
        # interaction type. The interactions considered are defined in
        # interactions_dict.
        for int_list in interactions:
            edge_label = int_list[0]
            int_name = int_list[1]
            int_fasta = int_list[2]

            for num in range(self.input_df.shape[0]):
                res_1 = self.input_df['domain_ids'][num] + self.input_df['res_ids'][num]
                res_list = self.input_df[int_name][num]
                if type(res_list) != list:
                    res_list = [res_list]

                for res_index, res_2 in enumerate(res_list):
                    res_2 = self.input_df['domain_ids'][num] + res_2
                    # Accounts for interactions between residue pairs where one
                    # residue is in the beta-barrel/sandwich domain and the
                    # other is within a loop region
                    aa_id = self.input_df[int_fasta][num][res_index]
                    if not res_2 in list(G.nodes()):
                        G.add_node(res_2, type='loop', aa_id=aa_id)
                    if aa_id != G.nodes()[res_2]['aa_id']:
                        print(aa_id, G.nodes()[res_2]['aa_id'])
                        raise ValueError(
                            'Identity of node {} is inconsistent according to '
                            'the pairwise interactions listed in {} '
                            '{}'.format(res_2, self.input_df_path, edge_label)
                        )

                    # Ensures interactions are only added to the network once
                    if G.has_edge(res_1, res_2) is False:
                        G.add_edge(res_1, res_2, interaction=edge_label)
                    elif G.has_edge(res_1, res_2) is True:
                        attributes = [val for label, sub_dict in
                                      dict(G[res_1][res_2]).items() for key,
                                      val in sub_dict.items()]
                        if not edge_label in attributes:
                            G.add_edge(res_1, res_2, interaction=edge_label)

        return G

    def add_random_initial_side_chains(self, G, test=False, input_aa={}):
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
            for node in list(H.nodes()):
                if H.nodes()[node]['type'] == 'loop':
                    continue
                if test is False:
                    random_aa = self.aa_list[random.randint(0, (len(self.aa_list)-1))]
                elif test is True:
                    random_aa = input_aa[node]
                new_node_aa_ids[node] = {'aa_id': random_aa}
            nx.set_node_attributes(H, new_node_aa_ids)

            unique_id = ''.join(
                [random.choice(string.ascii_letters + string.digits)
                for i in range(10)]
            )
            initial_networks[unique_id] = H

        return initial_networks

    def add_initial_side_chains_from_propensities(
        self, G, raw_or_rank, test=False, input_num={}
    ):
        """
        For each network, assigns an amino acid to each node in the network to
        generate an initial sequence. The likelihood of selection of an amino
        acid for a particular node is weighted by the raw / rank propensity and
        / or frequency of the amino acid for the structural features of that
        node. N.B. Only uses individual (i.e. not pairwise) propensity /
        frequency scores for this initial side chain assignment. Repeats
        2*pop_size times to generate a starting population of sequences to be
        fed into the genetic algorithm.
        """

        # Initialises dictionary of starting sequences
        initial_networks = OrderedDict()
        ids_list = []
        for num in range(2*self.pop_size):
            unique_id = ''.join(
                [random.choice(string.ascii_letters + string.digits)
                for i in range(10)]
            )
            initial_networks[unique_id] = copy.deepcopy(G)
            ids_list.append(unique_id)

        # Extracts individual amino acid propensity scales for the surface
        intext_index = self.dict_name_indices['intorext']
        eoc_index = self.dict_name_indices['edgeorcent']
        prop_index = self.dict_name_indices['prop1']
        pairindv_index = self.dict_name_indices['pairorindv']
        discorcont_index = self.dict_name_indices['discorcont']
        proporfreq_index = self.dict_name_indices['proporfreq']
        dicts = OrderedDict({
            dict_label: scale_dict for dict_label, scale_dict in
            {**self.propensity_dicts, **self.frequency_dicts}.items()
            if dict_label.split('_')[pairindv_index] == 'indv'
        })

        for node_index, node in enumerate(list(G.nodes())):
            if G.nodes()[node]['type'] == 'loop':
                continue

            int_ext = G.nodes()[node]['int_ext']
            sub_dicts = OrderedDict(
                {dict_label: scale_dict for dict_label, scale_dict in dicts.items()
                 if dict_label.split('_')[intext_index] in [int_ext, '-']}
            )
            if self.barrel_or_sandwich == '2.60':
                # Filters propensity and frequency scales depending upon
                # whether the node is in an edge or a central strand
                eoc = G.nodes()[node]['eoc']
                sub_dicts = OrderedDict(
                    {dict_label: scale_dict for dict_label, scale_dict in sub_dicts.items()
                     if dict_label.split('_')[eoc_index] in [eoc, '-']}
                )

            if sub_dicts == {}:
                if test is True:  # Avoids code crashing if e.g. there are no
                # dicts for ext residues
                    continue
                else:
                    raise ValueError('No propensity or scoring metrics '
                                     'available for node {}'.format(node))

            # Calculates summed propensity for each amino acid across all
            # structural features considered in the design process
            node_indv_propensities = np.full((len(sub_dicts), len(self.aa_list)), np.nan)
            node_indv_frequencies = np.full((len(sub_dicts), len(self.aa_list)), np.nan)

            for dict_index, dict_label in enumerate(list(sub_dicts.keys())):
                scale_dict = sub_dicts[dict_label]
                dict_weight = self.dict_weights[dict_label]
                node_prop = dict_label.split('_')[prop_index]
                node_val = np.nan

                if node_prop != '-':
                    try:
                        node_val = G.nodes()[node][node_prop]
                    except KeyError:
                        raise KeyError('{} not defined for node {}'.format(node_prop, node))
                # Converts non-float values into np.nan
                if node_val in ['', 'nan', 'NaN', np.nan]:
                    node_val = np.nan

                value = np.nan
                if dict_label.split('_')[discorcont_index] == 'cont' and not np.isnan(node_val):
                    if dict_label.split('_')[proporfreq_index] != 'propensity':
                        raise Exception(
                            'Unexpected dictionary {} - expect only continuous '
                            'propensity dictionaries'.format(dict_label)
                        )

                    # Interpolate dictionary
                    for aa, aa_scale in scale_dict.items():
                        aa_index = self.aa_list.index(aa)
                        value = linear_interpolation(node_val, aa_scale, dict_label)

                        if value <= 0:
                            raise ValueError(
                                '{} returned after interpolation of {} for node'
                                ' {}'.format(value, dict_label, node)
                            )

                        if np.isnan(value):
                            if dict_label.split('_')[proporfreq_index] == 'propensity':
                                value = 0.0001   # Since dataset used to generate
                                # prop and freq dicts is ~10,000 aas => smallest
                                # propensity could be is ((1/5000)/(5001/10000)
                                # = 0.0004 (for a discrete propensity))

                        if dict_label.split('_')[proporfreq_index] == 'propensity':
                            node_indv_propensities[dict_index][aa_index] = (
                                dict_weight*np.negative(np.log(value))
                            )

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
                                    raise Exception('{},{} not defined in {}'.format(
                                        node_val, aa, dict_label
                                    ))

                        if value <= 0:
                            raise ValueError(
                                '{} returned after interpolation of {} for node'
                                ' {}'.format(value, dict_label, node)
                            )

                        if np.isnan(value):
                            if dict_label.split('_')[proporfreq_index] == 'propensity':
                                value = 0.0001  # Since dataset used to generate
                                # prop and freq dicts is ~10,000 aas => smallest
                                # propensity could be is ((1/5000)/(5001/10000)
                                # = 0.0004 (for a discrete propensity))
                            else:
                                value = 0

                        if dict_label.split('_')[proporfreq_index] == 'propensity':
                            node_indv_propensities[dict_index][aa_index] = (
                                dict_weight*np.negative(np.log(value))
                            )
                        elif dict_label.split('_')[proporfreq_index] == 'frequency':
                            node_indv_frequencies[dict_index][aa_index] = dict_weight*value

            # Sums propensity and frequency values, then filters to remove amino
            # acids with a propensity / frequency of 0
            node_indv_propensities = np.nansum(node_indv_propensities, axis=0)
            node_indv_frequencies = np.nansum(node_indv_frequencies, axis=0)

            if (
                   set(node_indv_propensities) == {0}
                or (self.frequency_dicts != {} and set(node_indv_frequencies) == {0})
            ):
                raise Exception('Cannot select side chain identity for node '
                                '{}'.format(node))

            # Removes aas for which all propensity and/or frequency readings (as
            # appropriate) are np.nan
            filt_aa_list = []
            filt_node_prop = []
            filt_node_freq = []
            for aa_index, aa in enumerate(self.aa_list):
                prop = node_indv_propensities[aa_index]
                freq = node_indv_frequencies[aa_index]
                if (
                       (self.frequency_dicts == {} and prop == 0)
                    or (self.frequency_dicts != {} and prop == 0 and freq == 0)
                ):
                    continue
                filt_aa_list.append(aa)
                filt_node_prop.append(prop)
                filt_node_freq.append(freq)
            filt_aa_list = np.array(filt_aa_list)
            filt_node_prop = np.array(filt_node_prop)
            filt_node_freq = np.array(filt_node_freq)

            # Converts propensities and frequencies into probability
            # distributions
            (node_prop_index, node_prop, node_prop_probabilities
            ) = calc_probability_distribution(
                node_indv_propensities, 'propensity', raw_or_rank, test
            )
            node_prop_index = node_prop_index.astype(int)
            if self.frequency_dicts != {}:
                (node_freq_index, node_freq, node_freq_probabilities
                ) = calc_probability_distribution(
                     filt_node_freq, 'frequency', raw_or_rank, test
                )
            else:
                node_freq_index = copy.deepcopy(node_prop_index)
                node_freq = np.full(filt_node_freq.shape, 0)
                node_freq_probabilities = np.full(filt_node_freq.shape, 0)
            node_freq_index = node_freq_index.astype(int)

            node_probabilities = np.full(node_prop_probabilities.shape, np.nan)
            for index_prop, aa in np.ndenumerate(copy.deepcopy(node_prop_index)):
                index_freq = np.where(node_freq_index == aa)[0][0]
                propensity = node_prop_probabilities[index_prop]
                frequency = node_freq_probabilities[index_freq]

                # Since propensity_weight is a hyperparameter to be optimised
                # with hyperopt, for initial sequence generation the propensity
                # and frequency scales are weighted equally (unless not
                # performing hyperparameter optimisation, in which case use the
                # propensity_weight specified in the input file)
                try:
                    prop_weight = self.propensity_weight
                    freq_weight = 1 - self.propensity_weight
                except AttributeError:
                    prop_weight = 0.5
                    freq_weight = 0.5
                probability = (prop_weight*propensity) + (freq_weight*frequency)
                node_probabilities[index_prop] = probability
            filt_aa_list = filt_aa_list[node_prop_index]
            node_cumulative_probabilities = gen_cumulative_probabilities(
                node_probabilities, node
            )

            # Selects amino acid weighted by its probability
            for unique_id in ids_list:
                if test is False:
                    random_number = random.uniform(0, 1)
                elif test is True:
                    random_number = input_num[node]
                nearest_index = (np.abs(node_cumulative_probabilities-random_number)).argmin()

                if node_cumulative_probabilities[nearest_index] >= random_number:
                    selected_aa = filt_aa_list[nearest_index]
                else:
                    selected_aa = filt_aa_list[nearest_index+1]

                nx.set_node_attributes(
                    initial_networks[unique_id],
                    values={'{}'.format(node): {'aa_id': '{}'.format(selected_aa)}}
                )

        return initial_networks


class gen_ga_input(initialise_ga_object):

    def __init__(self, params, test=False):
        initialise_ga_object.__init__(self, params, test)

    def initial_sequences_pipeline(self):
        """
        Wrapper function to generate initial population of side chains for
        input into genetic algorithm.
        """

        input_calcs = gen_ga_input_calcs(self.params)

        # Creates networks of interacting residues from input dataframe
        if self.barrel_or_sandwich == '2.60':
            sheet_ids = list(set(self.input_df['sheet_number'].tolist()))
            if len(sheet_ids) != 2:
                raise Exception(
                    'Incorrect number of sheets in input beta-sandwich structure'
                )
        initial_network = input_calcs.generate_networks()

        # Adds side-chains in order to generate a population of starting
        # sequences to be fed into the genetic algorithm
        initial_sequences_dict = OrderedDict()
        print('Generating initial sequence population for backbone model')
        if self.method_initial_side_chains == 'random':
            initial_sequences_dict = input_calcs.add_random_initial_side_chains(initial_network)
        elif self.method_initial_side_chains in ['rawpropensity', 'rankpropensity']:
            raw_or_rank = self.method_initial_side_chains.replace('propensity', '')
            initial_sequences_dict = input_calcs.add_initial_side_chains_from_propensities(
                initial_network, raw_or_rank, False, ''
            )

        return initial_network, initial_sequences_dict
