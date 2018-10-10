
import os
import pickle
import random
import sys
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from collections import OrderedDict
prompt = '> '

# Initially, I should exclude contacts outside of the beta-strands of interest.


# Creates dataframes of residues on interior and exterior surfaces
class generate_ga_input():

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
            surface_dfs['ext_1'] = ext_surf_1_df

            ext_surf_2_df = self.input_df[(self.input_df['int_ext'] == 'exterior')
                                          & (self.input_df['sheet_number'] == sheet_ids[-1])]
            ext_surf_2_df = ext_surf_2_df.reset_index(drop=True)
            surface_dfs['ext_2'] = ext_surf_2_df

        return surface_dfs

    def generate_networks(self, surface_dfs):
        # Generates 2 (beta-barrels - interior and exterior surfaces) / 3
        # (beta-sandwiches - interior and (x2) exterior surfaces) networks of
        # interating residues.

        # Defines dictionary of residue interaction types to include as network
        # edges.
        interactions_dict = {'HB': 'hb_pairs',
                             'NHB': 'nhb_pairs',
                             'Minus 2': 'minus_2',
                             'Plus 2': 'plus_2'}

        # Creates networks of interacting residues on each surface
        networks = OrderedDict()

        for label, sub_df in surface_dfs.items():
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
                    G.add_node(node, aa_id='UNK', int_ext=label,
                               z_coord=z_coord)
            elif self.barrel_or_sandwich == '2.60':
                for num in range(sub_df.shape[0]):
                    node = sub_df['res_ids'][num]
                    z_coord = sub_df['sandwich_z_coords'][num]
                    bsa = sub_df['buried_surface_area'][num]
                    edge_or_central = sub_df['edge_or_central']
                    G.add_node(node, aa_id='UNK', int_ext=label,
                               z_coord=z_coord, buried_surface_area=bsa,
                               edge_or_central=edge_or_central)

            # Adds edges (= residue interactions) to MultiGraph, labelled by
            # interaction type. The interactions considered are defined in
            # interactions_dict.
            for label, interaction_type in interactions_dict.items():
                for num in range(sub_df.shape[0]):
                    res_1 = sub_df['res_ids'][num]

                    res_list = sub_df[interaction_type][num]
                    if type(res_list) != list:
                        res_list = [res_list]

                    if len(res_list) != 1:
                        sys.exit('Res list error {} {}'.format(res_1, interaction_type))
                    else:
                        res_2 = res_list[0]
                        # Only interactions between residues within sub_df are
                        # considered
                        if not res_2 in list(G.nodes()):
                            pass
                        else:
                            G.add_edge(res_1, res_2, label=label)

            networks[label] = G

        return networks

    def convert_propensities_to_probabilties(self):
        # Converts propensity scales into probabilities
        for label, propensity_scale in self.probability_dicts.items():
            

            probability_dicts[label] = probability_dict
        return probability_dicts

    def add_initial_side_chains(self, networks):
        # Uses user-specified propensity scales of the amino acids with
        # z-coordinate, buried surface area (sandwiches only) and edge vs.
        # central strands (sandwiches only).



        for network_label in list(networks.keys()):
            G = networks[network_label]

            # PROPENSITY SCALE DICTIONARIES NAMES MUST ALWAYS START WITH "int"
            # OR "ext", AS MUST NETWORK NAMES

            # Extracts propensity scales for the surface the residues in the
            # network are on
            sub_propensity_dicts = OrderedDict({
                dict_label, propensity_dict for dict_label, propensity_dict in
                self.propensity_dicts.items() if dict_label[0:3] == network_label[0:3]
            })

            # Creates a dictionary of cumulative probabilities for each
            # propensity scale
            for dict_label, propensity_dict in sub_propensity_dicts.items():
                node_z_dict = nx.get_node_attributes('G', 'z_coord')












                propensity_dict_1 = propensity_dicts['{}_z'.format(name[0:3])]

                # Dictionary of z-coordinates
                node_z_dict = nx.get_node_attributes(G, 'z_coord')

                for node, z_coord in node_z_dict.items():
                    print(node)
                    max_count = len(list(propensity_dict_1.keys()))
                    count = 0

                    aa_identities = ['']*max_count
                    aa_propensities = np.zeros((1, max_count))

                    for aa, array in propensity_dict_1.items():
                        aa_z_dist = array[0]
                        aa_propensity_dist = array[1]

                        index_1 = (np.abs(aa_z_dist - z_coord)).argmin()
                        z1 = aa_z_dist[index_1]
                        if z1 < z_coord:
                            index_2 = index_1 + 1
                            z2 = aa_z_dist[index_2]
                        elif z1 > z_coord:
                            index_2 = index_1 - 1
                            z2 = aa_z_dist[index_2]
                        else:
                            index_2 = ''

                        propensity_1 = aa_propensity_dist[index_1]
                        if index_2 != '':
                            propensity_2 = aa_propensity_dist[index_2]
                        else:
                            propensity_2 = ''

                        # Interpolate propensity values
                        if propensity_2 == '':
                            propensity = propensity_1
                        else:
                            propensity = ((((abs(z2-z_coord))*(propensity_1)) + ((abs(z_coord-z1))*(propensity_2))) / abs(z2 - z1))

                        aa_identities[count] = aa
                        aa_propensities[0][count] = propensity

                        count += 1

                    # Create probabilities array
                    aa_propensities = np.log(aa_propensities)
                    propensity_total = np.sum(aa_propensities)

                    aa_probabilities = aa_propensities / propensity_total
                    cumulative_probabilities = np.array([np.cumsum(aa_probabilities)])
                    print(cumulative_probabilities)

                    # Error with mixture of negative and positive log propensity values
                    if round(cumulative_probabilities[0][-1], 4) != 1.0:
                        sys.exit('ERROR {} {}_z: {}'.format(node, name[0:3], cumulative_probabilities[0][-1]))

                    random_number = random.uniform(0, 1)
                    nearest_index = (np.abs(cumulative_probabilities - z_coord)).argmin()

                    if cumulative_probabilities[0][nearest_index] >= random_number:
                        selected_aa = aa_identities[nearest_index]
                    elif cumulative_probabilities[0][nearest_index] < random_number:
                        selected_aa = aa_identities[nearest_index+1]

                    nx.set_node_attributes(G, {'{}'.format(node): {'aa_id': '{}'.format(selected_aa)}})

                networks[name] = G

                for node, z_coord in node_z_dict.items():
                    print(G.nodes[node]['aa_id'])











# Adds side chains to networks based upon propensities of individual amino
# acids for the considered structural features


for name in list(networks.keys()):
    G = networks[name]

    if barrel_or_sandwich == '2.40':
        propensity_dict_1 = propensity_dicts['{}_z'.format(name[0:3])]

        # Dictionary of z-coordinates
        node_z_dict = nx.get_node_attributes(G, 'z')

        for node, z_coord in node_z_dict.items():
            print(node)
            max_count = len(list(propensity_dict_1.keys()))
            count = 0

            aa_identities = ['']*max_count
            aa_propensities = np.zeros((1, max_count))

            for aa, array in propensity_dict_1.items():
                aa_z_dist = array[0]
                aa_propensity_dist = array[1]

                index_1 = (np.abs(aa_z_dist - z_coord)).argmin()
                z1 = aa_z_dist[index_1]
                if z1 < z_coord:
                    index_2 = index_1 + 1
                    z2 = aa_z_dist[index_2]
                elif z1 > z_coord:
                    index_2 = index_1 - 1
                    z2 = aa_z_dist[index_2]
                else:
                    index_2 = ''

                propensity_1 = aa_propensity_dist[index_1]
                if index_2 != '':
                    propensity_2 = aa_propensity_dist[index_2]
                else:
                    propensity_2 = ''

                # Interpolate propensity values
                if propensity_2 == '':
                    propensity = propensity_1
                else:
                    propensity = ((((abs(z2-z_coord))*(propensity_1)) + ((abs(z_coord-z1))*(propensity_2))) / abs(z2 - z1))

                aa_identities[count] = aa
                aa_propensities[0][count] = propensity

                count += 1

            # Create probabilities array
            aa_propensities = np.log(aa_propensities)
            propensity_total = np.sum(aa_propensities)

            aa_probabilities = aa_propensities / propensity_total
            cumulative_probabilities = np.array([np.cumsum(aa_probabilities)])
            print(cumulative_probabilities)

            # Error with mixture of negative and positive log propensity values
            if round(cumulative_probabilities[0][-1], 4) != 1.0:
                sys.exit('ERROR {} {}_z: {}'.format(node, name[0:3], cumulative_probabilities[0][-1]))

            random_number = random.uniform(0, 1)
            nearest_index = (np.abs(cumulative_probabilities - z_coord)).argmin()

            if cumulative_probabilities[0][nearest_index] >= random_number:
                selected_aa = aa_identities[nearest_index]
            elif cumulative_probabilities[0][nearest_index] < random_number:
                selected_aa = aa_identities[nearest_index+1]

            nx.set_node_attributes(G, {'{}'.format(node): {'aa_id': '{}'.format(selected_aa)}})

        networks[name] = G

        for node, z_coord in node_z_dict.items():
            print(G.nodes[node]['aa_id'])
