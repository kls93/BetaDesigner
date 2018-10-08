
import os
import pickle
import sys
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from collections import OrderedDict
prompt = '> '

# Currently will return all contacts of domain within parent assembly.
# Initially, I should exclude contacts outside of the beta-strands of interest.

"""
print('Specify absolute file path of input dataframe:')
input_df_loc = input(prompt).replace('\\', '/')
while not os.path.isfile(input_df_loc):
    print('Absolute file path of input dataframe not recognised')
    input_df_loc = input(prompt).replace('\\', '/')
    if os.path.isfile(input_df_loc):
        break

print('Barrel or sandwich?')
barrel_or_sandwich = input(prompt).lower()
while not barrel_or_sandwich in ['barrel', 'sandwich']:
    print('Please enter "barrel" or "sandwich":')
    barrel_or_sandwich = input(prompt).lower()
    if barrel_or_sandwich in ['barrel', 'sandwich']:
        break
    else:
        print('Structure type not recognised')
if barrel_or_sandwich == 'barrel':
    barrel_or_sandwich = '2.40'
elif barrel_or_sandwich == 'sandwich':
    barrel_or_sandwich = '2.60'
"""

input_df_loc = '/Users/ks17361/Documents/BetaDesigner_results/Program_output/Beta_res_dataframe.pkl'
barrel_or_sandwich = '2.40'

orig_df = pd.read_pickle(input_df_loc)

if len(set(orig_df['domain_ids'])) != 1:
    sys.exit('More than one structure listed in input dataframe')

# Creates dataframes of residues on interior and exterior surfaces
dfs = OrderedDict()
if barrel_or_sandwich == '2.40':
    int_surf_df = orig_df[orig_df['int_ext'] == 'interior']
    int_surf_df = int_surf_df.reset_index(drop=True)
    dfs['int'] = int_surf_df

    ext_surf_df = orig_df[orig_df['int_ext'] == 'exterior']
    ext_surf_df = ext_surf_df.reset_index(drop=True)
    dfs['ext'] = ext_surf_df

elif barrel_or_sandwich == '2.60':
    int_surf_df = orig_df[orig_df['int_ext'] == 'interior']
    int_surf_df = int_surf_df.reset_index(drop=True)
    dfs['int'] = int_surf_df

    sheet_ids = list(set(orig_df['sheet_number']))
    if len(sheet_ids) != 2:
        sys.exit('Incorrect number of sheets in input beta-sandwich structure')
    ext_surf_1_df = orig_df[(orig_df['int_ext'] == 'exterior')
                            & (orig_df['sheet_number'] == sheet_ids[0])]
    ext_surf_1_df = ext_surf_1_df.reset_index(drop=True)
    dfs['ext_1'] = ext_surf_1_df

    ext_surf_2_df = orig_df[(orig_df['int_ext'] == 'exterior')
                            & (orig_df['sheet_number'] == sheet_ids[-1])]
    ext_surf_2_df = ext_surf_2_df.reset_index(drop=True)
    dfs['ext_2'] = ext_surf_2_df

# Creates networks of interacting residues
networks = OrderedDict()
for name, sub_df in dfs.items():
    G = nx.MultiGraph()

    # Only interactions between residues within the beta-strands are considered
    if barrel_or_sandwich == '2.40':
        for num in range(sub_df.shape[0]):
            node = sub_df['res_id'][num]
            int_ext = name
            z = sub_df['z_coords'][num]
            G.add_node(node, fasta_id='UNK', int_ext=name, z=z_coord)

    elif barrel_or_sandwich == '2.60':
        for num in range(sub_df.shape[0]):
            node = sub_df['res_id'][num]
            int_ext = name
            z = sub_df['sandwich_z_coords'][num]
            bsa = sub_df['buried_surface_area'][num]
            G.add_node(node, fasta_id='UNK', int_ext=name, z=z_coord,
                       buried_surface_area=bsa)

    interactions_dict = {'HB': 'hb_pairs',
                         'NHB': 'nhb_pairs',
                         'Minus 2': 'minus_2',
                         'Plus 2': 'plus_2'}

    for label, interaction_type in interactions_dict.items():
        for num in range(sub_df.shape[0]):
            res_1 = sub_df['res_ids'][num]

            res_list = sub_df[interaction_type][num]
            if type(res_list) != list:
                res_list = [res_list]

            if len(res_list) > 1:
                sys.exit('Res list error {} {}'.format(res_1, interaction_type))
            elif len(res_list) == 0:
                pass
            else:
                res_2 = res_list[0]
                if not res_1 in list(G.nodes()):
                    pass
                elif not res_2 in list(G.nodes()):
                    pass
                else:
                    G.add_edge(res_1, res_2, label=label)
    networks[name] = G

# Adds side chains to networks based upon propensities of individual amino
# acids for the considered structural features
with open('/BetaDesigner_results/Program_input/Propensity_dicts.pkl', 'rb') as pickle_file:
    propensity_dicts = pickle.load(pickle_file)

# Converts propensities into probabilities
"""
probability_dicts = OrderedDict()
for name, propensity_dict in propensity_dicts.items():
    aa_names = []
    propensity_values = []
    for aa, propensity in propensity_dict.items():
        aa_names.append(aa)
        propensity_values.append(propensity)

    propensity_values = np.array(propensity_values)
    probability_values = np.log(propensity_values)
    probability_total = np.sum(probability_values)

    probability_dict = OrderedDict()
    cutoff = 0
    for index, probability in enumerate(list(probability_values)):
        new_cutoff = (probability/probability_total) + cutoff
        probability_dict[aa_names[index]] = np.array([cutoff, new_cutoff])

        cutoff = new_cutoff

    if float(cutoff) != 1.0:
        sys.exit('Error in calculating probability values')
"""

#
"""
if barrel_or_sandwich == '2.40':
    int_z_dict = probability_dicts['int_z']
    ext_z_dict = probability_dicts['ext_z']
elif barrel_or_sandwich == '2.60':
    int_z_dict = probability_dicts['int_z']
    ext_z_dict = probability_dicts['ext_z']
    int_bsa_dict = probability_dicts['int_bsa']
    ext_bsa_dict = probability_dicts['ext_bsa']
"""

for name, network in networks.items():
    if barrel_or_sandwich == '2.40':
        propensity_dict_1 = propensity_dicts['{}_z'.format(name[0:3])]

        # Dictionary of z-coordinates
        node_z_list = nx.get_node_attributes(G, 'z_coords')

        for node, z_coord in node_z_list.items():
            node_propensity_dict = OrderedDict()

            for aa, propensity_list in propensity_dict_1.items():
                z_coords = np.array(list(propensity_list.keys()))
                propensities = np.array(list(propensity_list.values()))

                index_1 = (np.abs(z_coords - z_coord)).argmin()
                if z_coords[index] < z_coord:
                    index_2 = index_1 + 1
                elif z_coords[index] > z_coord:
                    index_2 = index_1 - 1
                else:
                    index_2 = ''

                propensity_1 = propensities[index_1]
                if index_2 != '':
                    propensity_2 = propensities[index_2]

                propensities

    elif barrel_or_sandwich == '2.60':
        propensity_dict_1 = propensity_dicts['{}_z'.format(name[0:3])]
        propensity_dict_2 = propensity_dicts['{}_bsa'.format(name[0:3])]

        for node in network:
            node_z = nx.get_node_attributes(G, 'sandwich_z_coords')
            node_bsa = nx.get_node_attributes(G, 'buried_surface_area')
























            fgfgfgf
