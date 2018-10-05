
import os
import sys
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
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
dfs = {}
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
networks = {}
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
