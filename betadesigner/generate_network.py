
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

networks = {}
for name, sub_df in dfs.items():
    G = nx.MultiGraph()

    nodes = sub_df['res_ids'].tolist()
    for node in nodes:
        G.add_node(node, fasta_id='UNK')

    interactions_dict = {'HB': 'hb_pairs',
                         'NHB': 'nhb_pairs',
                         'Minus 2': 'minus_2',
                         'Plus 2': 'plus_2'}

    for label, interaction_type in interactions_dict.items():
        for num in range(sub_df.shape[0]):
            res_1 = sub_df['res_ids'][num]

            res_list = sub_df[interaction_type][num]
            if type(res_list) == str:
                res_list = [res_list]
            if len(res_list) > 1:
                sys.exit('Res list error {} {}'.format(res_1, interaction_type))
            elif len(res_list) == 0:
                pass
            else:
                res_2 = res_list[0]
                if not res_1 in list(G.nodes()):
                    # pass
                    print('Error: Res1 {} not in graph ({})'.format(res_1, interaction_type))
                elif not res_2 in list(G.nodes()):
                    # pass
                    print('Error: Res2 {} not in graph ({})'.format(res_2, interaction_type))
                else:
                    G.add_edge(res_1, res_2, label=label)
    networks[name] = G
    plt.clf()
    nx.draw_networkx(G, with_labels=True)
    plt.savefig(
        '{}/{}_network.png'.format('/'.join(input_df_loc.split('/')[:-1]), name)
    )
