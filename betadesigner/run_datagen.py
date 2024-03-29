
# Run with isambard_dev (since DataGen requires isambard_dev). All other scripts
# in BetaDesigner are run with isambardv2.

import os
import pickle
import shutil
import pandas as pd
from collections import OrderedDict

"""
barrel_or_sandwich = input('Barrel or sandwich? ').lower()
if barrel_or_sandwich == 'barrel':
    barrel_or_sandwich = '2.40'
elif barrel_or_sandwich == 'sandwich':
    barrel_or_sandwich = '2.60'
pdb_code = input('Specify PDB accession code / structure name: ')
pdb_code = pdb_code.lower()  # Needs to be lower case to be recognised in the OPM
input_pdb_file = input('Specify absolute file path of PDB coordinates of backbone structure: ')
datagen_file_path = input('Specify absolute file path of DataGen wrapper script (datagen.py): ')
output_directory = input('Specify absolute path of output directory: ')
opm_directory = input('Specify absolute path of OPM database: ')
"""

# Defines input params, sets up directory framework and writes input file
# for DataGen
barrel_or_sandwich = '2.40'
pdb_code = '2jmm'  # Needs to be lower case to be recognised in the OPM
pdb_code = pdb_code.lower()
input_pdb_file = '/home/ks17361/isambard2/Test_design_results/Barrel_designs/2jmm.pdb'
datagen_file_path = '/home/ks17361/isambard_dev/DataGen/datagen/datagen.py'
output_directory = '/home/ks17361/isambard2/Test_design_results/Barrel_designs/DataGen_output/'
opm_directory = '/home/shared/structural_bioinformatics/DataGen_repositories/opm/'

if os.path.isdir('{}/Parent_assemblies'.format(output_directory)):
    shutil.rmtree('{}/Parent_assemblies'.format(output_directory))
os.makedirs('{}/Parent_assemblies'.format(output_directory))
os.system('cp {} {}/Parent_assemblies/{}.pdb'.format(input_pdb_file, output_directory, pdb_code))

# Writes input dataframes for DataGen
rec = []
atmnum = []
atmname = []
conformer = []
resname = []
chain = []
resnum = []
insertion = []
xpos = []
ypos = []
zpos = []
occ = []
bfac = []
element = []
charge = []
chain_num_ins = []
lines = []

with open(input_pdb_file, 'r') as f:
    for line in f.readlines():
        if line[0:6].strip() in ['ATOM', 'HETATM']:
            rec.append(line[0:6].strip())
            atmnum.append(int(line[6:11].strip()))
            atmname.append(line[12:16].strip())
            conformer.append(line[16:17].strip())
            resname.append(line[17:20].strip())
            chain.append(line[21:22].strip())
            resnum.append(int(line[22:26].strip()))
            insertion.append(line[26:27].strip())
            xpos.append(float(line[30:38].strip()))
            ypos.append(float(line[38:46].strip()))
            zpos.append(float(line[46:54].strip()))
            occ.append(float(line[54:60].strip()))
            bfac.append(float(line[60:66].strip()))
            element.append(line[76:78].strip())
            charge.append(line[78:80].strip())
            chain_num_ins.append(line[21:27].replace(' ', ''))
            # Removes alternate conformer labels from pdb file lines (but not
            # from dataframe)
            lines.append(line[:16] + ' ' + line[17:].strip('\n'))

pdb_df = pd.DataFrame(OrderedDict({'PDB_FILE_LINES': lines,
                                   'REC': rec,
                                   'ATMNUM': atmnum,
                                   'ATMNAME': atmname,
                                   'CONFORMER': conformer,
                                   'RESNAME': resname,
                                   'CHAIN': chain,
                                   'RESNUM': resnum,
                                   'INSCODE': insertion,
                                   'XPOS': xpos,
                                   'YPOS': ypos,
                                   'ZPOS': zpos,
                                   'OCC': occ,
                                   'BFAC': bfac,
                                   'ELEMENT': element,
                                   'CHARGE': charge,
                                   'RES_ID': chain_num_ins}))
df_dict = {pdb_code: pdb_df}
cdhit_df = pd.DataFrame({'PDB_CODE': ['{}'.format(pdb_code)],
                         'DOMAIN_ID': ['{}'.format(pdb_code)],
                         'CHAIN_NUM': [list(set(pdb_df['RES_ID'].tolist()))]})
dataframe_loc = '{}DataGen_stage_2_input_dataframes.pkl'.format(output_directory)
with open(dataframe_loc, 'wb') as pickle_file:
    pickle.dump((df_dict, cdhit_df), pickle_file)

# generate_initial_sequences DataGen (stage 2 of analysis)
datagen_input_file = '{}/DataGen_Input_stage_2.txt'.format(output_directory)
with open(datagen_input_file, 'w') as f:
    f.write('Stage: 2\n' +
            'Structure database: CATH\n' +
            'ID: {}\n'.format(barrel_or_sandwich) +
            'Working directory: {}\n'.format(output_directory) +
            'OPM database: {}\n'.format(opm_directory) +
            'Dataframes: {}\n'.format(dataframe_loc))
os.system('python {} -i {} --betadesigner'.format(datagen_file_path, datagen_input_file))

# Runs DataGen (stage 3 of analysis)
datagen_input_file = '{}/DataGen_Input_stage_3.txt'.format(output_directory)
with open(datagen_input_file, 'w') as f:
    f.write('Stage: 3\n' +
            'Structure database: CATH\n' +
            'ID: {}\n'.format(barrel_or_sandwich) +
            'Working directory: {}\n'.format(output_directory) +
            'OPM database: {}\n'.format(opm_directory) +
            'Radius: 5\n')
os.system('python {} -i {} --betadesigner'.format(datagen_file_path, datagen_input_file))

# Runs DataGen (stage 4 of analysis)
datagen_input_file = '{}/DataGen_Input_stage_3.txt'.format(output_directory)
with open(datagen_input_file, 'w') as f:
    f.write('Stage: 4\n' +
            'Structure database: CATH\n' +
            'ID: {}\n'.format(barrel_or_sandwich) +
            'Working directory: {}\n'.format(output_directory) +
            'OPM database: {}\n'.format(opm_directory))
os.system('python {} -i {} --betadesigner'.format(datagen_file_path, datagen_input_file))
