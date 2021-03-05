
# Running a subset of a program in parallel with scoop / multiprocessing in
# Python requires the code in question to be separated into its own file and
# executed within an if __name__ == '__main__' clause

import argparse
import copy
import os
import pickle
import random
import string
import numpy as np
import pandas as pd
from collections import OrderedDict
from scoop import futures


def parse_rosetta_score_file(score_file_lines):
    """
    """

    if not type(score_file_lines) is list:
        total_energy = np.nan
    else:
        total_energy_index = score_file_lines[1].split().index('total_score')
        total_energy = float(score_file_lines[2].split()[total_energy_index])

    return total_energy


def parse_rosetta_pdb_file(pdb_path, rosetta_lines, pdb_lines):
    """
    """

    res_energies_dict = OrderedDict()

    pdb_res_list = []
    rosetta_res_list = []
    res_energies_list = []

    # Matches renumbered Rosetta res ids to res ids in the input PDB file
    for line in pdb_lines:
        # Will return '' rather than error if list index out of bounds
        res_id = '{}_{}_{}'.format(line[17:20], line[21:22], line[22:26].strip())
        if line[0:6].strip() in ['ATOM', 'HETATM'] and not res_id in pdb_res_list:
            pdb_res_list.append(res_id)

    if not type(rosetta_lines) is list:
        for res_id in pdb_res_list:
            res_energies_dict[res_id] = np.nan
        return res_energies_dict

    start = False
    res_energy_index = ''
    dropped_lines = ['label', 'weights', 'pose', 'MEM', 'VRT',
                     '#END_POSE_ENERGIES_TABLE']
    for line in rosetta_lines:
        if line.startswith('label'):
            start = True
            res_energy_index = line.split().index('total')

        if start is True and not any(line.startswith(x) for x in dropped_lines):
            line = line.split()
            rosetta_res_list.append(line[0])
            res_energies_list.append(float(line[res_energy_index]))

    try:
        res_ids_convers = pd.DataFrame({'PDB_res_ids': pdb_res_list,
                                        'Rosetta_res_ids': rosetta_res_list,
                                        'Res_energies': res_energies_list})
    except ValueError:
        res_ids_convers = pd.DataFrame({})
        raise Exception(
            'Residues in PDB file failed to be processed correctly by Rosetta.\n'
            'PDB res ids:\n{}\n\nRosetta res ids:\n{}\n\n'
            'Rosetta res energies:\n{}\n\n'.format(
                pdb_res_list, rosetta_res_list, res_energies_list
            )
        )

    # Extracts per-residue energy values
    for index, res in enumerate(res_ids_convers['PDB_res_ids'].tolist()):
        res_energy = res_ids_convers['Res_energies'][index]
        res_energies_dict[res] = res_energy

    return res_energies_dict


def score_pdb_rosetta(pdb_path, cwd, barrel_or_sandwich):
    """
    Relaxes structures and calculates their energy in the Rosetta force-field
    """

    pdb = pdb_path.split('/')[-1].replace('.pdb', '')
    nwd = '/'.join(pdb_path.split('/')[:-1])  # More complicated to use cwd
    # because PDB file is in its own directory
    nwd = '{}/{}_rosetta_results'.format(nwd, pdb)
    if not os.path.isdir(nwd):
        os.mkdir(nwd)
    os.chdir(nwd)  # Need to change directory so that when running RosettaMP output
    # spanfile is written here (unfortunately can't specify location with flag)

    # Relaxes a beta-sandwich structure and calculates the total energy of the
    # structure (in Rosetta Energy Units)
    unique_id = ''.join([random.choice(string.ascii_letters + string.digits)
                         for i in range(6)])  # Prevents interference of files
                         # when run in parallel
    if barrel_or_sandwich == '2.60':
        with open('{}/RosettaRelaxInputs{}'.format(cwd, unique_id), 'w') as f:
            f.write('-in:file:s {}\n'
                    '-out:path:pdb {}/\n'
                    '-out:file:scorefile {}/{}_score.sc\n'
                    '-relax:fast\n'
                    '-relax:constrain_relax_to_start_coords\n'
                    '-relax:ramp_constraints false'.format(
                        pdb_path, nwd, nwd, pdb
                    ))
        os.system(
            'relax.linuxgccrelease @{}/RosettaRelaxInputs{}'.format(cwd, unique_id)
        )
        os.remove('{}/RosettaRelaxInputs{}'.format(cwd, unique_id))

    # Relaxes a beta-barrel structure in the context of the membrane with
    # RosettaMP and calculates the total energy of the structure (in Rosetta
    # Energy Units)
    elif barrel_or_sandwich == '2.40':
        # N.B. INPUT STRUCTURE MUST BE ORIENTED SUCH THAT THE Z-AXIS IS ALIGNED
        # WITH THE MEMBRANE NORMAL (E.G. BY RUNNING THE STRUCTURE THROUGH THE OPM)

        # First generate spanfile if doesn't already exist (to avoid
        # interference between parallel processes)
        if not os.path.isfile('{}/{}.span'.format(nwd, pdb)):
            os.system('cp {} {}/{}.pdb'.format(pdb_path, nwd, pdb))
            os.system(
                'mp_span_from_pdb.linuxgccrelease -in:file:s {}/{}.pdb'.format(nwd, pdb)
            )
            os.system('rm {}/{}.pdb'.format(nwd, pdb))

        # Then relax structure with RosettaMP. Use mp_relax protocol updated for
        # ROSETTA3.
        with open('{}/RosettaMPRelaxInputs{}'.format(nwd, unique_id), 'w') as f:
            f.write('-parser:protocol /home/shared/rosetta/main/source/src/apps/public/membrane/mp_relax_updated.xml\n'
                    '-in:file:s {}\n'
                    '-nstruct 1\n'
                    '-mp:setup:spanfiles {}/{}.span\n'
                    '-mp:scoring:hbond\n'
                    '-relax:fast\n'
                    '-relax:jump_move true\n'
                    '-out:path:pdb {}/\n'
                    '-out:file:scorefile {}/{}_score.sc\n'
                    '-packing:pack_missing_sidechains 0'.format(
                        pdb_path, nwd, pdb, nwd, nwd, pdb
                    ))
        os.system('rosetta_scripts.linuxgccrelease @{}/RosettaMPRelaxInputs'
                  '{}'.format(nwd, unique_id))
        os.remove('{}/RosettaMPRelaxInputs{}'.format(nwd, unique_id))

    # N.B. Good total score value = < -2 x number of residues
    try:
        with open('{}/{}_score.sc'.format(nwd, pdb), 'r') as f:
            score_file_lines = f.readlines()
        total_energy = parse_rosetta_score_file(score_file_lines)
    except FileNotFoundError:
        total_energy = np.nan

    # Extracts per-residue energy values from the previously generated Rosetta
    # output files
    try:
        with open('{}/{}_0001.pdb'.format(nwd, pdb), 'r') as f:
            rosetta_lines = ('#BEGIN_POSE_ENERGIES_TABLE'
                             + f.read().split('#BEGIN_POSE_ENERGIES_TABLE')[1])
            rosetta_lines = [line for line in rosetta_lines.split('\n')
                             if line.strip() != '']
    except FileNotFoundError:
        rosetta_lines = np.nan
    with open(pdb_path, 'r') as f:
        pdb_lines = [line for line in f.readlines() if line.strip() != '']

    res_energies_dict = parse_rosetta_pdb_file(pdb_path, rosetta_lines, pdb_lines)

    return [pdb_path, total_energy, res_energies_dict]


if __name__ == '__main__':
    # Reads in command line inputs
    parser = argparse.ArgumentParser()
    parser.add_argument('-pdb_list', help='Absolute file path of pickled '
                        'structures output from running the GA')
    parser.add_argument('-bos', help='Specifies whether the structure is a '
                        'beta-barrel or -sandwich')
    parser.add_argument('-o', '--output', help='Location to which to save the '
                        'output pickled dictionary of Rosetta scores')
    args = parser.parse_args()

    pdb_list = vars(args)['pdb_list']
    with open(pdb_list, 'rb') as f:
        pdb_list = pickle.load(f)
    barrel_or_sandwich = vars(args)['bos']
    wd = vars(args)['output']

    struct_energies_dict = OrderedDict()
    res_energies_dict = OrderedDict()

    wd_list = [copy.deepcopy(wd) for n in range(len(pdb_list))]
    bos_list = [copy.deepcopy(barrel_or_sandwich) for n in range(len(pdb_list))]
    rosetta_scores_list = futures.map(
        score_pdb_rosetta, pdb_list, wd_list, bos_list
    )

    for tup in rosetta_scores_list:
        pdb_path = tup[0]
        total_energy = tup[1]
        res_energies_sub_dict = tup[2]
        struct_energies_dict[pdb_path] = total_energy
        res_energies_dict[pdb_path] = res_energies_sub_dict

    with open('{}/Rosetta_scores.pkl'.format(wd), 'wb') as f:
        pickle.dump((struct_energies_dict, res_energies_dict), f)
