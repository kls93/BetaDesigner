
# Running a subset of a program in parallel with scoop / multiprocessing in
# Python requires the code in question to be separated into its own file and
# executed within an if __name__ == '__main__' clause

import argparse
import copy
import isambard
import isambard.modelling as modelling
import numpy as np
import os
import pickle
import shutil
from collections import OrderedDict
from scoop import futures


def measure_fitness_clashscore(ampal_object, network_num, G, work_dir, rigid_rotamers):
    """
    Uses SCWRL4 to pack network side chains onto a backbone structure and
    measures the total energy of the model within BUDE
    """

    # Makes FASTA sequence to feed into SCWRL4. BEWARE: if FASTA sequence is
    # shorter than AMPAL object, SCWRL4 will add random amino acids to the end
    # of the sequence until it is the same length.
    fasta_seq = ''
    for res in ampal_object.get_monomers():
        res_id = '{}{}{}{}'.format(
            res.parent.parent.id, res.parent.id, res.id, res.insertion_code
        )  # structure id, chain id, residue number, insertion code e.g. 4pnbD24

        if res_id in list(G.nodes):
            fasta_seq += G.nodes[res_id]['aa_id']
        else:
            fasta_seq += res.mol_letter  # Retains original ids of residues
            # outside of sequence to be mutated, e.g. in loop regions

    if len(fasta_seq) != len(list(ampal_object.get_monomers())):
        raise Exception('FASTA sequence and AMPAL object contain different '
                        'numbers of amino acids')

    # Packs side chains with SCWRL4. NOTE that fasta sequence must be provided
    # as a list. NOTE: Setting rigid_rotamers to True increases the speed of
    # side-chain but results in a concomitant decrease in accuracy.
    new_ampal_object = modelling.pack_side_chains_scwrl(
        ampal_object, [fasta_seq], rigid_rotamer_model=rigid_rotamers,
        hydrogens=False
    )

    os.mkdir('{}/{}_molprobity_output'.format(work_dir, network_num))
    with open(
        '{}/{}_molprobity_output/{}.pdb'.format(work_dir, network_num, network_num), 'w'
    ) as f:
        f.write(new_ampal_object.make_pdb(header=False, footer=False))

    # Calculates clash score of design
    os.system(
        'clashscore {}/{}_molprobity_output/{}.pdb > {}/{}_molprobity_output/'
        '{}_molprobity_clashscore.txt'.format(
            work_dir, network_num, network_num, work_dir, network_num, network_num
        )
    )
    clash = np.nan
    with open(
        '{}/{}_molprobity_output/{}_molprobity_clashscore.txt'.format(
            work_dir, network_num, network_num
        ), 'r'
    ) as f:
        file_lines = f.read().split('\n')
        for line in file_lines:
            if line[0:13] == 'clashscore = ':
                clash = float(line.replace('clashscore = ', ''))
                break
    shutil.rmtree('{}/{}_molprobity_output'.format(work_dir, network_num))

    if np.isnan(clash):
        raise Exception(
            'MolProbity failed to run for {}'.format(network_num)
        )

    return [network_num, clash]


if __name__ == '__main__':
    # Reads in command line inputs
    parser = argparse.ArgumentParser()
    parser.add_argument('-pdb', help='Absolute file path to input PDB file')
    parser.add_argument('-net', help='Absolute file path to pickled dictionary '
                        'of sequence networks')
    parser.add_argument('-o', '--output', help='Location to which to save the '
                        'output pickled dictionary of clashscores')
    args = parser.parse_args()

    input_pdb = vars(args)['pdb']
    networks_dict_loc = vars(args)['net']
    with open(networks_dict_loc, 'rb') as f:
        networks_dict = pickle.load(f)
    wd = vars(args)['output']
    wd_list = [copy.deepcopy(wd) for n in range(len(networks_dict))]

    rigid_rotamers_list = [False]*len(networks_dict)

    # Loads backbone model into ISAMBARD. NOTE must have been pre-processed
    # to remove ligands etc. so that only backbone coordinates remain.
    pdb = isambard.ampal.load_pdb(input_pdb)
    pdb_copies = [copy.deepcopy(pdb) for n in range(len(networks_dict))]

    network_clashes_list = futures.map(
        measure_fitness_clashscore, pdb_copies, list(networks_dict.keys()),
        list(networks_dict.values()), wd_list, rigid_rotamers_list
    )
    print(network_clashes_list)
    network_clashes = OrderedDict()
    for pair in network_clashes_list:
        network_clashes[pair[0]] = pair[1]

    with open('{}/Network_clashes.pkl'.format(wd), 'wb') as f:
        pickle.dump(network_clashes, f)
