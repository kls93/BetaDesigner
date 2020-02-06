
# Running a subset of a program in parallel with scoop / multiprocessing in
# Python requires the code in question to be separated into its own file and
# executed within an if __name__ == '__main__' clause

import argparse
import budeff
import copy
import isambard
import os
import pickle
from collections import OrderedDict
from scoop import futures
from betadesigner.subroutines.calc_bude_energy_in_parallel import pack_side_chains

def write_pdb(num, G, wd, surface, ampal_pdb, pdb):
    """
    Uses SCWRL4 to pack network side chains onto the backbone structure and
    writes a PDB file of the output structure. Note that each network is
    considered individually, hence only a single surface is replaced at a time
    (and so in the case of a barrel for example if an exterior face network
    were packed onto the structure, the interior face and loops would remain
    the same as the original input structure)
    """

    os.mkdir('{}/Program_output/{}_{}/'.format(wd, surface, num))
    struct_name = '{}/Program_output/{}_{}/{}_{}.pdb'.format(
        wd, surface, num, surface, num
    )

    # Packs network side chains onto the model with SCWRL4 and calculates model
    # energies in BUDE
    new_pdb, new_energy = pack_side_chains(ampal_pdb, G, False)
    # Calculates total energy of input PDB structure within BUDE (note that this
    # does not include the interaction of the object with its surrounding
    # environment, hence hydrophobic side chains will not be penalised on the
    # surface of a globular protein and vice versa for membrane proteins).
    # Hence this is just a rough measure of side-chain clashes.
    orig_ampal_pdb = isambard.ampal.load_pdb(pdb)
    orig_energy = budeff.get_internal_energy(orig_ampal_pdb).total_energy

    # Writes PDB file of model. N.B. Currently code only designs
    # sequences containing the 20 canonical amino acids, but have
    # included HETATM anyway to avoid bugs in future iterations of
    # the code
    with open(struct_name, 'w') as f:
        for line in new_pdb.make_pdb().split('\n'):
            if line[0:6].strip() in ['ATOM', 'HETATM', 'TER', 'END']:
                f.write('{}\n'.format(line))

    # Writes FASTA sequence of model
    fasta_seq = ''
    for res_id in list(G.nodes):
        fasta_seq += G.nodes[res_id]['aa_id']
    fasta_name = struct_name.replace('.pdb', '.fasta')
    with open(fasta_name, 'w') as f:
        f.write('>{}_{}\n'.format(surface, num))
        f.write('{}\n'.format(fasta_seq))

    return [struct_name, num, G, new_energy, orig_energy]


if __name__ == '__main__':
    # Reads in command line inputs
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--seq_dict', help='Absolute file path of '
                        'pickled structures output from running the GA')
    parser.add_argument('-pdb', help='Absolute file path of input PDB file')
    parser.add_argument('-o', '--output', help='Location to which to save the '
                        'output pickled dictionary of Rosetta scores')
    args = parser.parse_args()

    sequences_dict = vars(args)['seq_dict']
    with open(sequences_dict, 'rb') as f:
        sequences_dict = pickle.load(f)
    pdb = vars(args)['pdb']
    wd = vars(args)['output']

    updated_sequences_dict = OrderedDict()
    structures_dict = OrderedDict()
    bude_energies_dict = OrderedDict()

    # Loads backbone model into ISAMBARD. NOTE must have been pre-processed
    # to remove ligands etc. so that only backbone coordinates remain.
    ampal_pdb = isambard.ampal.load_pdb(pdb)

    for surface, networks_dict in sequences_dict.items():
        print('Packing side chains for {} surface'.format(surface))

        updated_sequences_dict[surface] = OrderedDict()
        structures_dict[surface] = []
        bude_energies_dict[surface] = OrderedDict()

        wd_list = [copy.deepcopy(wd) for n in range(len(networks_dict))]
        surface_list = [copy.deepcopy(surface) for n in range(len(networks_dict))]
        ampal_pdb_list = [copy.deepcopy(ampal_pdb) for n in range(len(networks_dict))]
        pdb_list = [copy.deepcopy(pdb) for n in range(len(networks_dict))]

        structures_list = futures.map(
            write_pdb, list(networks_dict.keys()), list(networks_dict.values()),
            wd_list, surface_list, ampal_pdb_list, pdb_list
        )

        for index, tup in enumerate(structures_list):
            struct_name = tup[0]
            num = tup[1]
            network = tup[2]
            new_struct_energy = tup[3]
            orig_struct_energy = tup[4]

            bude_energies_dict[surface][struct_name] = new_struct_energy
            structures_dict[surface].append(struct_name)
            updated_sequences_dict[surface][struct_name] = network

            if index == 0:
                with open('{}/Program_output/Model_energies.txt'.format(wd), 'a') as f:
                    f.write('\nInput structure: {}\n\n\n'.format(orig_energy))
            with open('{}/Program_output/Model_energies.txt'.format(wd), 'a') as f:
                f.write('{}_{}: {}\n'.format(surface, num, new_struct_energy))

    with open('{}/BUDE_energies.pkl'.format(wd), 'wb') as f:
        pickle.dump((updated_sequences_dict, structures_dict, bude_energies_dict), f)
