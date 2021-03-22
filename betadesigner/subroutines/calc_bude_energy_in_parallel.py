
# Running a subset of a program in parallel with scoop / multiprocessing in
# Python requires the code in question to be separated into its own file and
# executed within an if __name__ == '__main__' clause

import argparse
import budeff
import copy
import isambard
import isambard.modelling as modelling
import pickle
from collections import OrderedDict
from scoop import futures


def pack_side_chains(ampal_object, G, rigid_rotamers):
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

    # Calculates total energy of the AMPAL object within BUDE (note that this
    # does not include the interaction of the object with its surrounding
    # environment, hence hydrophobic side chains will not be penalised on the
    # surface of a globular protein and vice versa for membrane proteins).
    # Hence this is just a rough measure of side-chain clashes.
    energy = budeff.get_internal_energy(new_ampal_object).total_energy

    return new_ampal_object, energy


def measure_fitness_allatom(pdb, network_num, G):
    """
    """

    # Packs network side chains onto the model with SCWRL4 and measures
    # the total model energy within BUDE
    new_pdb, energy = pack_side_chains(pdb, G, False)

    return [network_num, energy]


if __name__ == '__main__':
    # Reads in command line inputs
    parser = argparse.ArgumentParser()
    parser.add_argument('-pdb', help='Absolute file path to input PDB file')
    parser.add_argument('-net', help='Absolute file path to pickled dictionary '
                        'of sequence networks')
    parser.add_argument('-o', '--output', help='Location to which to save the '
                        'output pickled dictionary of BUDE scores')
    args = parser.parse_args()

    input_pdb = vars(args)['pdb']
    networks_dict_loc = vars(args)['net']
    with open(networks_dict_loc, 'rb') as f:
        networks_dict = pickle.load(f)
    wd = vars(args)['output']

    # Loads backbone model into ISAMBARD. NOTE must have been pre-processed
    # to remove ligands etc. so that only backbone coordinates remain.
    pdb = isambard.ampal.load_pdb(input_pdb)
    pdb_copies = [copy.deepcopy(pdb) for n in range(len(networks_dict))]

    network_energies_list = futures.map(
        measure_fitness_allatom, pdb_copies, list(networks_dict.keys()),
        list(networks_dict.values())
    )
    network_energies = OrderedDict()
    for pair in network_energies_list:
        network_energies[pair[0]] = pair[1]

    with open('{}/Network_energies.pkl'.format(wd), 'wb') as f:
        pickle.dump(network_energies, f)
