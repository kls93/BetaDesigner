
import isambard
from collections import OrderedDict

from find_parameters import initialise_class
from run_genetic_algorithm import pack_side_chains_scwrl

class gen_output():

    def __init__(self, parameters):
        gen_ga_input_pipeline.__init__(self, parameters)

    def write_pdb(self, sequences_dict):
        # Uses SCWRL4 to pack network side chains onto the backbone structure
        # and writes a PDB file of the output structure. Note that each network
        # is considered individually, hence only a single surface is replaced
        # at a time (and so in the case of a barrel for example if an exterior
        # face network were packed onto the structure, the interior face and
        # loops would remain the same as the original input structure)

        # Loads backbone model into ISAMBARD. NOTE must have been pre-processed
        # to remove ligands etc. so that only backbone coordinates remain.
        pdb = isambard.ampal.load_pdb(self.input_pdb)

        # Creates dictionary of residue ids and amino acid codes from AMPAL
        # object
        res_id_dict = OrderedDict()
        for res in pdb.get_monomers():
            fasta = res.mol_letter
            res_id = '{}{}{}'.format(res.parent.id, res.id, res.insertion_code)
            res_id_dict[res_id] = fasta

        for surface, networks_dict in sequences_dict.items():
            for num, G in networks_dict.items():
                # Packs network side chains onto the model with SCWRL4
                new_pdb, energy = pack_side_chains_scwrl(pdb, G, False)

                # Writes PDB file of model
                with open('Program_output/{}_{}.pdb'.format(surface, num), 'w') as f:
                    f.write(new_pdb.make_pdb())

                with open('Program_output/Model_energies', 'a') as f:
                    f.write('{}_{}: {}\n'.format(surface, num, energy))
