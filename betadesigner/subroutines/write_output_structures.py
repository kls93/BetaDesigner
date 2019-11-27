
import isambard
from collections import OrderedDict

if __name__ == 'subroutines.write_output_structures':
    from subroutines.find_parameters import initialise_ga_object
    from subroutines.run_genetic_algorithm import pack_side_chains
else:
    from betadesigner.subroutines.find_parameters import initialise_ga_object
    from betadesigner.subroutines.run_genetic_algorithm import pack_side_chains

class gen_output(initialise_ga_object):

    def __init__(self, params):
        initialise_ga_object.__init__(self, params)

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

        for surface, networks_dict in sequences_dict.items():
            print('Packing side chains for {} surface'.format(surface))

            for num, G in networks_dict.items():
                # Packs network side chains onto the model with SCWRL4
                new_pdb, energy = pack_side_chains(pdb, G, False)

                # Writes PDB file of model
                with open('Program_output/{}_{}.pdb'.format(surface, num), 'w') as f:
                    f.write(new_pdb.make_pdb())

                with open('Program_output/Model_energies.txt', 'a') as f:
                    f.write('{}_{}: {}\n'.format(surface, num, energy))
