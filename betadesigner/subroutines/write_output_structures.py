
import os
import pickle
from collections import OrderedDict

if __name__ == 'subroutines.write_output_structures':
    from subroutines.find_parameters import initialise_ga_object
    from subroutines.calc_bude_energy_in_parallel import pack_side_chains
else:
    from betadesigner.subroutines.find_parameters import initialise_ga_object
    from betadesigner.subroutines.calc_bude_energy_in_parallel import pack_side_chains


class gen_output(initialise_ga_object):

    def __init__(self, params, best_bayes_params, test=False):
        initialise_ga_object.__init__(self, params, test)
        self.unfit_fraction = best_bayes_params['unfitfraction']
        self.crossover_prob = best_bayes_params['crossoverprob']
        self.mutation_prob = best_bayes_params['mutationprob']
        self.propensity_weight = best_bayes_params['propensityweight']

    def write_pdb(self, sequences_dict):
        """
        Uses SCWRL4 to pack network side chains onto the backbone structure
        and writes a PDB file of the output structure. Note that each network
        is considered individually, hence only a single surface is replaced
        at a time (and so in the case of a barrel for example if an exterior
        face network were packed onto the structure, the interior face and
        loops would remain the same as the original input structure)
        """

        print('Writing output structures and scoring with BUDE')

        with open('{}/Sequences_dict.pkl'.format(self.working_directory), 'wb') as f:
            pickle.dump(sequences_dict, f)

        os.system('python -m scoop {}/write_output_pdb_in_parallel.py '
                  '-s {}/Sequences_dict.pkl -pdb {} -o {}'.format(
                  os.path.dirname(os.path.abspath(__file__)),
                  self.working_directory, self.input_pdb, self.working_directory))

        os.remove('{}/Sequences_dict.pkl'.format(self.working_directory))
        with open('{}/BUDE_energies.pkl'.format(self.working_directory), 'rb') as f:
            updated_sequences_dict, structures_dict, bude_energies_dict = pickle.load(f)
        os.remove('{}/BUDE_energies.pkl'.format(self.working_directory))

        return updated_sequences_dict, structures_dict, bude_energies_dict

    def score_pdb_rosetta(self, structures_dict):
        """
        Relaxes structures and calculates their energy in the Rosetta force-field
        """

        print('Scoring structures with ROSETTA')

        with open('{}/Structures_dict.pkl'.format(self.working_directory), 'wb') as f:
            pickle.dump(structures_dict, f)

        os.system('python -m scoop {}/calc_rosetta_score_in_parallel.py '
                  '-s {}/Structures_dict.pkl -bos {} -o {}'.format(
                  os.path.dirname(os.path.abspath(__file__)),
                  self.working_directory, self.barrel_or_sandwich,
                  self.working_directory))

        os.remove('{}/Structures_dict.pkl'.format(self.working_directory))
        with open('{}/Rosetta_scores.pkl'.format(self.working_directory), 'rb') as f:
            struct_energies_dict, res_energies_dict = pickle.load(f)
        os.remove('{}/Rosetta_scores.pkl'.format(self.working_directory))

        return struct_energies_dict, res_energies_dict

    def calc_rosetta_frag_coverage(self, structures_dict):
        """
        Runs basic fragment generation with make_fragments.pl in Rosetta
        """

        print('Generating fragments with make_fragments.pl in ROSETTA')

        with open('{}/Structures_dict.pkl'.format(self.working_directory), 'wb') as f:
            pickle.dump(structures_dict, f)

        os.system('python -m scoop {}/calc_rosetta_frag_coverage_in_parallel.py '
                  '-s {}/Structures_dict.pkl -o {}'.format(
                  os.path.dirname(os.path.abspath(__file__)),
                  self.working_directory, self.working_directory))

        os.remove('{}/Structures_dict.pkl'.format(self.working_directory))
        with open('{}/Rosetta_frag_coverage.pkl'.format(self.working_directory), 'rb') as f:
            worst_best_frag_dict, num_frag_dict, frag_cov_dict = pickle.load(f)
        os.remove('{}/Rosetta_frag_coverage.pkl'.format(self.working_directory))

        return worst_best_frag_dict, num_frag_dict, frag_cov_dict

    def score_pdb_molprobity(self, structures_dict):
        """
        """

        print('Calculating MolProbity scores')

        with open('{}/Structures_dict.pkl'.format(self.working_directory), 'wb') as f:
            pickle.dump(structures_dict, f)

        os.system('python -m scoop {}/calc_molprobity_score_in_parallel.py '
                  '-s {}/Structures_dict.pkl -o {}'.format(
                  os.path.dirname(os.path.abspath(__file__)),
                  self.working_directory, self.working_directory))

        os.remove('{}/Structures_dict.pkl'.format(self.working_directory))
        with open('{}/MolProbity_scores.pkl'.format(self.working_directory), 'rb') as f:
            per_struct_molp_dict, per_res_molp_dict = pickle.load(f)
        os.remove('{}/MolProbity_scores.pkl'.format(self.working_directory))

        return per_struct_molp_dict, per_res_molp_dict
