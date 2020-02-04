
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

    def calc_rosetta_frag_coverage(
        self, structures_dict, tools_dir='/opt/rosetta/tools/fragment_tools'
    ):
        """
        make_fragments.pl is REALLY slow, forward-folding will be even slower
        => don't think either is a feasible scoring approach
        """

        # 2 metrics: % coverage and precision
        num_frag_dict = OrderedDict()
        frag_cov_dict = OrderedDict()

        for surface, pdb_path_list in structures_dict.items():
            num_frag_dict[surface] = OrderedDict()
            frag_cov_dict[surface] = OrderedDict()

            for pdb_path in pdb_path_list:
                # First generate ...
                pdb = pdb_path.split('/')[-1].replace('.pdb', '')
                wd = '/'.join(pdb_path.split('/')[:-1])  # More complicated to use
                # self.working_directory because PDB file is in its own directory
                wd = '{}/{}_rosetta_results'.format(wd, pdb)
                if not os.path.isdir(wd):
                    os.mkdir(wd)
                os.chdir(wd)

                fasta_path = pdb_path.replace('.pdb', '.fasta')
                # Only running psipred for secondary structure prediction, no
                # sam or jufo
                os.system('.{}/make_fragments.pl {} -nohoms'.format(
                    tools_dir, fasta_path
                ))

                """
                Uses BestFragmentsProtocol for speed and simplicity

                if not os.path.isfile('{}/simple.wghts'format(self.working_directory)):
                    with open('{}/simple.wghts'format(self.working_directory), 'w') as f:
                        f.write(
                            '# score name          priority  wght   max_allowed  extras\n'
                            'RamaScore               400     2.0     -       predA\n'
                            'SecondarySimilarity     350     1.0     -       predA\n'
                            'FragmentCrmsd             0     0.0     -\n'
                        )
                with open('{}/frag_pick_opt'.format(wd), 'w') as f:
                    f.write('-in:file:vall {}/vall.jul19.2011.gz\n'
                            '-in:file:fasta {}\n'
                            '-in:file:s {}\n'
                            '-frags:ss_pred {}/{}.psipred_ss2 predA\n'
                            '-frags:scoring:config {}/simple.wghts\n'
                            '-frags:bounded_protocol\n'
                            '-frags:frag_sizes 3 9\n'
                            '-frags:n_candidates 200\n'
                            '-frags:n_frags 200\n'
                            '-out:file:frag_prefix {}/{}_frags\n'
                            '-frags:describe_fragments {}/{}_frags.fsc\n'.format(
                                tools_dir, fasta_path, pdb_path, wd, pdb,
                                self.working_directory, wd, pdb, wd, pdb
                            ))
                os.system('fragment_picker.linuxgccrelease @{}/frag_pick_opt'.format(wd))
                os.remove('{}/frag_pick_opt'.format(wd))
                """

                # In spite of looking at multiple tutorials, I don't know how to
                # perform fragment picking and forward folding yet - need to ask
                # Fabio. Also ask Fabio about comparison of REU scores between
                # structures (incl. per-residue REU scores)

                # Make sure num_frag + frag_cov values are ints / floats rather
                # than strings!
                num_frag_dict[surface][pdb_path] = num_frag
                frag_cov_dict[surface][pdb_path] = frag_cov

        # os.remove('{}/simple.wghts'format(self.working_directory))  Uncomment if use BestFragmentsProtocol

        return num_frag_dict, frag_cov_dict

    def run_rosetta_forward_folding(self, structures_dict):
        """
        Does Fabio think running forward folding calculations is a good idea,
        or will they take too long?
        """

        ff_scores_dict = OrderedDict()
        for surface, pdb_path_list in structures_dict.items():
            ff_scores_dict[surface] = OrderedDict()

            for pdb_path in pdb_path_list:
                # Run forward folding calculations in Rosetta. See
                # https://www.rosettacommons.org/docs/latest/Biased-forward-folding
                # for tutorial.
                ff_scores_dict[surface][pdb_path] = ff_score

        return ff_scores_dict

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
