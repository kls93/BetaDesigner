
import copy
import isambard
import os
import pickle
import shutil
import numpy as np
import pandas as pd
from collections import OrderedDict

if __name__ == 'subroutines.write_output_structures':
    from subroutines.find_parameters import initialise_ga_object
    from subroutines.calc_bude_energy_in_parallel import pack_side_chains
else:
    from betadesigner.subroutines.find_parameters import initialise_ga_object
    from betadesigner.subroutines.calc_bude_energy_in_parallel import pack_side_chains


def parse_molprobity_struct_output(
    molp_stdout_struct, stdout_struct_header, per_struct_molp_dict, surface,
    pdb_path, struct_index
):
    """
    """

    cbeta_outliers_index = stdout_struct_header.index('cbeta>0.25')
    cbeta_tot_index = stdout_struct_header.index('numCbeta')
    rot_outliers_index = stdout_struct_header.index('rota<1%')
    rot_tot_index = stdout_struct_header.index('numRota')
    rama_favoured_index = stdout_struct_header.index('ramaFavored')
    rama_allowed_index = stdout_struct_header.index('ramaAllowed')
    rama_outlier_index = stdout_struct_header.index('ramaOutlier')
    rama_tot_index = stdout_struct_header.index('numRama')
    clashscore_index = stdout_struct_header.index('clashscore')
    clashscore_percentile_index = stdout_struct_header.index('pct_rank')

    try:
        cbeta_outliers = (float(molp_stdout_struct[cbeta_outliers_index])
                          / float(molp_stdout_struct[cbeta_tot_index]))
    except ValueError:
        cbeta_outliers = np.nan
    try:
        rot_outliers = (float(molp_stdout_struct[rot_outliers_index])
                        / float(molp_stdout_struct[rot_tot_index]))
    except ValueError:
        rot_outliers = np.nan
    try:
        rama_favoured = (float(molp_stdout_struct[rama_favoured_index])
                         / float(molp_stdout_struct[rama_tot_index]))
    except ValueError:
        rama_favoured = np.nan
    try:
        rama_allowed = (float(molp_stdout_struct[rama_allowed_index])
                        / float(molp_stdout_struct[rama_tot_index]))
    except ValueError:
        rama_allowed = np.nan
    try:
        rama_outlier = (float(molp_stdout_struct[rama_outlier_index])
                        / float(molp_stdout_struct[rama_tot_index]))
    except ValueError:
        rama_outlier = np.nan
    try:
        clashscore = float(molp_stdout_struct[clashscore_index])
    except ValueError:
        clashscore = np.nan
    try:
        clashscore_percentile = float(molp_stdout_struct[clashscore_percentile_index])
    except ValueError:
        clashscore_percentile = np.nan

    per_struct_molp_dict[surface]['Structure_id'][struct_index] = pdb_path
    per_struct_molp_dict[surface]['C_Beta_outliers'][struct_index] = cbeta_outliers
    per_struct_molp_dict[surface]['Rotamer_outliers'][struct_index] = rot_outliers
    per_struct_molp_dict[surface]['Ramachandran_favoured'][struct_index] = rama_favoured
    per_struct_molp_dict[surface]['Ramachandran_allowed'][struct_index] = rama_allowed
    per_struct_molp_dict[surface]['Ramachandran_outliers'][struct_index] = rama_outlier
    per_struct_molp_dict[surface]['Clashscore'][struct_index] = clashscore
    per_struct_molp_dict[surface]['Clashscore_percentile'][struct_index] = clashscore_percentile

    return per_struct_molp_dict


def parse_molprobity_res_output(
    molp_stdout_res, stdout_res_header, per_res_molp_dict, surface, pdb_path
):
    """
    """

    per_res_molp_dict[surface][pdb_path] = OrderedDict({
        'Residue_id': ['']*len(molp_stdout_res),
        'C_Beta_deviation': ['']*len(molp_stdout_res),
        'Rotamer_score': ['']*len(molp_stdout_res),
        'Rotamer_allowed': ['']*len(molp_stdout_res),
        'Ramachandran_score': ['']*len(molp_stdout_res),
        'Ramachandran_allowed': ['']*len(molp_stdout_res),
        'Worst_clash': ['']*len(molp_stdout_res)
    })

    res_id_index = stdout_res_header.index('residue')
    cbeta_dev_index = stdout_res_header.index('CB_dev')
    rota_score_index = stdout_res_header.index('rotamer_score')
    rota_eval_index = stdout_res_header.index('rotamer_eval')
    rama_score_index = stdout_res_header.index('rama_score')
    rama_eval_index = stdout_res_header.index('rama_eval')
    worst_clash_index = stdout_res_header.index('worst_clash')

    res_index = 0
    for res in molp_stdout_res:
        res_data = res.split(',')
        res_id = res_data[res_id_index].strip().split()
        res_id = '{}_{}_{}_{}'.format(pdb_path, res_id[2], res_id[0], res_id[1])
        try:
            cbeta_dev = float(res_data[cbeta_dev_index])
        except ValueError:
            cbeta_dev = np.nan
        try:
            rota_score = float(res_data[rota_score_index])
        except ValueError:
            rota_score = np.nan
        rota_eval = res_data[rota_eval_index]
        try:
            rama_score = float(res_data[rama_score_index])
        except ValueError:
            rama_score = np.nan
        rama_eval = res_data[rama_eval_index]
        try:
            worst_clash = float(res_data[worst_clash_index])
        except ValueError:
            worst_clash = np.nan

        per_res_molp_dict[surface][pdb_path]['Residue_id'][res_index] = res_id
        per_res_molp_dict[surface][pdb_path]['C_Beta_deviation'][res_index] = cbeta_dev
        per_res_molp_dict[surface][pdb_path]['Rotamer_score'][res_index] = rota_score
        per_res_molp_dict[surface][pdb_path]['Rotamer_allowed'][res_index] = rota_eval
        per_res_molp_dict[surface][pdb_path]['Ramachandran_score'][res_index] = rama_score
        per_res_molp_dict[surface][pdb_path]['Ramachandran_allowed'][res_index] = rama_eval
        per_res_molp_dict[surface][pdb_path]['Worst_clash'][res_index] = worst_clash

        res_index +=1

    per_res_molp_dict[surface][pdb_path] = pd.DataFrame(per_res_molp_dict[surface][pdb_path])

    return per_res_molp_dict

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

        structures_dict = OrderedDict()
        bude_energies_dict = OrderedDict()

        # Loads backbone model into ISAMBARD. NOTE must have been pre-processed
        # to remove ligands etc. so that only backbone coordinates remain.
        pdb = isambard.ampal.load_pdb(self.input_pdb)

        for surface, networks_dict in sequences_dict.items():
            print('Packing side chains for {} surface'.format(surface))
            structures_dict[surface] = []
            bude_energies_dict[surface] = OrderedDict()

            for num, G in networks_dict.items():
                os.mkdir('{}/Program_output/{}_{}/'.format(
                    self.working_directory, surface, num
                ))
                structure_name = '{}/Program_output/{}_{}/{}_{}.pdb'.format(
                    self.working_directory, surface, num, surface, num
                )

                # Packs network side chains onto the model with SCWRL4 and
                # calculates model energies in BUDE
                new_pdb, energy = pack_side_chains(pdb, G, False)
                with open('{}/Program_output/Model_energies.txt'.format(
                    self.working_directory), 'a') as f:
                    f.write('{}_{}: {}\n'.format(surface, num, energy))
                bude_energies_dict[surface][structure_name] = energy

                # Writes PDB file of model. N.B. Currently code only designs
                # sequences containing the 20 canonical amino acids, but have
                # included HETATM anyway to avoid bugs in future iterations of
                # the code
                with open(structure_name, 'w') as f:
                    for line in new_pdb.make_pdb().split('\n'):
                        if line[0:6].strip() in ['ATOM', 'HETATM', 'TER', 'END']:
                            f.write('{}\n'.format(line))
                structures_dict[surface].append(structure_name)
                # Writes FASTA sequence of model
                fasta_seq = ''
                for res_id in list(G.nodes):
                    fasta_seq += G.nodes[res_id]['aa_id']
                fasta_name = structure_name.replace('.pdb', '.fasta')
                with open(fasta_name, 'w') as f:
                    f.write('>{}_{}\n'.format(surface, num))
                    f.write('{}\n'.format(fasta_seq))

        return structures_dict, bude_energies_dict

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
                pdb = pdb_path.split('/')[-1].strip('.pdb')
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

        per_struct_molp_dict = OrderedDict()
        per_res_molp_dict = OrderedDict()

        for surface, pdb_path_list in structures_dict.items():
            per_struct_molp_dict[surface] = OrderedDict({
                'Structure_id': ['']*len(pdb_path_list),
                'C_Beta_outliers': ['']*len(pdb_path_list),
                'Rotamer_outliers': ['']*len(pdb_path_list),
                'Ramachandran_favoured': ['']*len(pdb_path_list),
                'Ramachandran_allowed': ['']*len(pdb_path_list),
                'Ramachandran_outliers': ['']*len(pdb_path_list),
                'Clashscore': ['']*len(pdb_path_list),
                'Clashscore_percentile': ['']*len(pdb_path_list)
            })
            per_res_molp_dict[surface] = OrderedDict()

            struct_index = 0
            for pdb_path in pdb_path_list:
                pdb = pdb_path.split('/')[-1]
                wd = '/'.join(pdb_path.split('/')[:-1])  # More complicated to use
                # self.working_directory because PDB file is in its own directory

                # Structure-wide metrics. Note oneline-analysis is provided with
                # an input directory rather than an input pdb file, hence the
                # input pdb file is temporarily copied to a separate directory
                os.mkdir('{}/temporary/'.format(wd))
                shutil.copy(pdb_path, '{}/temporary/'.format(wd))
                os.system(
                    'oneline-analysis {}/temporary/ > {}/{}_struct_molprobity_'
                    'stdout.txt'.format(wd, wd, pdb)
                )
                shutil.rmtree('{}/temporary/'.format(wd))  # Deletes files
                # created whilst running MolProbity

                molp_stdout_struct = []
                stdout_struct_header = []
                with open('{}/{}_struct_molprobity_stdout.txt'.format(wd, pdb), 'r') as f:
                    for index, line in enumerate(f.readlines()):
                        if line.startswith(pdb):
                            molp_stdout_struct = line.split(':')
                        elif line.startswith('#pdbFileName'):
                            stdout_struct_header = line.split(':')
                if molp_stdout_struct == [] or stdout_struct_header == []:
                    raise Exception(
                        'Unexpected formatting of Molprobity standard output'
                    )

                per_struct_molp_dict = parse_molprobity_struct_output(
                    molp_stdout_struct, stdout_struct_header,
                    per_struct_molp_dict, surface, pdb_path, struct_index
                )
                struct_index += 1

                # Per-residue metrics
                os.system(
                    'residue-analysis {} > {}/{}_res_molprobity_stdout'
                    '.txt'.format(pdb_path, wd, pdb)
                )
                shutil.rmtree('{}/molprobity_results/'.format(wd))  # Deletes
                # files created by running MolProbity

                molp_stdout_res = []
                stdout_res_header = []
                with open('{}/{}_res_molprobity_stdout.txt'.format(wd, pdb), 'r') as f:
                    for line in f.readlines():
                        if line.startswith('#file_name'):
                            stdout_res_header = line.split(',')
                        elif line.startswith(pdb):
                            molp_stdout_res.append(line)
                if molp_stdout_res == [] or stdout_res_header == []:
                    raise Exception(
                        'Unexpected formatting of Molprobity standard output'
                    )

                per_res_molp_dict = parse_molprobity_res_output(
                    molp_stdout_res, stdout_res_header, per_res_molp_dict,
                    surface, pdb_path
                )

            per_struct_molp_dict[surface] = pd.DataFrame(per_struct_molp_dict[surface])

        return per_struct_molp_dict, per_res_molp_dict
