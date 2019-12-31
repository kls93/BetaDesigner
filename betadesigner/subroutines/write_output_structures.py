
import isambard
import os
import shutil
import pandas as pd
from collections import OrderedDict

if __name__ == 'subroutines.write_output_structures':
    from subroutines.find_parameters import initialise_ga_object
    from subroutines.run_genetic_algorithm import pack_side_chains
else:
    from betadesigner.subroutines.find_parameters import initialise_ga_object
    from betadesigner.subroutines.run_genetic_algorithm import pack_side_chains


"""
In the case of OmpA, put the structure in a membrane with MemProtRosetta
First energy minimise
Then calculate total energy and per-atom energy (can also split
this into van der Waals and solvation components etc.) - can then
colour residue by energy in VR, Might also want to return RMSD
of relaxed to original structure
"""


def parse_rosetta_output(
    res_energies_dict, surface, pdb_path, rosetta_lines, pdb_lines
):
    """
    """

    res_energies_dict[surface][pdb_path] = OrderedDict()

    rosetta_res = []
    pdb_res = []

    # Matches renumbered Rosetta res ids to res ids in the input PDB file
    for line in rosetta_lines:
        if (
                line[0:6].strip() in ['ATOM', 'HETATM']
            and not line[17:20] == 'MEM'
            and not '{}_{}'.format(line[17:20], line[22:26].strip()) in rosetta_res
        ):
            rosetta_res.append('{}_{}'.format(line[17:20], line[22:26].strip()))
    rosetta_res[0] = (rosetta_res[0].split('_')[0] + ':NtermProteinFull_'
                      + rosetta_res[0].split('_')[1])
    rosetta_res[-1] = (rosetta_res[-1].split('_')[0] + ':CtermProteinFull_'
                       + rosetta_res[-1].split('_')[1])

    for line in pdb_lines:
        if (
                line[0:6].strip() in ['ATOM', 'HETATM']
            and not '{}_{}_{}'.format(line[17:20], line[21:22], line[22:26].strip()) in pdb_res
        ):
            pdb_res.append('{}_{}_{}'.format(line[17:20], line[21:22], line[22:26].strip()))

    if len(rosetta_res) == len(pdb_res)
        rosetta_pdb_convers = OrderedDict(zip(rosetta_res, pdb_res))
    else:
        rosetta_pdb_convers = OrderedDict()
        raise Exception(
            'Residues in PDB file failed to be processed '
            'correctly by Rosetta.\nPDB res ids:\n{}\n\nRosetta'
            ' res ids:\n{}\n\n'.format(pdb_res, rosetta_res)
        )

    # Extracts per-residue energy values
    start = False
    res_energy_index = ''
    for index, line in enumerate(rosetta_lines):
        if line.startswith('#BEGIN_POSE_ENERGIES_TABLE'):
            start = True
            res_energy_index = rosetta_lines[index+1].index('total')
        if start is True:
            line = line.split()
            if line[0] in ['label', 'weights', 'pose', '#END_POSE_ENERGIES_TABLE']:
                continue

            try:
                res = rosetta_pdb_convers[line[0]]
            except KeyError:
                raise KeyError(
                    'Residue {} failed to be parsed from {}/Program_output/'
                    '{}_rosetta_results/{}_0001.pdb'.format(
                        line[0], self.working_directory, pdb, pdb
                    ))
            # Must make this an if statement if no longer raise KeyError above
            res_energy = line[res_energy_index]
            res_energies_dict[surface][pdb_path][res] = res_energy

    return res_energy

class gen_output(initialise_ga_object):

    def __init__(self, params, best_bayes_params, test=False):
        initialise_ga_object.__init__(self, params, test)
        self.unfit_fraction = best_bayes_params['unfitfraction']
        self.crossover_prob = best_bayes_params['crossoverprob']
        self.mutation_prob = best_bayes_params['mutationprob']
        self.propensity_weight = best_bayes_params['propvsfreqweight']

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

        return structures_dict, bude_energies_dict

    def score_pdb_rosetta(self, structures_dict):
        """
        Relaxes structures and calculates their energy in the Rosetta force-field
        """

        struct_energies_dict = OrderedDict()
        res_energies_dict = OrderedDict()

        for surface in structures_dict.keys():
            struct_energies_dict[surface] = OrderedDict()
            res_energies_dict[surface] = OrderedDict()

            for pdb_path in structures_dict[surface]:
                pdb = pdb_path.split('/')[-1].strip('.pdb')
                wd = '/'.join(pdb_path.split('/')[:-1])  # More complicated to use
                # self.working_directory because PDB file is in its own directory
                wd = '{}/{}_rosetta_results'.format(wd, pdb)
                os.mkdir(wd)
                os.chdir(wd)

                # Relaxes a beta-sandwich structure and calculates the total
                # energy of the structure (in Rosetta Energy Units)
                if self.barrel_or_sandwich == '2.60':
                    with open('RosettaRelaxInputs', 'w') as f:
                        f.write('-in:file:s {}\n'
                                '-out:path:pdb {}/\n'
                                '-out:path:score {}/{}_score.sc\n'
                                '-relax:fast\n'
                                '-relax:constrain_relax_to_start_coords\n'
                                '-relax:ramp_constraints false'.format(
                                    pdb_path, wd, wd, pdb
                                ))
                    os.system('relax.linuxgccrelease @RosettaRelaxInputs')
                    os.remove('RosettaRelaxInputs')

                # Relaxes a beta-barrel structure in the context of the
                # membrane with RosettaMP and calculates the total energy of
                # the structure (in Rosetta Energy Units)
                elif self.barrel_or_sandwich == '2.40':
                    # N.B. INPUT STRUCTURE MUST BE ORIENTED SUCH THAT THE
                    # Z-AXIS IS ALIGNED WITH THE MEMBRANE NORMAL (E.G. BY
                    # RUNNING THE STRUCTURE THROUGH THE OPM)

                    # First generate spanfile
                    os.system('mp_span_from_pdb.linuxgccrelease -in:file:s '
                              '{}'.format(pdb_path))

                    # Then relax structure with RosettaMP. Use mp_relax
                    # protocol updated for ROSETTA3.
                    with open('RosettaMPRelaxInputs', 'w') as f:
                        f.write('-parser:protocol $ROSETTA3/src/apps/public/membrane/mp_relax_updated.xml\n'
                                '-in:file:s {}\n'
                                '-nstruct 1\n'
                                '-mp:setup:spanfiles {}/{}.span\n'
                                '-mp:scoring:hbond\n'
                                '-relax:fast\n'
                                '-relax:jump_move true\n'
                                '-out:pdb {}/\n'
                                '-out:file:scorefile {}/{}_score.sc\n'
                                '-packing:pack_missing_sidechains 0'.format(
                                    pdb_path, wd, pdb, wd, wd, pdb
                                ))
                    os.system('rosetta_scripts.linuxgccrelease @RosettaMPRelaxInputs')
                    os.remove('RosettaMPRelaxInputs')

                # N.B. Good total score value = < -2 x number of residues
                with open('{}/{}_score.sc'.format(wd, pdb), 'r') as f:
                    lines = f.readlines()
                total_energy_index = lines[1].split().index('total_score')
                total_energy = lines[2].split()[total_energy_index]
                struct_energies_dict[surface][pdb_path] = total_energy


                # Extracts per-residue energy values from the previously
                # generated Rosetta output files
                with open('{}/{}_0001.pdb'.format(wd, pdb), 'r') as f:
                    rosetta_lines = f.readlines()
                with open(pdb_path, 'r') as f:
                    pdb_lines = f.readlines()

                res_energies_dict = parse_rosetta_output(
                    res_energies_dict, surface, pdb_path, rosetta_lines, pdb_lines
                )

        return struct_energies_dict, res_energies_dict

    def calc_rosetta_frag_coverage(self, structures_dict):
        """
        """

        # 2 metrics: % coverage and precision
        frag_cov_dict = OrderedDict()
        for surface, pdb_path_list in structures_dict.items():
            frag_cov_dict[surface] = OrderedDict()

            for pdb_path in pdb_path_list:
                frag_cov_dict[surface][pdb_path] = frag_cov

        return

    def score_pdb_molprobity(self, structures_dict):
        """
        """

        # For full structure analysis, include % Cbeta outliers, % rotamer
        # outliers, % Ramachandran favoured, allowed and outlier, clashscore
        # and clashscore percentile value
        # For per-residue analysis, include Cbeta_dev, rama_score, rama_eval,
        # rotamer_score, rotamer_eval and worst_clash
        # N.B.: NEED TO FIX SCRIPTS SO THAT OUTPUT IS WRITTEN TO CWD
        # RATHER THAN ROOT - THIS WILL AVOID THE NEED FOR SUDO ACCESS

        cwd = os.getcwd()

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
                shutil.copy('{} {}/temporary/'.format(pdb_path, wd))
                os.system(
                    'oneline-analysis {}/temporary/ > {}/molprobity_struct_stdout.txt'.format(wd, wd)
                )
                shutil.rmtree('{}/temporary/'.format(wd))  # Deletes files
                # created by running MolProbity

                molp_stdout_struct = []
                with open('{}/molprobity_struct_stdout.txt'.format(wd), 'r') as f:
                    for index, line in enumerate(f.readlines()):
                        if line.startswith(pdb):
                            molp_stdout_struct = line.split(':')
                if molp_stdout_struct == []:
                    raise Exception(
                        'Unexpected formatting of Molprobity standard output'
                    )

                cbeta_outliers = molp_stdout_struct[15] / molp_stdout_struct[16]
                rot_outliers = molp_stdout_struct[17] / molp_stdout_struct[18]
                rama_favoured = molp_stdout_struct[21] / molp_stdout_struct[22]
                rama_allowed = molp_stdout_struct[20] / molp_stdout_struct[22]
                rama_outlier = molp_stdout_struct[19] / molp_stdout_struct[22]
                clashscore = molp_stdout_struct[8]
                clashscore_percentile = molp_stdout_struct[13]

                per_struct_molp_dict[surface]['Structure_id'][struct_index] = pdb_path
                per_struct_molp_dict[surface]['C_Beta_outliers'][struct_index] = cbeta_outliers
                per_struct_molp_dict[surface]['Rotamer_outliers'][struct_index] = rot_outliers
                per_struct_molp_dict[surface]['Ramachandran_favoured'][struct_index] = rama_favoured
                per_struct_molp_dict[surface]['Ramachandran_allowed'][struct_index] = rama_allowed
                per_struct_molp_dict[surface]['Ramachandran_outliers'][struct_index] = rama_outlier
                per_struct_molp_dict[surface]['Clashscore'][struct_index] = clashscore
                per_struct_molp_dict[surface]['Clashscore_percentile'][struct_index] = clashscore_percentile

                struct_index += 1

                # Per-residue metrics
                os.system(
                    'residue-analysis {} > {}/molprobity_res_stdout.txt'.format(pdb_path, wd)
                )
                shutil.rmtree('{}/molprobity_results/'.format(wd))  # Deletes
                # files created by running MolProbity

                molp_stdout_res = []
                with open('{}/molprobity_res_stdout.txt'.format(wd), 'r') as f:
                    molp_stdout_res = [line for line in f.readlines()
                                       if line.startswith(pdb)]
                if molp_stdout_res == []:
                    raise Exception(
                        'Unexpected formatting of Molprobity standard output'
                    )

                per_res_molp_dict[surface][pdb_path] = OrderedDict({
                    'Residue_id': ['']*len(molp_stdout_res),
                    'C_Beta_deviation': ['']*len(molp_stdout_res),
                    'Rotamer_score': ['']*len(molp_stdout_res),
                    'Rotamer_allowed': ['']*len(molp_stdout_res),
                    'Ramachandran_score': ['']*len(molp_stdout_res),
                    'Ramachandran_allowed': ['']*len(molp_stdout_res),
                    'Worst_clash': ['']*len(molp_stdout_res)
                })

                res_index = 0
                for res in molp_stdout_res:
                    res_data = res.split(',')

                    res_id = '{}_{}'.format(pdb_path, res_data[2])
                    cbeta_dev = res_data[9]
                    rota_score = res_data[10]
                    rota_eval = res_data[11]
                    rama_score = res_data[13]
                    rama_eval = res_data[14]
                    worst_clash = res_data[5]

                    per_res_molp_dict[surface][pdb_path]['Residue_id'][res_index] = res_id
                    per_res_molp_dict[surface][pdb_path]['C_Beta_deviation'][res_index] = cbeta_dev
                    per_res_molp_dict[surface][pdb_path]['Rotamer_score'][res_index] = rota_score
                    per_res_molp_dict[surface][pdb_path]['Rotamer_allowed'][res_index] = rota_eval
                    per_res_molp_dict[surface][pdb_path]['Ramachandran_score'][res_index] = rama_score
                    per_res_molp_dict[surface][pdb_path]['Ramachandran_allowed'][res_index] = rama_eval
                    per_res_molp_dict[surface][pdb_path]['Worst_clash'][res_index] = worst_clash

                    res_index +=1

                per_res_molp_dict[surface][pdb_path] = pd.DataFrame(per_res_molp_dict[surface][pdb_path])
            per_struct_molp_dict[surface] = pd.DataFrame(per_struct_molp_dict[surface])

        return per_struct_molp_dict, per_res_molp_dict
