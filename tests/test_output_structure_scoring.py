
# python -m unittest tests/test_output_structure_scoring.py
# Tests code for scoring output structures with BUDE, PyRosetta and MolProbity

import os
import unittest
import numpy as np
import pandas as pd
from collections import OrderedDict
from betadesigner.subroutines.calc_rosetta_score_in_parallel import (
    parse_rosetta_score_file, parse_rosetta_pdb_file
)
from betadesigner.subroutines.calc_molprobity_score_in_parallel import (
    parse_molprobity_struct_output, parse_molprobity_res_output
)


class test_output_structure_scoring(unittest.TestCase):
    """
    """

    def test_parse_rosetta_output(self):
        """
        Tests that output file from running Rosetta(MP) relax is being parsed
        correctly
        """

        # Tests parsing of score.sc file to extract score of entire structure
        for pdb_dir in os.listdir('tests/test_files/'):
            if pdb_dir.endswith('_output_test'):
                pdb = pdb_dir.split('_')[0]
                with open('tests/test_files/{}/{}_score.sc'.format(pdb_dir, pdb)) as f:
                    score_file_lines = f.readlines()

                exp_tot_energy = float(score_file_lines[2].split()[1])
                meas_tot_energy = parse_rosetta_score_file(score_file_lines)
                self.assertEqual(exp_tot_energy, meas_tot_energy)

        # Tests parsing of PDB file to extract individual residue scores
        for pdb_dir in os.listdir('tests/test_files/'):
            if pdb_dir.endswith('_output_test'):
                pdb = pdb_dir.split('_')[0]
                rosetta_path = 'tests/test_files/{}/example_pdb_{}_0001.pdb'.format(
                        pdb_dir, pdb
                )
                pdb_path = 'tests/test_files/{}/example_pdb_{}.pdb'.format(
                        pdb_dir, pdb
                )
                with open(rosetta_path, 'r') as f:
                    rosetta_file_lines = [line for line in f.readlines()
                                          if not line.strip() == '']
                with open(pdb_path, 'r') as f:
                    pdb_file_lines = [line for line in f.readlines() if not line.strip() == '']

                pdb_res = []
                rosetta_energies = []
                rosetta_score_start = False
                for line in rosetta_file_lines:
                    res_id = '{}_{}_{}'.format(
                        line[17:20], line[21:22], line[22:26].strip()
                    )  # Will return '' rather than error if list index out of bounds
                    if (
                            line[0:6].strip() in ['ATOM', 'HETATM']
                        and not line[17:20] == 'MEM'
                        and not res_id in pdb_res
                    ):
                        pdb_res.append(res_id)
                    elif line.startswith('pose'):
                        rosetta_score_start = True

                    if (
                            rosetta_score_start is True
                        and not any(line.startswith(x) for x in [
                            'pose', 'VRT', 'MEM', '#END_POSE_ENERGIES_TABLE'
                        ])
                    ):
                        rosetta_energies.append(float(line.split()[-1]))
                exp_res_energies_dict = OrderedDict(zip(pdb_res, rosetta_energies))

                meas_res_energies_dict = parse_rosetta_pdb_file(
                    pdb_path, rosetta_file_lines, pdb_file_lines
                )

        self.assertDictEqual(meas_res_energies_dict, exp_res_energies_dict)

    def test_parse_molprobity_output(self):
        """
        Tests that output file from running MolProbity is being parsed correctly
        """

        for pdb_dir in os.listdir('tests/test_files/'):
            if pdb_dir.endswith('_output_test'):
                pdb = pdb_dir.split('_')[0]
                pdb_file = 'example_pdb_{}.pdb'.format(pdb)
                with open('tests/test_files/{}/{}_struct_molprobity_stdout'
                          '.txt'.format(pdb_dir, pdb), 'r') as f:
                    lines = [line for line in f.readlines()]
                    molp_struct_out = lines[3].split(':')
                    struct_header = lines[2].split(':')
                with open('tests/test_files/{}/{}_res_molprobity_stdout'
                          '.txt'.format(pdb_dir, pdb), 'r') as f:
                    lines = [line for line in f.readlines()]
                    molp_res_out = lines[2:]
                    res_header = lines[1].split(',')

                # Parse test files with program code
                meas_struct_molp = parse_molprobity_struct_output(
                    molp_struct_out, struct_header, pdb_file
                )
                meas_res_molp = parse_molprobity_res_output(
                    molp_res_out, res_header, pdb_file
                )

                # Parse test files with test code
                exp_struct_molp = [
                    molp_struct_out[0],
                    (float(molp_struct_out[15]) / float(molp_struct_out[16])),
                    (float(molp_struct_out[17]) / float(molp_struct_out[18])),
                    (float(molp_struct_out[21]) / float(molp_struct_out[22])),
                    (float(molp_struct_out[20]) / float(molp_struct_out[22])),
                    (float(molp_struct_out[19]) / float(molp_struct_out[22])),
                    float(molp_struct_out[8]),
                    float(molp_struct_out[13])
                ]

                exp_res_molp = OrderedDict({
                    'Residue_id': [],
                    'C_Beta_deviation': [],
                    'Rotamer_score': [],
                    'Rotamer_allowed': [],
                    'Ramachandran_score': [],
                    'Ramachandran_allowed': [],
                    'Worst_clash': []
                })
                for line in molp_res_out:
                    if not line.startswith('#'):
                        line = line.split(',')
                        res_id = line[2].strip().split()
                        exp_res_molp['Residue_id'].append(
                            '{}_{}_{}'.format(res_id[2], res_id[0], res_id[1])
                        )
                        try:
                            exp_res_molp['C_Beta_deviation'].append(float(line[9]))
                        except ValueError:
                            exp_res_molp['C_Beta_deviation'].append(np.nan)
                        try:
                            exp_res_molp['Rotamer_score'].append(float(line[10]))
                        except ValueError:
                            exp_res_molp['Rotamer_score'].append(np.nan)
                        exp_res_molp['Rotamer_allowed'].append(line[11])
                        try:
                            exp_res_molp['Ramachandran_score'].append(float(line[13]))
                        except ValueError:
                            exp_res_molp['Ramachandran_score'].append(np.nan)
                        exp_res_molp['Ramachandran_allowed'].append(line[14])
                        try:
                            exp_res_molp['Worst_clash'].append(float(line[5]))
                        except ValueError:
                            exp_res_molp['Worst_clash'].append(np.nan)
                exp_res_molp = pd.DataFrame(exp_res_molp)

                # Test for equality
                self.assertEqual(meas_struct_molp, exp_struct_molp)
                pd.testing.assert_frame_equal(meas_res_molp, exp_res_molp)
