
# Running a subset of a program in parallel with scoop / multiprocessing in
# Python requires the code in question to be separated into its own file and
# executed within an if __name__ == '__main__' clause

import argparse
import os
import pickle
import shutil
import numpy as np
import pandas as pd
from collections import OrderedDict
from scoop import futures


def parse_molprobity_struct_output(
    molp_stdout_struct, stdout_struct_header, pdb_path
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

    molp_struct_prop = [pdb_path, cbeta_outliers, rot_outliers, rama_favoured,
                        rama_allowed, rama_outlier, clashscore,
                        clashscore_percentile]

    return molp_struct_prop


def parse_molprobity_res_output(molp_stdout_res, stdout_res_header, pdb_path):
    """
    """

    per_res_molp_dict = OrderedDict({
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
        res_id = '{}_{}_{}'.format(res_id[2], res_id[0], res_id[1])
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

        per_res_molp_dict['Residue_id'][res_index] = res_id
        per_res_molp_dict['C_Beta_deviation'][res_index] = cbeta_dev
        per_res_molp_dict['Rotamer_score'][res_index] = rota_score
        per_res_molp_dict['Rotamer_allowed'][res_index] = rota_eval
        per_res_molp_dict['Ramachandran_score'][res_index] = rama_score
        per_res_molp_dict['Ramachandran_allowed'][res_index] = rama_eval
        per_res_molp_dict['Worst_clash'][res_index] = worst_clash

        res_index +=1

    per_res_molp_df = pd.DataFrame(per_res_molp_dict)

    return per_res_molp_df


def score_pdb_molprobity(pdb_path):
    """
    """

    pdb = pdb_path.split('/')[-1]
    wd = '/'.join(pdb_path.split('/')[:-1])  # More complicated to use
    # self.working_directory because PDB file is in its own directory

    # Structure-wide metrics. Note oneline-analysis is provided with an input
    # directory rather than an input pdb file, hence the input pdb file is
    # temporarily copied to a separate directory
    os.mkdir('{}/temporary/'.format(wd))
    shutil.copy(pdb_path, '{}/temporary/'.format(wd))
    os.system(
        'oneline-analysis {}/temporary/ > {}/{}_struct_molprobity_'
        'stdout.txt'.format(wd, wd, pdb)
    )
    shutil.rmtree('{}/temporary/'.format(wd))  # Deletes files created whilst
    # running MolProbity

    molp_stdout_struct = []
    stdout_struct_header = []
    with open('{}/{}_struct_molprobity_stdout.txt'.format(wd, pdb), 'r') as f:
        for index, line in enumerate(f.readlines()):
            if line.startswith(pdb):
                molp_stdout_struct = line.split(':')
            elif line.startswith('#pdbFileName'):
                stdout_struct_header = line.split(':')
    if molp_stdout_struct == [] or stdout_struct_header == []:
        raise Exception('Unexpected formatting of Molprobity standard output')

    per_struct_molp_prop = parse_molprobity_struct_output(
        molp_stdout_struct, stdout_struct_header, pdb_path
    )

    # Per-residue metrics
    os.system(
        'residue-analysis {} > {}/{}_res_molprobity_stdout'
        '.txt'.format(pdb_path, wd, pdb)
    )
    shutil.rmtree('{}/molprobity_results/'.format(wd))  # Deletes files created
    # by running MolProbity

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

    per_res_molp_df = parse_molprobity_res_output(
        molp_stdout_res, stdout_res_header, pdb_path
    )

    return [pdb_path, per_struct_molp_prop, per_res_molp_df]


if __name__ == '__main__':
    # Reads in command line inputs
    parser = argparse.ArgumentParser()
    parser.add_argument('-pdb_list', help='Absolute file path of pickled '
                        'structures output from running the GA')
    parser.add_argument('-o', '--output', help='Location to which to save the '
                        'output pickled dictionary of Rosetta scores')
    args = parser.parse_args()

    pdb_list = vars(args)['pdb_list']
    with open(pdb_list, 'rb') as f:
        pdb_list = pickle.load(f)
    wd = vars(args)['output']

    per_struct_molp_dict = OrderedDict({
        'Structure_id': ['']*len(pdb_list),
        'C_Beta_outliers': ['']*len(pdb_list),
        'Rotamer_outliers': ['']*len(pdb_list),
        'Ramachandran_favoured': ['']*len(pdb_list),
        'Ramachandran_allowed': ['']*len(pdb_list),
        'Ramachandran_outliers': ['']*len(pdb_list),
        'Clashscore': ['']*len(pdb_list),
        'Clashscore_percentile': ['']*len(pdb_list)
    })
    per_res_molp_dict = OrderedDict()

    molprobity_scores = futures.map(score_pdb_molprobity, pdb_list)

    for index, tup in enumerate(molprobity_scores):
        pdb_path = tup[0]
        struct_list = tup[1]
        res_df = tup[2]

        if pdb_path != struct_list[0]:
            raise Exception(
                'PDB path {} != PDB path {}'.format(pdb_path, struct_list[0])
            )
        per_struct_molp_dict['Structure_id'][index] = struct_list[0]
        per_struct_molp_dict['C_Beta_outliers'][index] = struct_list[1]
        per_struct_molp_dict['Rotamer_outliers'][index] = struct_list[2]
        per_struct_molp_dict['Ramachandran_favoured'][index] = struct_list[3]
        per_struct_molp_dict['Ramachandran_allowed'][index] = struct_list[4]
        per_struct_molp_dict['Ramachandran_outliers'][index] = struct_list[5]
        per_struct_molp_dict['Clashscore'][index] = struct_list[6]
        per_struct_molp_dict['Clashscore_percentile'][index] = struct_list[7]
        per_struct_molp_df = pd.DataFrame(per_struct_molp_dict)

        per_res_molp_dict[pdb_path] = res_df

    with open('{}/MolProbity_scores.pkl'.format(wd), 'wb') as f:
        pickle.dump((per_struct_molp_df, per_res_molp_dict), f)
