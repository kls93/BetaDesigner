
import copy
import math
import os
import pickle
import shutil
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from itertools import combinations


def calc_pop_score_vs_generation(
    data_dir, output_dir, generations, num_sequences, unchanged_res,
    save_interval=10, bude=True, molprobity=True
):
    """
    """

    per_seq_prop_scores = [[] for n in range(len(generations))]
    per_seq_freq_scores = [[] for n in range(len(generations))]
    per_seq_bude_scores = [[] for n in range(len(generations))]
    per_seq_clash_scores = [[] for n in range(len(generations))]
    pop_prop_scores = [np.nan]*len(generations)
    pop_freq_scores = [np.nan]*len(generations)
    pop_bude_scores = [np.nan]*len(generations)
    pop_clash_scores = [np.nan]*len(generations)
    orig_struct_prop_score = np.nan
    orig_struct_freq_score = np.nan
    orig_struct_bude_score = np.nan
    orig_struct_clash_score = np.nan

    max_cycle = math.ceil(max(generations) / save_interval)
    sub_dirs = ['Optimisation_cycle_{}'.format(int(n))
                for n in range(1, max_cycle+1)]

    for sub_dir in sub_dirs:
        with open(
            '{}/{}/final_run/Program_output/Sequence_track.txt'.format(data_dir, sub_dir), 'r'
        ) as f:
            file_lines = [line for line in f.read().split('\n') if line != '']

        gen_count = np.nan
        for index, line in enumerate(copy.deepcopy(file_lines)):
            # Extracts propensity, frequency and BUDE scores of the input structure
            if line.startswith('initial_network') and file_lines[index-2] == 'Input structure':
                if sub_dir == 'Optimisation_cycle_1':
                    orig_struct_prop_score = float(line.split(',')[2])
                    orig_struct_freq_score = float(line.split(',')[3])
                    orig_struct_bude_score = float(line.split(',')[4])
                    orig_struct_clash_score = float(line.split(',')[5])
                else:
                    if (   np.testing.assert_almost_equal(orig_struct_prop_score, float(line.split(',')[2]), decimal=5) is False
                        or np.testing.assert_almost_equal(orig_struct_freq_score, float(line.split(',')[3]), decimal=5) is False
                        or np.testing.assert_almost_equal(orig_struct_bude_score, float(line.split(',')[4]), decimal=5) is False
                        or np.testing.assert_almost_equal(orig_struct_clash_score, float(line.split(',')[5]), decimal=5) is False
                    ):
                        raise Exception(
                            'Disagreement between generations about initial sequence scores\n'
                            'Orig score: {}\n{} score: {}\n'
                            'Orig score: {}\n{} score: {}\n'
                            'Orig score: {}\n{} score: {}\n'
                            'Orig score: {}\n{} score: {}\n'.format(
                                orig_struct_prop_score, sub_dir, float(line.split(',')[2]),
                                orig_struct_freq_score, sub_dir, float(line.split(',')[3]),
                                orig_struct_bude_score, sub_dir, float(line.split(',')[4]),
                                orig_struct_clash_score, sub_dir, float(line.split(',')[5])
                            )
                        )
            elif line.startswith('Generation'):
                gen_count = int(line.split()[1])
            # Extracts per-sequence propensity, frequency and BUDE scores
            elif unchanged_res in line and not np.isnan(gen_count):
                if gen_count % 2 == 1:
                    per_seq_prop_scores[gen_count-1].append(float(line.split(',')[2]))
                    per_seq_freq_scores[gen_count-1].append(float(line.split(',')[3]))
                    per_seq_bude_scores[gen_count-1].append(np.nan)
                    per_seq_clash_scores[gen_count-1].append(np.nan)
                elif gen_count % 2 == 0:
                    if bude is True and molprobity is False:
                        per_seq_prop_scores[gen_count-1].append(np.nan)
                        per_seq_freq_scores[gen_count-1].append(np.nan)
                        per_seq_bude_scores[gen_count-1].append(float(line.split(',')[2]))
                        per_seq_clash_scores[gen_count-1].append(np.nan)
                    elif bude is False and molprobity is True:
                        per_seq_prop_scores[gen_count-1].append(np.nan)
                        per_seq_freq_scores[gen_count-1].append(np.nan)
                        per_seq_bude_scores[gen_count-1].append(np.nan)
                        per_seq_clash_scores[gen_count-1].append(float(line.split(',')[2]))
                    elif bude is True and molprobity is True:
                        if gen_count % 4 == 2:
                            per_seq_prop_scores[gen_count-1].append(np.nan)
                            per_seq_freq_scores[gen_count-1].append(np.nan)
                            per_seq_bude_scores[gen_count-1].append(float(line.split(',')[2]))
                            per_seq_clash_scores[gen_count-1].append(np.nan)
                        elif gen_count % 4 == 0:
                            per_seq_prop_scores[gen_count-1].append(np.nan)
                            per_seq_freq_scores[gen_count-1].append(np.nan)
                            per_seq_bude_scores[gen_count-1].append(np.nan)
                            per_seq_clash_scores[gen_count-1].append(float(line.split(',')[2]))
                    elif bude is False and molprobity is False:
                        per_seq_prop_scores[gen_count-1].append(float(line.split(',')[2]))
                        per_seq_freq_scores[gen_count-1].append(float(line.split(',')[3]))
                        per_seq_bude_scores[gen_count-1].append(np.nan)
                        per_seq_clash_scores[gen_count-1].append(np.nan)
            # Extracts per-generation propensity, frequency and BUDE scores
            elif line.startswith('Total:') and not np.isnan(gen_count):
                if (gen_count - 1) % save_interval == 0 and gen_count != 1:
                    pass
                elif gen_count % 2 == 1:
                    pop_prop_scores[gen_count-1] = float(line.replace('Total:', '').split(',')[0])
                    pop_freq_scores[gen_count-1] = float(line.replace('Total:', '').split(',')[1])
                elif gen_count % 2 == 0:
                    if bude is True and molprobity is False:
                        pop_bude_scores[gen_count-1] = float(line.replace('Total:', '').split(',')[0])
                    elif bude is False and molprobity is True:
                        pop_clash_scores[gen_count-1] = float(line.replace('Total:', '').split(',')[0])
                    elif bude is True and molprobity is True:
                        if gen_count % 4 == 2:
                            pop_bude_scores[gen_count-1] = float(line.replace('Total:', '').split(',')[0])
                        elif gen_count % 4 == 0:
                            pop_clash_scores[gen_count-1] = float(line.replace('Total:', '').split(',')[0])
                    elif bude is False and molprobity is False:
                        pop_prop_scores[gen_count-1] = float(line.replace('Total:', '').split(',')[0])
                        pop_freq_scores[gen_count-1] = float(line.replace('Total:', '').split(',')[1])
            elif line.startswith('Output generation'):
                break

    # Checks that lists of scores have the expected dimensions
    for i in range(len(generations)):
        if i != 0 and i % save_interval == 0:
            if (   len(per_seq_prop_scores[i]) != (num_sequences / 2)
                or len(per_seq_freq_scores[i]) != (num_sequences / 2)
            ):
                raise Exception(
                    '{} sequences in generation {} - propensity\n{} sequences '
                    'in generation {} - frequency'.format(
                        len(per_seq_prop_scores[i]), i+1,
                        len(per_seq_freq_scores[i]), i+1
                    )
                )
        elif i % 2 == 0:
            if (   len(per_seq_prop_scores[i]) != num_sequences
                or len(per_seq_freq_scores[i]) != num_sequences
            ):
                raise Exception(
                    '{} sequences in generation {} - propensity\n{} sequences '
                    'in generation {} - frequency'.format(
                        len(per_seq_prop_scores[i]), i+1,
                        len(per_seq_freq_scores[i]), i+1
                    )
                )
        elif i % 2 == 1:
            if bude is True and molprobity is False:
                if len(per_seq_bude_scores[i]) != num_sequences:
                    raise Exception(
                        '{} sequences in generation {} - BUDE score\n'.format(
                            len(per_seq_bude_scores[i]), i+1
                        )
                    )
            elif bude is False and molprobity is True:
                if len(per_seq_clash_scores[i]) != num_sequences:
                    raise Exception(
                        '{} sequences in generation {} - clash score\n'.format(
                            len(per_seq_clash_scores[i]), i+1
                        )
                    )
            elif bude is True and molprobity is True:
                if i % 4 == 1:
                    if len(per_seq_bude_scores[i]) != num_sequences:
                        raise Exception(
                            '{} sequences in generation {} - BUDE score\n'.format(
                                len(per_seq_bude_scores[i]), i+1
                            )
                        )
                elif i % 4 == 3:
                    if len(per_seq_clash_scores[i]) != num_sequences:
                        raise Exception(
                            '{} sequences in generation {} - clash score\n'.format(
                                len(per_seq_clash_scores[i]), i+1
                            )
                        )
            elif bude is False and molprobity is False:
                if (   len(per_seq_prop_scores[i]) != num_sequences
                    or len(per_seq_freq_scores[i]) != num_sequences
                ):
                    raise Exception(
                        '{} sequences in generation {} - propensity\n{} '
                        'sequences in generation {} - frequency'.format(
                            len(per_seq_prop_scores[i]), i+1,
                            len(per_seq_freq_scores[i]), i+1
                        )
                    )

    # Generates per-generation plots
    plt.clf()
    sns.scatterplot(x=generations, y=pop_prop_scores, s=15)
    plt.savefig('{}/Generations_vs_net_propensity.svg'.format(output_dir))
    plt.clf()
    sns.scatterplot(x=generations, y=pop_freq_scores, s=15)
    plt.savefig('{}/Generations_vs_net_frequency.svg'.format(output_dir))
    if bude is True:
        plt.clf()
        sns.scatterplot(x=generations, y=pop_bude_scores, s=15)
        plt.savefig('{}/Generations_vs_net_BUDE_score.svg'.format(output_dir))
    if molprobity is True:
        plt.clf()
        sns.scatterplot(x=generations, y=pop_clash_scores, s=15)
        plt.savefig('{}/Generations_vs_net_clash_score.svg'.format(output_dir))

    # Generates per-sequence plots
    expanded_generations = []
    for i in range(len(generations)):
        expanded_generations += [generations[i]]*3

    per_seq_prop_scores_summary = []
    per_seq_prop_hues = []
    per_seq_freq_scores_summary = []
    per_seq_freq_hues = []
    per_seq_bude_scores_summary = []
    per_seq_bude_hues = []
    per_seq_clash_scores_summary = []
    per_seq_clash_hues = []
    for i in range(len(generations)):
        if i % 2 == 0:
            per_seq_prop_scores_summary.append(np.percentile(per_seq_prop_scores[i], 0.025))
            per_seq_prop_scores_summary.append(np.percentile(per_seq_prop_scores[i], 0.5))
            per_seq_prop_scores_summary.append(np.percentile(per_seq_prop_scores[i], 0.975))
            per_seq_prop_hues += ['2.5', '50', '97.5']

            per_seq_freq_scores_summary.append(np.percentile(per_seq_freq_scores[i], 0.025))
            per_seq_freq_scores_summary.append(np.percentile(per_seq_freq_scores[i], 0.5))
            per_seq_freq_scores_summary.append(np.percentile(per_seq_freq_scores[i], 0.975))
            per_seq_freq_hues += ['2.5', '50', '97.5']

            per_seq_bude_scores_summary += [np.nan, np.nan, np.nan]
            per_seq_bude_hues += [np.nan, np.nan, np.nan]

            per_seq_clash_scores_summary += [np.nan, np.nan, np.nan]
            per_seq_clash_hues += [np.nan, np.nan, np.nan]

        else:
            if bude is True and molprobity is False:
                per_seq_bude_scores_summary.append(np.percentile(per_seq_bude_scores[i], 0.025))
                per_seq_bude_scores_summary.append(np.percentile(per_seq_bude_scores[i], 0.5))
                per_seq_bude_scores_summary.append(np.percentile(per_seq_bude_scores[i], 0.975))
                per_seq_bude_hues += ['2.5', '50', '97.5']

                per_seq_clash_scores_summary += [np.nan, np.nan, np.nan]
                per_seq_clash_hues += [np.nan, np.nan, np.nan]

            elif bude is False and molprobity is True:
                per_seq_clash_scores_summary.append(np.percentile(per_seq_clash_scores[i], 0.025))
                per_seq_clash_scores_summary.append(np.percentile(per_seq_clash_scores[i], 0.5))
                per_seq_clash_scores_summary.append(np.percentile(per_seq_clash_scores[i], 0.975))
                per_seq_clash_hues += ['2.5', '50', '97.5']

                per_seq_bude_scores_summary += [np.nan, np.nan, np.nan]
                per_seq_bude_hues += [np.nan, np.nan, np.nan]

            elif bude is True and molprobity is True:
                if i % 4 == 1:
                    per_seq_bude_scores_summary.append(np.percentile(per_seq_bude_scores[i], 0.025))
                    per_seq_bude_scores_summary.append(np.percentile(per_seq_bude_scores[i], 0.5))
                    per_seq_bude_scores_summary.append(np.percentile(per_seq_bude_scores[i], 0.975))
                    per_seq_bude_hues += ['2.5', '50', '97.5']

                    per_seq_clash_scores_summary += [np.nan, np.nan, np.nan]
                    per_seq_clash_hues += [np.nan, np.nan, np.nan]

                elif i % 4 == 3:
                    per_seq_clash_scores_summary.append(np.percentile(per_seq_clash_scores[i], 0.025))
                    per_seq_clash_scores_summary.append(np.percentile(per_seq_clash_scores[i], 0.5))
                    per_seq_clash_scores_summary.append(np.percentile(per_seq_clash_scores[i], 0.975))
                    per_seq_clash_hues += ['2.5', '50', '97.5']

                    per_seq_bude_scores_summary += [np.nan, np.nan, np.nan]
                    per_seq_bude_hues += [np.nan, np.nan, np.nan]

            if bude is False and molprobity is False:
                per_seq_prop_scores_summary.append(np.percentile(per_seq_prop_scores[i], 0.025))
                per_seq_prop_scores_summary.append(np.percentile(per_seq_prop_scores[i], 0.5))
                per_seq_prop_scores_summary.append(np.percentile(per_seq_prop_scores[i], 0.975))
                per_seq_prop_hues += ['2.5', '50', '97.5']

                per_seq_freq_scores_summary.append(np.percentile(per_seq_freq_scores[i], 0.025))
                per_seq_freq_scores_summary.append(np.percentile(per_seq_freq_scores[i], 0.5))
                per_seq_freq_scores_summary.append(np.percentile(per_seq_freq_scores[i], 0.975))
                per_seq_freq_hues += ['2.5', '50', '97.5']

                per_seq_bude_scores_summary += [np.nan, np.nan, np.nan]
                per_seq_bude_hues += [np.nan, np.nan, np.nan]

                per_seq_clash_scores_summary += [np.nan, np.nan, np.nan]
                per_seq_clash_hues += [np.nan, np.nan, np.nan]

            else:
                per_seq_prop_scores_summary += [np.nan, np.nan, np.nan]
                per_seq_prop_hues += [np.nan, np.nan, np.nan]

                per_seq_freq_scores_summary += [np.nan, np.nan, np.nan]
                per_seq_freq_hues += [np.nan, np.nan, np.nan]

    plt.clf()
    sns.scatterplot(
        x=expanded_generations, y=per_seq_prop_scores_summary,
        hue=per_seq_prop_hues, s=15,
        palette={'2.5': 'tab:green', '50': 'tab:blue', '97.5': 'tab:orange'}
    )
    plt.plot([min(expanded_generations), max(expanded_generations)],
             [orig_struct_prop_score, orig_struct_prop_score], 'k')
    plt.savefig('{}/Generations_vs_range_propensity.svg'.format(output_dir))
    plt.clf()
    sns.scatterplot(
        x=expanded_generations, y=per_seq_freq_scores_summary,
        hue=per_seq_freq_hues, s=15,
        palette={'2.5': 'tab:green', '50': 'tab:blue', '97.5': 'tab:orange'}
    )
    plt.plot([min(expanded_generations), max(expanded_generations)],
             [orig_struct_freq_score, orig_struct_freq_score], 'k')
    plt.savefig('{}/Generations_vs_range_frequency.svg'.format(output_dir))
    if bude is True:
        plt.clf()
        sns.scatterplot(
            x=expanded_generations, y=per_seq_bude_scores_summary,
            hue=per_seq_bude_hues, s=15,
            palette={'2.5': 'tab:green', '50': 'tab:blue', '97.5': 'tab:orange'}
        )
        plt.plot([min(expanded_generations), max(expanded_generations)],
                 [orig_struct_bude_score, orig_struct_bude_score], 'k')
        plt.savefig('{}/Generations_vs_range_BUDE_score.svg'.format(output_dir))
    if molprobity is True:
        plt.clf()
        sns.scatterplot(
            x=expanded_generations, y=per_seq_clash_scores_summary,
            hue=per_seq_clash_hues, s=15,
            palette={'2.5': 'tab:green', '50': 'tab:blue', '97.5': 'tab:orange'}
        )
        plt.plot([min(expanded_generations), max(expanded_generations)],
                 [orig_struct_clash_score, orig_struct_clash_score], 'k')
        plt.savefig('{}/Generations_vs_range_clash_score.svg'.format(output_dir))


def track_sequence_diversity(generations, data_dir, output_dir, save_interval=10):
    """
    """

    max_cycle = math.ceil(max(generations) / save_interval)
    sub_dirs = ['Optimisation_cycle_{}'.format(int(n))
                for n in range(1, max_cycle+1)]

    seq_diversity = []
    generations = []
    for sub_dir in sub_dirs:
        seq_file = '{}/{}/final_run/Program_output/Sequence_track.txt'.format(
            data_dir, sub_dir
        )
        with open(seq_file, 'r') as f:
            file_lines = [line for line in f.read().split('\n') if line != '']

        # Extracts generation ids and sequences from Sequence_track.txt log
        gen_count = np.nan
        if sub_dir.split('_')[-1] == '1':
            sequences = [[] for n in range(save_interval)]
            sub_gens = []
        else:
            sequences = [[] for n in range(save_interval-1)]
            sub_gens = []
        for index, line in enumerate(copy.deepcopy(file_lines)):
            if line.startswith('Generation'):
                gen_count = int(line.replace('Generation', ''))
                if sub_dir.split('_')[-1] == '1':
                    sub_gens.append(gen_count)
                elif (    sub_dir.split('_')[-1] != '1'
                      and (gen_count - 1) % save_interval != 0
                ):
                    sub_gens.append(gen_count)
            elif unchanged_res in line and not np.isnan(gen_count):
                if sub_dir.split('_')[-1] == '1':
                    sequences[gen_count-1].append(
                        line.split(',')[1].replace(unchanged_res, '')
                    )
                else:
                    if (gen_count % save_interval) != 1:
                        sequences[(gen_count-2) % save_interval].append(
                            line.split(',')[1].replace(unchanged_res, '')
                        )
            elif line.startswith('Output generation'):
                break

        if len(sequences) != len(sub_gens):
            raise Exception(
                'Discrepancy in number of generations measured - {} generations'
                ' listed in {}, but only {} cohorts of sequences recorded'.format(
                    len(sub_gens), seq_file, len(sequences)
                )
            )

        # Measures sequence diversity as the sum of the number of positions at
        # which every possible sequence pair in a generation differ. All
        # mutations are treated equally, i.e. exchanges of chemically dissimilar
        # residues are not upweighted relative to exchanges between more similar
        # residues (e.g. D and E)
        for sub_seq in sequences:
            sub_seq_diversity = 0
            seq_pairs = [[comb[0], comb[1]] for comb in combinations(sub_seq, 2)]
            for seq_pair in seq_pairs:
                seq_a = seq_pair[0]
                seq_b = seq_pair[1]
                if len(seq_a) != len(seq_b):
                    raise Exception(
                        'Sequences are different lengths:\n{}\n{}'.format(seq_a, seq_b)
                    )
                for n in range(len(seq_a)):
                    if seq_a[n] == seq_b[n]:
                        pass
                    else:
                        sub_seq_diversity += 1
            seq_diversity.append(sub_seq_diversity)
        generations += sub_gens

    # Generates plot of sequence diversity vs. generation
    plt.clf()
    sns.scatterplot(x=generations, y=seq_diversity, s=15)
    plt.savefig('{}/Generations_vs_sequence_diversity.svg'.format(output_dir))


def plot_output_structure_per_struct_scores(data_dir, output_dir):
    """
    """

    struct_df = pd.read_pickle('{}/Per_struct_scores.pkl'.format(data_dir))

    # Scores for original structure (2JMM)
    cbeta_orig = 0.0
    rotamer_outlier_orig = 0.041666666666666664
    ramachandran_favoured_orig = 0.7857142857142857
    ramachandran_allowed_orig = 0.16233766233766234
    ramachandran_outlier_orig = 0.05194805194805195
    clashscore_orig = 8.43
    clashscore_percentile_orig = 80.0
    bude_score_orig = -1562.4468908638771
    rosetta_score_orig = -470.386

    # Per-structure score plots
    struct_list = [np.nan]*struct_df.shape[0]
    cbeta_outliers = [np.nan]*struct_df.shape[0]
    rotamer_outliers = [np.nan]*struct_df.shape[0]
    ramachandran_outliers = [[np.nan, np.nan, np.nan]
                             for n in range(struct_df.shape[0])]
    clashscores = [np.nan]*struct_df.shape[0]
    clash_percentile_scores = [np.nan]*struct_df.shape[0]
    bude_scores = [np.nan]*struct_df.shape[0]
    rosetta_scores = [np.nan]*struct_df.shape[0]

    for n in range(struct_df.shape[0]):
        struct_list[n] = struct_df['Structure_id'][n].split('/')[-1].replace('.pdb', '')
        cbeta_outliers[n] = struct_df['C_Beta_outliers'][n]
        rotamer_outliers[n] = struct_df['Rotamer_outliers'][n]
        ramachandran_outliers[n] = [struct_df['Ramachandran_favoured'][n],
                                    struct_df['Ramachandran_allowed'][n],
                                    struct_df['Ramachandran_outliers'][n]]
        clashscores[n] = struct_df['Clashscore'][n]
        clash_percentile_scores[n] = struct_df['Clashscore_percentile'][n]
        bude_scores[n] = struct_df['BUDE_score'][n]
        rosetta_scores[n] = struct_df['Rosetta_score'][n]

    plt.clf()
    sns.barplot(x=struct_list, y=cbeta_outliers, ci=None, color='tab:blue')
    plt.plot(struct_list, [cbeta_orig]*len(struct_list), 'k')
    plt.xticks(rotation=90)
    plt.savefig(
        '{}/Output_structure_CBeta_outliers_vs_sequence.svg'.format(output_dir)
    )

    plt.clf()
    sns.barplot(x=struct_list, y=rotamer_outliers, ci=None, color='tab:blue')
    plt.plot(struct_list, [rotamer_outlier_orig]*len(struct_list), 'k')
    plt.xticks(rotation=90)
    plt.savefig(
        '{}/Output_structure_rotamer_outliers_vs_sequence.svg'.format(output_dir)
    )

    ramachandran_outliers = [val for sub_list in ramachandran_outliers
                             for val in sub_list]
    expanded_struct_list = []
    for n in range(len(struct_list)):
        expanded_struct_list += [struct_list[n]]*3
    hue_list = ['Favoured', 'Allowed', 'Outlier']*len(struct_list)
    plt.clf()
    sns.barplot(x=expanded_struct_list, y=ramachandran_outliers, hue=hue_list,
                ci=None)
    plt.plot(struct_list, [ramachandran_favoured_orig]*len(struct_list), 'k')
    plt.plot(struct_list, [ramachandran_allowed_orig]*len(struct_list), 'k')
    plt.plot(struct_list, [ramachandran_outlier_orig]*len(struct_list), 'k')
    plt.xticks(rotation=90)
    plt.savefig(
        '{}/Output_structure_ramachandran_outliers_vs_sequence.svg'.format(output_dir)
    )

    plt.clf()
    sns.barplot(x=struct_list, y=clashscores, ci=None, color='tab:blue')
    plt.plot(struct_list, [clashscore_orig]*len(struct_list), 'k')
    plt.xticks(rotation=90)
    plt.savefig('{}/Output_structure_clashscore_vs_sequence.svg'.format(output_dir))

    plt.clf()
    sns.barplot(x=struct_list, y=clash_percentile_scores, ci=None, color='tab:blue')
    plt.plot(struct_list, [clashscore_percentile_orig]*len(struct_list), 'k')
    plt.xticks(rotation=90)
    plt.savefig(
        '{}/Output_structure_clashscore_percentile_vs_sequence.svg'.format(output_dir)
    )

    plt.clf()
    sns.barplot(x=struct_list, y=bude_scores, ci=None, color='tab:blue')
    plt.plot(struct_list, [bude_score_orig]*len(struct_list), 'k')
    plt.xticks(rotation=90)
    plt.savefig('{}/Output_structure_BUDE_score_vs_sequence.svg'.format(output_dir))

    plt.clf()
    sns.barplot(x=struct_list, y=rosetta_scores, ci=None, color='tab:blue')
    plt.plot(struct_list, [rosetta_score_orig]*len(struct_list), 'k')
    plt.xticks(rotation=90)
    plt.savefig(
        '{}/Output_structure_Rosetta_scores_vs_sequence.svg'.format(output_dir)
    )


def plot_output_structure_per_res_scores(data_dir, output_dir):
    """
    """

    res_df_dict = pd.read_pickle('{}/Per_res_scores.pkl'.format(data_dir))

    # Per-residue score plots
    for full_struct, res_df in res_df_dict.items():
        struct_id = full_struct.split('/')[-1].replace('.pdb', '')
        plot_dir = '{}/{}'.format(output_dir, struct_id)
        os.mkdir(plot_dir)

        res_list = [np.nan]*res_df.shape[0]
        cbeta_scores = [np.nan]*res_df.shape[0]
        rotamer_scores = [np.nan]*res_df.shape[0]
        rotamer_categories = [np.nan]*res_df.shape[0]
        ramachandran_scores = [np.nan]*res_df.shape[0]
        ramachandran_categories = [np.nan]*res_df.shape[0]
        clash_scores = [np.nan]*res_df.shape[0]
        rosetta_scores = [np.nan]*res_df.shape[0]

        for n in range(res_df.shape[0]):
            res_list[n] = res_df['Residue_id'][n]
            cbeta_scores[n] = res_df['C_Beta_deviation'][n]
            rotamer_scores[n] = res_df['Rotamer_score'][n]
            rotamer_categories[n] = res_df['Rotamer_allowed'][n]
            ramachandran_scores[n] = res_df['Ramachandran_score'][n]
            ramachandran_categories[n] = res_df['Ramachandran_allowed'][n]
            clash_scores[n] = res_df['Worst_clash'][n]
            rosetta_scores[n] = res_df['Rosetta_score'][n]

        plt.clf()
        sns.barplot(x=res_list, y=cbeta_scores, ci=None, color='tab:blue')
        plt.xticks(rotation=90)
        plt.savefig(
            '{}/{}_CBeta_deviations_vs_residue.svg'.format(plot_dir, struct_id)
        )

        plt.clf()
        sns.barplot(
            x=res_list, y=rotamer_scores, hue=rotamer_categories, ci=None
        )
        plt.xticks(rotation=90)
        plt.savefig(
            '{}/{}_rotamer_scores_vs_residue.svg'.format(plot_dir, struct_id)
        )

        plt.clf()
        sns.barplot(
            x=res_list, y=ramachandran_scores, hue=ramachandran_categories,
            ci=None
        )
        plt.xticks(rotation=90)
        plt.savefig(
            '{}/{}_ramachandran_categories_vs_residue.svg'.format(plot_dir, struct_id)
        )

        plt.clf()
        sns.barplot(x=res_list, y=clash_scores, ci=None, color='tab:blue')
        plt.xticks(rotation=90)
        plt.savefig(
            '{}/{}_worst_clash_score_vs_residue.svg'.format(plot_dir, struct_id)
        )

        plt.clf()
        sns.barplot(x=res_list, y=rosetta_scores, ci=None, color='tab:blue')
        plt.xticks(rotation=90)
        plt.savefig(
            '{}/{}_rosetta_scores_vs_residue.svg'.format(plot_dir, struct_id)
        )


# Runs loop of functions above to make plots for all parameter combinations tested
prop_weights = [0.1, 0.3, 0.5, 0.7, 0.9]
mut_probs = [0.025]  # [0.005, 0.01, 0.025, 0.04, 0.05]
cross_probs = [0.3]  # [0.1, 0.2, 0.3, 0.4, 0.5]

num_sequences = 200
unchanged_res = 'NDQLYPRKLDVSEPSDYQVNRGPKARIRNNFGEPKATAQ'

for prop_weight in prop_weights:
    for mut_prob in mut_probs:
        for cross_prob in cross_probs:
            if prop_weight == 0.1:
                generations = list(range(1, 411))
            else:
                generations = list(range(1, 1001))
            data_dir = (
                '/home/ks17361/isambard2/Test_design_results/Barrel_designs/'
                'BetaDesigner_results_bude/2jmm_mutprob_{}_crossprob'
                '_{}_propweight_{}'.format(mut_prob, cross_prob, prop_weight)
            )
            output_dir = '{}/Output_plots'.format(data_dir)
            if os.path.isdir(output_dir):
                shutil.rmtree(output_dir)
            os.mkdir(output_dir)

            calc_pop_score_vs_generation(
                data_dir, output_dir, generations, num_sequences, unchanged_res,
                10, True, True
            )
            track_sequence_diversity(generations, data_dir, output_dir, 10)
            plot_output_structure_per_struct_scores(data_dir, output_dir)
            plot_output_structure_per_res_scores(data_dir, output_dir)
