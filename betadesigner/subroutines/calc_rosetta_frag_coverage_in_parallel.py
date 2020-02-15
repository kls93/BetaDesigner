
import argparse
import os
import pickle
from collections import OrderedDict


def calc_rosetta_frag_coverage(
    pdb_path, wd, tools_dir='/home/shared/rosetta/tools/fragment_tools'
):
    """
    Fabio recommends just running make_fragments.pl rather than the fragment
    picker, since will be much quicker
    """

    pdb = pdb_path.split('/')[-1].replace('.pdb', '')
    wd = '/'.join(pdb_path.split('/')[:-1])  # More complicated to use
    # self.working_directory because PDB file is in its own directory
    wd = '{}/{}_rosetta_results'.format(wd, pdb)
    if not os.path.isdir(wd):
        os.mkdir(wd)
    fasta_path = pdb_path.replace('.pdb', '.fasta')

    # Only running psipred for secondary structure prediction, no sam or jufo
    os.system('.{}/make_fragments.pl {} -rundir {} -nohoms -frag_sizes 9'.format(
        tools_dir, wd, fasta_path  # FASTA file has already been written, at the
        # same time as output PDB file
    ))

    # Extracts fragment Crmsd scores from output of running make_fragments.pl
    frag_scores = OrderedDict()
    with open('{}/frags.200.9mers'.format(wd), 'r') as f:
        frag_score_lines = f.readlines()
    pos_index = ''
    rmsd_index = ''
    for line in frag_score_lines:
        if line.startswith('#'):
            line = line.replace('#', '').strip()
            pos_index = line.split().index('query_pos')
            rmsd_index = line.split().index('FragmentCrmsd')
        else:
            pos = line.split()[pos_index]
            rmsd = line.split()[rmsd_index]
            if not pos in frag_scores:
                frag_scores[pos] = []
            frag_scores[pos].append(float(rmsd))

    # Calculates metrics
    worst_best = ''
    num_frag = 0
    frag_cov = 0

    for pos, rmsds in frag_scores.items():
        pos_worst_best = min(rmsds)
        if worst_best < pos_worst_best:
            worst_best = pos_worst_best

        pos_num_frag = len([x for x in rmsds if x < 1.5])
        num_frag += pos_num_frag

        if pos_num_frag >= 1:
            frag_cov += 1

    return [worst_best, num_frag, frag_cov]


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

    worst_best_frag_dict = OrderedDict()
    num_frag_dict = OrderedDict()
    frag_cov_dict = OrderedDict()

    wd_list = [copy.deepcopy(wd) for n in range(len(pdb_list))]
    rosetta_frag_cov_list = futures.map(
        calc_rosetta_frag_coverage, pdb_list, wd_list
    )

    for tup in rosetta_frag_cov_list:
        worst_best_frag_dict[tup[0]] = tup[1]
        num_frag_dict[tup[0]] = tup[2]
        frag_cov_dict[tup[0]] = tup[3]

    with open('{}/Rosetta_frag_coverage.pkl'.format(wd), 'wb') as f:
        pickle.dump((worst_best_frag_dict, num_frag_dict, frag_cov_dict), f)
