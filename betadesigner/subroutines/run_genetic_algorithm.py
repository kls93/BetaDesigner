
# Do I need to worry about genetic algorithm overfitting? NO - GAs, like other
# optimisation techniques, can't be overfit. You can get overfitting if using
# the optimisation technique on something which can be overfit, e.g. for
# hyperpameter selection for an ML algorithm, but this is not the case here.

import budeff
import copy
import isambard
import random
import isambard.modelling as modelling
import networkx as nx
import numpy as np
from collections import OrderedDict
from operator import itemgetter

if __name__ == 'subroutines.run_genetic_algorithm':
    from subroutines.find_parameters import initialise_ga_object
    from subroutines.generate_initial_sequences import (
        linear_interpolation, random_shuffle, propensity_to_probability_distribution,
        frequency_to_probability_distribution, gen_cumulative_probabilities
    )
    from subroutines.variables import gen_amino_acids_dict
else:
    from betadesigner.subroutines.find_parameters import initialise_ga_object
    from betadesigner.subroutines.generate_initial_sequences import (
        linear_interpolation, random_shuffle, propensity_to_probability_distribution,
        frequency_to_probability_distribution, gen_cumulative_probabilities
    )
    from betadesigner.subroutines.variables import gen_amino_acids_dict

aa_code_dict = gen_amino_acids_dict()

def pack_side_chains(ampal_object, G, rigid_rotamers):
    """
    Uses SCWRL4 to pack network side chains onto a backbone structure and
    measures the total energy of the model within BUDE
    """

    # Makes FASTA sequence to feed into SCWRL4. BEWARE: if FASTA sequence is
    # shorter than AMPAL object, SCWRL4 will add random amino acids to the end
    # of the sequence until it is the same length.
    fasta_seq = ''
    for res in ampal_object.get_monomers():
        res_id = '{}{}{}{}'.format(
            res.parent.parent.id, res.parent.id, res.id, res.insertion_code
        )  # structure id, chain id, residue number, insertion code e.g. 4pnbD24

        if res_id in list(G.nodes):
            fasta_seq += G.nodes[res_id]['aa_id']
        else:
            fasta_seq += res.mol_letter  # Retains original ids of residues
            # outside of sequence to be mutated, e.g. in loop regions

    if len(fasta_seq) != len(list(ampal_object.get_monomers())):
        raise Exception('FASTA sequence and AMPAL object contain different '
                        'numbers of amino acids')

    # Packs side chains with SCWRL4. NOTE that fasta sequence must be provided
    # as a list. NOTE: Setting rigid_rotamers to True increases the speed of
    # side-chain but results in a concomitant decrease in accuracy.
    new_ampal_object = modelling.pack_side_chains_scwrl(
        ampal_object, [fasta_seq], rigid_rotamer_model=rigid_rotamers,
        hydrogens=False
    )

    # Calculates total energy of the AMPAL object within BUDE (note that this
    # does not include the interaction of the object with its surrounding
    # environment, hence hydrophobic side chains will not be penalised on the
    # surface of a globular protein and vice versa for membrane proteins).
    # Hence this is just a rough measure of side-chain clashes.
    energy = budeff.get_internal_energy(new_ampal_object).total_energy

    return new_ampal_object, energy


class run_ga_calcs(initialise_ga_object):
    """
    Functions required to run each stage of the GA (measuring fitness,
    selecting a mating population, performing crossover and mutation, and
    merging the parent and child generations)
    """

    def __init__(self, params, test=False):
        initialise_ga_object.__init__(self, params, test)
        self.propensity_weight = params['propvsfreqweight']
        self.unfit_fraction = params['unfitfraction']
        self.crossover_prob = params['crossoverprob']
        self.mutation_prob = params['mutationprob']

    def measure_fitness_propensity(self, surface, networks_dict):
        """
        Measures fitness of amino acid sequences from their propensities for
        the structural features of the input backbone structure.
        """

        print('Measuring {} network fitness'.format(surface))

        # Initialises dictionaries of network propensity and frequency scores
        network_propensity_scores = OrderedDict()
        network_frequency_scores = OrderedDict()

        # Extracts propensity and frequency scales for the surface (both
        # individual and pairwise)
        intext_index = self.dict_name_indices['intorext']
        eoc_index = self.dict_name_indices['edgeorcent']
        prop_index = self.dict_name_indices['prop1']
        interaction_index = self.dict_name_indices['interactiontype']
        pairindv_index = self.dict_name_indices['pairorindv']
        discorcont_index = self.dict_name_indices['discorcont']
        proporfreq_index = self.dict_name_indices['proporfreq']

        sub_indv_dicts = OrderedDict({
            dict_label: aa_dict for dict_label, aa_dict in
            {**self.propensity_dicts, **self.frequency_dicts}.items() if
             (    (dict_label.split('_')[intext_index] in [surface[0:3], '-'])
              and (dict_label.split('_')[pairindv_index] == 'indv'))
        })
        sub_pair_dicts = OrderedDict({
            dict_label: aa_dict for dict_label, aa_dict in
            {**self.propensity_dicts, **self.frequency_dicts}.items() if
             (    (dict_label.split('_')[intext_index] in [surface[0:3], '-'])
              and (dict_label.split('_')[pairindv_index] == 'pair'))
        })

        for num in list(networks_dict.keys()):
            G = networks_dict[num]
            # Total propensity count (across all nodes in network)
            propensity_count = 0
            frequency_count = 0

            for node_1 in list(G.nodes):
                # Filters propensity and / or frequency scales depending upon
                # whether the node is in an edge or a central strand
                if self.barrel_or_sandwich == '2.60':
                    eoc_1 = G.nodes[node_1]['eoc']

                    sub_indv_dicts = OrderedDict({
                        dict_label: aa_dict for dict_label, aa_dict
                        in sub_indv_dicts.items()
                        if dict_label.split('_')[eoc_index] in [eoc_1, '-']
                    })
                    sub_pair_dicts = OrderedDict({
                        dict_label: aa_dict for dict_label, aa_dict
                        in sub_pair_dicts.items()
                        if dict_label.split('_')[eoc_index] in [eoc_1, '-']
                    })

                # Calculates interpolated propensity of each node for all
                # individual structural features considered
                aa_1 = G.nodes[node_1]['aa_id']
                for dict_label, scale_dict in sub_indv_dicts.items():
                    weight = self.dict_weights[dict_label]

                    node_prop = dict_label.split('_')[prop_index]
                    node_val = np.nan

                    if node_prop != '-':
                        try:
                            node_val = G.nodes[node_1][node_prop]
                        except KeyError:
                            raise KeyError('{} not defined for node {}'.format(
                                node_prop, node_1
                            ))

                    # Converts non-float values into np.nan
                    if node_val in ['', 'nan', 'NaN', np.nan]:
                        node_val = np.nan

                    value = np.nan
                    if (    dict_label.split('_')[discorcont_index] == 'cont'
                        and not np.isnan(node_val)
                    ):
                        # Interpolate dictionary
                        value = linear_interpolation(node_val, scale_dict[aa_1], dict_label)

                    elif dict_label.split('_')[discorcont_index] == 'disc':
                        # Filter dataframe
                        scale_dict_copy = copy.deepcopy(scale_dict).set_index('FASTA', drop=True)
                        if node_prop == '-':
                            try:
                                value = scale_dict_copy.iloc[:,0][aa_1]
                            except KeyError:
                                raise Exception('{} not defined in {}'.format(aa_1, dict_label))
                        elif node_prop == 'phipsi':
                            if not np.isnan(node_val):
                                try:
                                    value = scale_dict_copy[node_val][aa_1]
                                except KeyError:
                                    raise Exception('{} not defined in {}'.format(aa_1, dict_label))

                    if not np.isnan(value):
                        if dict_label.split('_')[proporfreq_index] == 'propensity':
                            # NOTE: Must take -ve logarithm of each
                            # individual propensity score before summing
                            # (rather than taking the -ve logarithm of the
                            # summed propensities)
                            value = weight*np.negative(np.log(value))
                            propensity_count += value
                        elif dict_label.split('_')[proporfreq_index] == 'frequency':
                            value *= weight
                            frequency_count += value

                # Loops through each node pair to sum pairwise propensity values
                for node_pair in G.edges(node_1):
                    # No need to randomly order pair since each will be counted
                    # twice in this analysis (once for node 1 and once for node 2)
                    # (and so which node's value will be used
                    node_2 = node_pair[1]
                    aa_1 = G.nodes[node_1]['aa_id']
                    aa_2 = G.nodes[node_2]['aa_id']
                    aa_pair = '{}_{}'.format(aa_1, aa_2)

                    # Loops through each interaction between a pair of nodes
                    for edge in G[node_1][node_2]:
                        edge_label = G[node_1][node_2][edge]['interaction']

                        for dict_label, scale_dict in sub_pair_dicts.items():
                            if dict_label.split('_')[interaction_index] == edge_label:
                                weight = self.dict_weights[dict_label]

                                node_prop = dict_label.split('_')[prop_index]
                                node_val = np.nan

                                if node_prop != '-':
                                    try:
                                        node_val = G.nodes[node_1][node_prop]
                                    except KeyError:
                                        raise KeyError('{} not defined for node {}'.format(
                                            node_prop, node_1
                                        ))

                                # Converts non-float values into np.nan
                                if node_val in ['', 'nan', 'NaN', np.nan]:
                                    node_val = np.nan

                                value = np.nan
                                if (    dict_label.split('_')[discorcont_index] == 'cont'
                                    and not np.isnan(node_val)
                                ):
                                    aa_scale = scale_dict[aa_pair]
                                    value = linear_interpolation(node_val, aa_scale, dict_label)

                                elif dict_label.split('_')[discorcont_index] == 'disc':
                                    # Filter dataframe
                                    scale_dict_copy = scale_dict.set_index('FASTA', drop=True)
                                    value = scale_dict_copy[aa_1][aa_2]

                                if not np.isnan(value):
                                    if dict_label.split('_')[proporfreq_index] == 'propensity':
                                        # NOTE: Must take -ve logarithm of each
                                        # individual propensity score before
                                        # summing (rather than taking the -ve
                                        # logarithm of the summed propensities)
                                        value = weight*np.negative(np.log(value))
                                        propensity_count += value
                                    elif dict_label.split('_')[proporfreq_index] == 'frequency':
                                        value *= weight
                                        frequency_count += value

            if propensity_count == 0:
                print('WARNING: propensity for network {} is 0'.format(num))
            network_propensity_scores[num] = propensity_count
            network_frequency_scores[num] = frequency_count

        return network_propensity_scores, network_frequency_scores

    def combine_prop_and_freq_scores(self, network_prop_scores,
                                     network_freq_scores, raw_or_rank):
        """
        Combines propensity and frequency scores
        """

        for index, network_score_dict in enumerate([network_prop_scores, network_freq_scores]):
            # Only need to order propensities if converting to probability
            # scores via their rank
            if index == 0:
                prop_or_freq = 'propensity'
                if raw_or_rank == 'rank':
                    network_score_dict = OrderedDict(sorted(
                        network_score_dict.items(), key=itemgetter(1), reverse=False
                    ))
            elif index == 1:
                prop_or_freq == 'frequency'

            network_num = np.array(list(network_score_dict.keys()))
            network_scores = np.array(list(network_score_dict.values()))

            if index == 0 and raw_or_rank == 'raw':
                (network_num, network_scores, network_prob
                ) = propensity_to_probability_distribution(
                    network_num, network_scores
                )
            elif index == 0 and raw_or_rank == 'rank':
                (network_num, network_scores, network_prob
                ) = frequency_to_probability_distribution(
                    network_num, network_scores, prop_or_freq
                )
            elif index == 1:
                if self.frequency_dicts != {}:
                    (network_num, network_scores, network_prob
                    ) = frequency_to_probability_distribution(
                        network_num, network_scores, prop_or_freq
                    )
                else:
                    network_prob = np.full(network_num.shape, 0)

            if prop_or_freq == 'propensity':
                propensity_array = np.array([copy.deepcopy(network_num),
                                             copy.deepcopy(network_prob)])
            elif prop_or_freq == 'frequency':
                frequency_array = np.array([copy.deepcopy(network_num),
                                             copy.deepcopy(network_prob)])

        network_fitness_scores = OrderedDict()
        for index_prop, network_num in np.ndenumerate(propensity_array[0]):
            index_prop = index_prop[0]
            index_freq = np.where(frequency_array[0] == network_num)[0][0]

            propensity = propensity_array[1][index_prop]
            frequency = frequency_array[1][index_freq]

            probability = (  (self.propensity_weight['propensity']*propensity)
                           + (self.propensity_weight['frequency']*frequency))
            network_fitness_scores[network_num] = probability

        return network_fitness_scores

    def measure_fitness_allatom(self, surface, networks_dict):
        """
        Measures fitness of sequences using an all-atom scoring function
        within BUDE
        """

        print('Measuring {} network fitness'.format(surface))

        # Initialises dictionary of fitness scores
        network_energies = OrderedDict()

        # Loads backbone model into ISAMBARD. NOTE must have been pre-processed
        # to remove ligands etc. so that only backbone coordinates remain.
        pdb = isambard.ampal.load_pdb(self.input_pdb)

        for num, G in networks_dict.items():
            # Packs network side chains onto the model with SCWRL4 and measures
            # the total model energy within BUDE
            new_pdb, energy = pack_side_chains(pdb, G, True)
            network_energies[num] = energy

        return network_energies

    def convert_energies_to_probabilities(self, network_energies):
        """
        Converts energy values output from BUDE into probabilities
        """

        network_fitness_scores = OrderedDict()
        for network_num in list(network_energies.keys()):
            energy = network_energies[network_num]  # Energies output from BUDE
            # are in units of kJ/mol, Boltzmann equation = e^-(ΔE / kT)
            energy = (energy*1000) / (8.314*293)
            eqm_constant = np.exp(np.negative(energy))
            network_fitness_scores[network_num] = eqm_constant
        total = np.sum(list(network_fitness_scores.values()))

        for network_num in list(network_energies.keys()):
            eqm_constant = network_fitness_scores[network_num]
            network_fitness_scores[network_num] = eqm_constant / total

        # Equivalent shuffling of the networks as occurs in the propensity /
        # frequency measures of fitness
        network_fitness_keys = np.array(list(network_fitness_scores.keys()))
        network_fitness_vals = np.array(list(network_fitness_scores.values()))
        dummy_array = np.full(network_fitness_keys.shape, np.nan)
        network_fitness_keys, network_fitness_vals, dummy_array = random_shuffle(
            network_fitness_keys, network_fitness_vals, dummy_array
        )
        network_fitness_scores = OrderedDict(zip(network_fitness_keys, network_fitness_vals))

        return network_fitness_scores

    def create_mat_pop_fittest(self, surface, networks_dict, network_fitness_scores,
                               pop_size, unfit_fraction):
        """
        Creates mating population from the fittest sequences plus a subset of
        less fit sequences (so as to maintain diversity in the mating
        population in order to prevent convergence on a non-global minimum)
        """

        print('Creating mating population for {}'.format(surface))

        # Determines numbers of fittest and random sequences to be added in
        unfit_pop_size = round((pop_size*unfit_fraction), 0)
        pop_size -= unfit_pop_size

        # Initialises dictionary of fittest networks
        mating_pop_dict = OrderedDict()

        # Sorts networks by their fitness values, from most (largest
        # probability) to least (smallest probability) fit
        network_fitness_scores = OrderedDict(sorted(
            network_fitness_scores.items(), key=itemgetter(1), reverse=True
        ))

        # Adds fittest individuals to mating population
        for index, num in enumerate(list(network_fitness_scores.keys())):
            if index < pop_size:
                mating_pop_dict[num] = copy.deepcopy(networks_dict[num])
                network_fitness_scores[num] = ''
            else:
                break

        # Removes fittest networks already included in mating population from
        # dictionary
        unfit_network_indices = [num for num, fitness in
                                 network_fitness_scores.items() if fitness != '']

        # Adds unfit individuals (selected at random) to mating population
        count = 0
        while count < unfit_pop_size:
            random_index = random.randint(0, (len(unfit_network_indices)-1))
            network_num = unfit_network_indices[random_index]
            mating_pop_dict[network_num] = copy.deepcopy(networks_dict[network_num])
            unfit_network_indices = [num for num in unfit_network_indices
                                     if num != network_num]
            count += 1

        return mating_pop_dict

    def create_mat_pop_roulette_wheel(self, surface, networks_dict,
                                      network_fitness_scores, pop_size):
        """
        Creates mating population from individuals, with the likelihood of
        selection of each sequence being weighted by its raw fitness score
        """

        print('Creating mating population for {}'.format(surface))

        # Initialises dictionary of fittest networks
        mating_pop_dict = OrderedDict()

        # Adds individuals (their likelihood of selection weighted by their raw
        # fitness scores) to mating population. Arrays of network numbers,
        # fitness scores and their corresponding probability distribution are
        # updated every cycle to remove the selected network (this prevents the
        # loop from taking a very long time once the highest probability
        # networks have been selected)
        count = 0
        while count < pop_size:
            network_num_array = (
                np.array(list(network_fitness_scores.keys()))
            )
            network_fitness_array = (
                np.array(list(network_fitness_scores.values()))
            )
            network_cumulative_prob = gen_cumulative_probabilities(
                network_fitness_array
            )

            random_number = random.uniform(0, 1)
            nearest_index = (np.abs(network_cumulative_prob-random_number)).argmin()

            if network_cumulative_prob[nearest_index] < random_number:
                nearest_index += 1

            selected_network_num = network_num_array[nearest_index]
            mating_pop_dict[selected_network_num] = copy.deepcopy(
                networks_dict[selected_network_num]
            )

            if count != (pop_size - 1):
                network_fitness_scores = OrderedDict(
                    zip(np.delete(network_num_array, nearest_index),
                        np.delete(network_fitness_array, nearest_index))
                )

            count += 1

        return mating_pop_dict

    def uniform_crossover(self, surface, mating_pop_dict):
        """
        Selects pairs of individuals at random from mating population and
        performs uniform crossover
        """

        print('Performing crossovers for {}'.format(surface))

        # Initialises dictionary of child networks
        crossover_pop_dict = OrderedDict()

        # Selects pairs of networks at random to crossover with each other
        network_num = list(mating_pop_dict.keys())
        random.shuffle(network_num)
        network_num = iter(network_num)  # Do not merge with line below, and do
        # not introduce any lines of code between them!
        network_num = list(zip(network_num, network_num))

        # Performs uniform crossover
        for network_pair in network_num:
            network_num_1 = network_pair[0]
            network_num_2 = network_pair[1]
            mate_1 = copy.deepcopy(mating_pop_dict[network_num_1])
            mate_2 = copy.deepcopy(mating_pop_dict[network_num_2])

            for node in list(mate_1.nodes):
                random_number = random.uniform(0, 1)
                if random_number <= self.crossover_prob:
                    # Copy to prevent these dictionaries from updating when the
                    # node attributes are updated in the code below (otherwise
                    # both nodes will be assigned the same identity as the node
                    # in mate_1, instead of the node identities being crossed
                    # over)
                    mate_1_node_attributes = copy.deepcopy(mate_1.nodes[node])
                    mate_2_node_attributes = copy.deepcopy(mate_2.nodes[node])
                    mate_1.nodes[node] = {}
                    mate_2.nodes[node] = {}
                    nx.set_node_attributes(mate_1, values={node: mate_2_node_attributes})
                    nx.set_node_attributes(mate_2, values={node: mate_1_node_attributes})

            crossover_pop_dict[network_num_1] = mate_1
            crossover_pop_dict[network_num_2] = mate_2

        return crossover_pop_dict

    def segmented_crossover(self, surface, mating_pop_dict):
        """
        Selects pairs of individuals at random from mating population and
        performs segmented crossover
        """

        print('Performing crossovers for {}'.format(surface))

        # Initialises dictionary of child networks
        crossover_pop_dict = OrderedDict()

        # Selects pairs of networks at random to crossover with each other
        network_num = list(mating_pop_dict.keys())
        random.shuffle(network_num)
        network_num = iter(network_num)  # Do not merge with line below, and do
        # not introduce any lines of code between them!
        network_num = list(zip(network_num, network_num))

        # Performs segmented crossover
        for network_pair in network_num:
            network_num_1 = network_pair[0]
            network_num_2 = network_pair[1]
            mate_1 = copy.deepcopy(mating_pop_dict[network_num_1])
            mate_2 = copy.deepcopy(mating_pop_dict[network_num_2])

            swap = False
            for node in list(mate_1.nodes):
                random_number = random.uniform(0, 1)

                if swap is False:
                    if random_number <= self.swap_start_prob:
                        swap = True
                    else:
                        swap = False
                elif swap is True:
                    if random_number <= self.swap_stop_prob:
                        swap = False
                    else:
                        swap = True

                if swap is True:
                    # Copy to prevent these dictionaries from updating when the
                    # node attributes are updated in the code below (otherwise
                    # both nodes will be assigned the same identity as the node
                    # in mate_1, instead of the node identities being crossed
                    # over)
                    mate_1_attributes = copy.deepcopy(mate_1.nodes[node])
                    mate_2_attributes = copy.deepcopy(mate_2.nodes[node])
                    mate_1.nodes[node] = {}
                    mate_2.nodes[node] = {}
                    nx.set_node_attributes(mate_1, values={node: mate_2_attributes})
                    nx.set_node_attributes(mate_2, values={node: mate_1_attributes})

            crossover_pop_dict[network_num_1] = mate_1
            crossover_pop_dict[network_num_2] = mate_2

        return crossover_pop_dict

    def swap_mutate(self, surface, crossover_pop_dict):
        """
        Performs swap mutations (= mutates randomly selected individual
        network nodes to a randomly selected (different) amino acid identity)
        """

        print('Performing mutations for {}'.format(surface))

        # Initialises dictionary of mutated child networks
        mutated_pop_dict = OrderedDict()

        # Mutates the amino acid identities of randomly selected nodes
        for network_num in list(crossover_pop_dict.keys()):
            G = copy.deepcopy(crossover_pop_dict[network_num])

            for node in list(G.nodes):
                random_number = random.uniform(0, 1)
                if random_number <= self.mutation_prob:
                    orig_aa = G.nodes[node]['aa_id']
                    poss_aas = copy.deepcopy(self.aa_list)
                    poss_aas.remove(orig_aa)
                    new_aa = poss_aas[random.randint(0, (len(poss_aas)-1))]

                    nx.set_node_attributes(G, values={node: {'aa_id': new_aa}})

            mutated_pop_dict[network_num] = G

        return mutated_pop_dict

    def scramble_mutate(self, surface, crossover_pop_dict):
        """
        Performs scramble mutations (= scrambles the identities of a subset
        of amino acids selected at random)
        """

        print('Performing mutations for {}'.format(surface))

        # Initialises dictionary of mutated child networks
        mutated_pop_dict = OrderedDict()

        # Scrambles the amino acid identities of randomly selected nodes
        for network_num in list(crossover_pop_dict.keys()):
            G = copy.deepcopy(crossover_pop_dict[network_num])

            scrambled_nodes = []
            aa_ids = []
            for node in list(G.nodes):
                random_number = random.uniform(0, 1)
                if random_number <= self.mutation_prob:
                    scrambled_nodes.append(node)
                    aa_ids.append(G.nodes[node]['aa_id'])

            random.shuffle(aa_ids)
            attributes = OrderedDict({
                node: {'aa_id': aa_id} for node, aa_id in zip(scrambled_nodes, aa_ids)
            })
            nx.set_node_attributes(G, values=attributes)

            mutated_pop_dict[network_num] = G

        return mutated_pop_dict

    def add_children_to_parents(self, surface, mutated_pop_dict,
                                mating_pop_dict, index):
        """
        Combines parent and child generations
        """

        print('Combining parent and child generations for {}'.format(surface))

        # Renumbers networks to prevent overlap
        if index == 0:
            count = 0
        elif index == 1:
            count = 2*self.propensity_pop_size  # The number of networks
            # already added to sequences_dict following propensity scoring
        merged_networks_dict = OrderedDict()

        for num, G in mutated_pop_dict.items():
            merged_networks_dict[count] = copy.deepcopy(G)
            count += 1
        for num, G in mating_pop_dict.items():
            merged_networks_dict[count] = copy.deepcopy(G)
            count += 1

        return merged_networks_dict


def run_genetic_algorithm(params):
    """
    Pipeline function to run genetic algorithm
    """

    print('Running genetic algorithm')

    # Unpacks parameters (unfortunately hyperopt can only feed a single
    # parameter into the objective function, so can't use class inheritance to
    # avoid this issue, and am instead using pickling)
    input_params_file = '{}/Program_input/Input_parameters.pkl'.format(params['workingdirectory'])
    with open(input_params_file, 'rb') as f:
        params = pickle.load(f)
    if type(params) != dict:
        raise TypeError('Data in {} is not a pickled dictionary'.format(input_params_file))

    # Records sequences and their fitnesses after each generation
    with open('{}/Program_output/Sequence_track.txt'.format(
        params['workingdirectory']), 'w') as f:
        f.write('Tracking GA optimisation progress\n')

    ga_calcs = run_ga_calcs(params)

    if params['matingpopmethod'] in ['fittest', 'roulettewheel']:
        raw_or_rank = 'raw'
    elif params['matingpopmethod'] == 'rankroulettewheel':
        raw_or_rank = 'rank'

    gen = 0
    while gen < params['maxnumgenerations']:
        gen += 1
        print('Generation {}'.format(gen))
        with open('Program_output/Sequence_track.txt'.format(
            params['workingdirectory']), 'w') as f:
            f.write('\n\n\n\n\nGeneration {}\n'.format(count))

        for surface in list(params['sequencesdict'].keys()):
            all_networks_dict = params['sequencesdict'][surface]
            all_networks_list = []

            # Splits networks to optimise via different objectives (propensity
            # and side-chain packing) if self.method_fitness_score == 'split'
            if params['fitnessscoremethod'] == 'split':
                prop_networks = OrderedDict(
                    {key: all_networks_dict[key] for index, key in
                    enumerate(list(all_networks_dict.keys()))
                    if index < 2*params['propensitypopsize']}
                )
                energymin_networks = OrderedDict(
                    {key: all_networks_dict[key] for index, key in
                    enumerate(list(all_networks_dict.keys()))
                    if index >= 2*params['propensitypopsize']}
                )
                all_networks_list = [prop_networks, energymin_networks]
                pop_sizes = [params['propensitypopsize'],
                             (params['populationsize']-params['propensitypopsize'])]

            else:
                all_networks_list = [all_networks_dict]
                pop_sizes = [params['populationsize']]

            for index, networks_dict in enumerate(all_networks_list):
                # Measures fitness of sequences in starting population
                if (
                    (params['fitnessscoremethod'] == 'propensity')
                    or
                    (params['fitnessscoremethod'] == 'alternate' and gen % 2 == 1)
                    or
                    (params['fitnessscoremethod'] == 'split' and index == 0)
                ):
                    (network_propensity_scores, network_frequency_scores
                    ) = ga_calcs.measure_fitness_propensity(surface, networks_dict)
                    network_fitness_scores = ga_calcs.combine_prop_and_freq_scores(
                        network_propensity_scores, network_frequency_scores, raw_or_rank
                    )
                elif (
                    (params['fitnessscoremethod'] == 'allatom')
                    or
                    (params['fitnessscoremethod'] == 'alternate' and gen % 2 == 0)
                    or
                    (params['fitnessscoremethod'] == 'split' and index == 1)
                ):
                    (network_energies
                    ) = ga_calcs.measure_fitness_allatom(surface, networks_dict)
                    (network_fitness_scores
                    ) = ga_calcs.convert_energies_to_probabilities(network_energies)

                # Selects subpopulation for mating
                if params['matingpopmethod'] == 'fittest':
                    mating_pop_dict = ga_calcs.create_mat_pop_fittest(
                        surface, networks_dict, network_fitness_scores,
                        pop_sizes[index], params['unfitfraction']
                    )
                elif params['matingpopmethod'] in ['roulettewheel', 'rankroulettewheel']:
                    mating_pop_dict = ga_calcs.create_mat_pop_roulette_wheel(
                        surface, networks_dict, network_fitness_scores, pop_sizes[index], params['']
                    )

                # Performs crossover of parent sequences to generate child sequences
                if params['crossovermethod'] == 'uniform':
                    crossover_pop_dict = ga_calcs.uniform_crossover(surface, mating_pop_dict)
                elif params['crossovermethod'] == 'segmented':
                    crossover_pop_dict = ga_calcs.segmented_crossover(surface, mating_pop_dict)

                # Mutates child sequences
                if params['mutationmethod'] == 'swap':
                    mutated_pop_dict = ga_calcs.swap_mutate(surface, crossover_pop_dict)
                elif params['mutationmethod'] == 'scramble':
                    mutated_pop_dict = ga_calcs.scramble_mutate(surface, crossover_pop_dict)

                # Combines parent and child sequences into single generation
                merged_networks_dict = ga_calcs.add_children_to_parents(
                    surface, mutated_pop_dict, mating_pop_dict, index
                )

                # Shuffles metworks dictionary so that in the case of a split
                # optimisation a mixture of parent and child networks are
                # combined into the sub-classes whose fitnesses are measured by
                # different methods in the following round of optimisation
                if params['fitnessscoremethod'] == 'split' and index == 1:
                    merged_networks_dict = OrderedDict(
                        {**params['sequencesdict'][surface], **merged_networks_dict}
                    )

                merged_networks_keys = list(merged_networks_dict.keys())
                merged_networks_vals = list(merged_networks_dict.values())
                random.shuffle(merged_networks_vals)
                shuffled_merged_networks_dict = OrderedDict(
                    {merged_networks_keys[i]: merged_networks_vals[i]
                     for i in range(len(merged_networks_keys))}
                )
                params['sequencesdict'][surface] = shuffled_merged_networks_dict

                # Records sequences output from this generation and their
                # associated fitnesses
                with open('Program_output/Sequence_track.txt'.format(
                    params['workingdirectory']), 'w') as f:
                    f.write('{}\n'.format(surface))
                    for network, G in params['sequencesdict'][surface].items():
                        sequence = ''.join([G.nodes[node]['aa_id'] for node in G.nodes])
                        probability = network_fitness_scores[network]
                        f.write('{}, {}, {}\n'.format(network, sequence, probability))
                    f.write('\n')

    # Calculates fitness of output sequences and filters population to maintain
    # the fittest 50%, plus sums the probabilities of the retained sequences and
    # returns this value (to be minimised with hyperopt)
    summed_fitness = 0
    for surface in list(params['sequencesdict'].keys()):
        networks_dict = params['sequencesdict'][surface]

        if params['fitnessscoremethod'] != 'allatom':
            (network_propensity_scores, network_frequency_scores
            ) = ga_calcs.measure_fitness_propensity(surface, networks_dict)
            network_fitness_scores = ga_calcs.combine_prop_and_freq_scores(
                network_propensity_scores, network_frequency_scores, raw_or_rank
            )
        elif params['fitnessscoremethod'] == 'allatom':
            network_energies = ga_calcs.measure_fitness_allatom(surface, networks_dict)
            (network_fitness_scores
            ) = ga_calcs.convert_energies_to_probabilities(network_energies)

        mating_pop_dict = ga_calcs.create_mat_pop_fittest(
            surface, networks_dict, network_fitness_scores,
            params['populationsize'], unfit_fraction=0
        )
        params['sequencesdict'][surface] = mating_pop_dict

        for network in params['sequencesdict'][surface].keys():
            summed_fitness += network_fitness_scores[network]

    with open('Program_output/GA_output_sequences_dict.pkl'.format(
        params['workingdirectory']), 'wb') as f:
        pickle.dump(params['sequencesdict'], f)

    return -summed_fitness
