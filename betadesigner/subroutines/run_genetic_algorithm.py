
# Do I need to worry about genetic algorithm overfitting?
import itertools
from collections import OrderedDict

class run_ga():

    def __init__(self, propensity_dicts):
        self.propensity_dicts = propensity_dicts

    def measure_fitness(sequences_dict, propensity_dict_weights):
        fitness_scores = OrderedDict()

        for surface_label, networks_dict in sequences_dict.items():
            sub_indv_propensity_dicts = OrderedDict({
                dict_label: propensity_dict for dict_label, propensity_dict in
                self.propensity_dicts.items() if
                ((dict_label.split('_')[0] == surface_label.split('_')[0])
                 and (dict_label.split('_')[2] == 'indv'))
            })

            sub_pairwise_propensity_dicts = OrderedDict({
                dict_label: propensity_dict for dict_label, propensity_dict in
                self.propensity_dicts.items() if
                ((dict_label.split('_')[0] == surface_label.split('_')[0])
                 and (dict_label.split('_')[2] == 'pairwise'))
            })

            network_fitnesses = OrderedDict()
            for num, G in networks_dict.items():
                propensity_count = 0
                node_propensities_dict = OrderedDict({node: 0 for node in list(G.nodes)})

                for node in list(G.nodes):
                    aa = G.nodes[node]['aa_id']

                    for dict_label, propensity_dict in sub_indv_propensity_dicts.items():
                        node_prop = G.nodes[node][dict_label.split('_')[1]]
                        prop_weight = propensity_dict_weights[dict_label]
                        aa_propensity_scale = propensity_dict[aa]

                        index_1 = (np.abs(aa_propensity_scale[0]-node_prop)).argmin()
                        prop_val_1 = aa_propensity_scale[0][index_1]
                        propensity_1 = aa_propensity_scale[1][index_1]

                        index_2 = ''
                        if prop_val_1 < node_prop:
                            index_2 = index_1 + 1
                        elif prop_val_1 > node_prop:
                            index_2 = index_1 - 1

                        if index_2 == '':
                            propensity == aa_propensity_scale[1][index_1]
                        else:
                            prop_val_2 = aa_propensity_scale[0][index_2]
                            propensity_2 = aa_propensity_scale[1][index_2]

                            weight_1 = abs(prop_val_2 - node_prop)
                            weight_2 = abs(prop_val_1 - node_prop)
                            propensity = (((propensity_1*weight_1) + (propensity_2*weight_2))
                                          / abs(prop_val_2 - prop_val_1))

                        propensity_count += (prop_weight*np.negative(np.log(propensity)))

                        node_propensities_dict[node] += propensity

                    for dict_label, propensity_dict in sub_pairwise_propensity_dicts.items():
                        for edge in G.edges(node):
                            


                network_fitnesses[num] = propensity_count

            fitness_scores[surface_label] = network_fitnesses

        return fitness_scores

    def create_mating_population(sequences_dict, fitness_scores, pop_size):
        # Creates mating population from fittest individuals
        mating_pop_dict = OrderedDict()

        for surface_label, networks_dict in sequences_dict.items():
            network_fitnesses = fitness_scores[surface_label]

            # Sorts networks by their fitness values, from most to least fit
            network_fitnesses = OrderedDict(sorted(
                network_fitnesses.items(), key=itemgetter(1), reverse=False)
            )

            surface_mates = OrderedDict()
            for index, num in enumerate(list(network_fitnesses.keys())):
                while index < pop_size:
                    surface_mates[num] = sequences_dict[num]

            mating_pop_dict[surface_label] = surface_mates

        return mating_pop_dict

    def create_mating_population(sequences_dict, fitness_scores, pop_size):
        # Creates mating population from individuals weighted by their fitness

    def create_mating_population(sequences_dict, fitness_scores, pop_size):
        # Creates mating population from individuals weighted by their fitness rank

    def crossover(mating_pop_dict):
        # Selects pairs of individuals at random from mating population,
        # generates crossover
        for

    def mutate():

    def add_children_to_parents():

    def run_genetic_algorithm():
