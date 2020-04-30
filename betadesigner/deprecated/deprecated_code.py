

# Linear interpolation in 2 dimensions
elif aa_propensity_scale.shape[0] == 3:
      if (    aa_propensity_scale[0][0][0] <= node_val_1 <= aa_propensity_scale[0][0][-1]
          and aa_propensity_scale[1][0][0] <= node_val_2 <= aa_propensity_scale[1][-1][0]
      ):
          x_dimension = aa_propensity_scale[0][0]
          y_dimension = aa_propensity_scale[1][:,0]
          propensities = aa_propensity_scale[2]

          index_x1 = (np.abs(x_dimension-node_val_1)).argmin()
          prop_val_x1 = x_dimension[index_x1]
          index_x2 = find_indices(index_x1, prop_val_x1, node_val_1, x_dimension)
          prop_val_x2 = x_dimension[index_x2]

          index_y1 = (np.abs(y_dimension-node_val_2)).argmin()
          prop_val_y1 = y_dimension[index_y1]
          index_y2 = find_indices(index_y1, prop_val_y1, node_val_2, y_dimension)
          prop_val_y2 = y_dimension[index_y2]

          # Bilinear interpolation
          propensity_x1y1 = propensities[index_y1][index_x1]
          propensity_x1y2 = propensities[index_y2][index_x1]
          propensity_x2y1 = propensities[index_y1][index_x2]
          propensity_x2y2 = propensities[index_y2][index_x2]

          x1_weight = abs(prop_val_x2 - node_val_1) / abs(prop_val_x2 - prop_val_x1)
          x2_weight = abs(prop_val_x1 - node_val_1) / abs(prop_val_x2 - prop_val_x1)
          y1_weight = abs(prop_val_y2 - node_val_2) / abs(prop_val_y2 - prop_val_y1)
          y2_weight = abs(prop_val_y1 - node_val_2) / abs(prop_val_y2 - prop_val_y1)

          propensity_xy1 = (propensity_x1y1*x1_weight) + (propensity_x2y1*x2_weight)
          propensity_xy2 = (propensity_x1y2*x1_weight) + (propensity_x2y2*x2_weight)
          propensity = (propensity_xy1*y1_weight) + (propensity_xy2*y2_weight)

      else:
          propensity = np.nan
          print('Parameter values of input backbone coordinate model '
                'structural features ({}) are outside of the range of '
                'parameter values used to construct propensity scales - node '
                'value 1 = {}, parameter range 1 = {}:{}, node value 2 = {}, '
                'parameter range 2 = {}:{}'.format(
                    dict_label, node_val_1, aa_propensity_scale[0][0][0],
                    aa_propensity_scale[0][0][-1], node_val_2,
                    aa_propensity_scale[1][0][0], aa_propensity_scale[1][-1][0]
              ))





# Test for writing output PDB files. But can't test for this with circleci
# because needs SCWRL4 to pack side chains...
def test_write_output_pdb(self):
    """
    """

    structures_dict, bude_energies_dict = self.gen_structures_dict()

    exp_model_info = pd.DataFrame({
        'structure': ['int_1', 'int_2', 'int_3', 'int_4'],
        'sequence': ['AAA', 'ANR', 'RAN', 'NRN'],
        'energy': [-1.5490, -4.4122, -2.3936, -3.8846],
        'res_id': [{'1': 'ALA', '2': 'ALA', '3': 'ALA'},
                   {'1': 'ALA', '2': 'ASN', '3': 'ARG'},
                   {'1': 'ARG', '2': 'ALA', '3': 'ASN'},
                   {'1': 'ASN', '2': 'ARG', '3': 'ASN'}]
    })

    # Checks expected files have been written
    assert os.path.isdir('Program_output/')
    for model in exp_model_info['structure']:
        assert os.path.isdir('Program_output/{}'.format(model))
        assert os.path.isfile('Program_output/{}/{}.pdb'.format(model, model))
        assert os.path.isfile('Program_output/{}/{}.fasta'.format(model, model))

    # Checks energies calculated by BUDE.
    with open('Program_output/Model_energies.txt', 'r') as f:
        model_energies = f.readlines()
    for line in model_energies:
        model = line.split(':')
        energy = round(float(line.split(':')), 4)
        model_index = exp_model_info['structure'].tolist().index(model)
        self.assertEqual(energy, exp_model_info['energy'][model_index])

    # Checks output FASTA files. Assumes that ISAMBARD is writing output
    # PDB file correctly (since there are tests for this within ISAMBARD)
    for model_index, model in enumerate(exp_model_info['structure'].tolist()):
        with open('Program_output/{}/{}.fasta'.format(model, model), 'r') as f:
            fasta_lines = f.readlines()
        self.assertEqual(fasta_lines[0], '>{}'.format(model))
        self.assertEqual(fasta_lines[1], exp_model_info['sequence'][model_index])


def nansum_axis_1(input_array):
    """
    Overwrites behaviour of np.nansum to return np.nan instead of 0.0 when
    summing an array along axis=1 (i.e. along a row, or the x-axis if you like)
    consisting entirely of NaN values
    """

    output_array = []
    for row in input_array:
        if np.isnan(row).all():
            output_array.append(np.nan)
        else:
            output_array.append(np.nansum(row))
    output_array = np.array(output_array)

    return output_array


def combine_propensities(node_indv_propensities, node_indv_frequencies,
                         sub_dicts, dict_weights, aa_list):
    """
    Sums weighted propensities across structural features considered
    NOTE: must take -ve logarithm of each individual propensity
    score before summing (rather than taking the -ve logarithm of
    the summed propensities)
    """

    node_indv_propensities = np.negative(np.log(node_indv_propensities))

    for index, dict_label in enumerate(list(sub_dicts.keys())):
        dict_weight = dict_weights[dict_label]
        node_indv_propensities[:,index] *= dict_weight
        node_indv_frequencies[:,index] *= dict_weight

    node_indv_propensities = nansum_axis_1(node_indv_propensities)
    node_indv_frequencies = nansum_axis_1(node_indv_frequencies)

    # Removes NaN values
    filtered_aa_list = np.array(copy.deepcopy(aa_list))

    nan_propensity = np.isnan(node_indv_propensities)
    node_indv_propensities = node_indv_propensities[~nan_propensity]
    node_indv_frequencies = node_indv_frequencies[~nan_propensity]
    filtered_aa_list = filtered_aa_list[~nan_propensity]

    nan_frequency = np.isnan(node_indv_frequencies)
    node_indv_propensities = node_indv_propensities[~nan_frequency]
    node_indv_frequencies = node_indv_frequencies[~nan_frequency]
    filtered_aa_list = filtered_aa_list[~nan_frequency]

    return node_indv_propensities, node_indv_frequencies, filtered_aa_list
