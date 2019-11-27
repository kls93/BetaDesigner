

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
