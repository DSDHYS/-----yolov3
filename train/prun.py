from import tf
model_path = 'model_data/train.pth'
reader = tf.train.NewCheckpointReader(model_path)
all_variables = reader.get_variable_to_shape_map()
