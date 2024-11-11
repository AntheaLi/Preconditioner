heat_train_config = {
    # 'num_diffusivities': 5, 
    # 'diffusivities': [0.2, 0.4, 0.6, 0.7],
    # 'diffusivities': [0.5, 1.5, 5.0, 10.0],
    # 'diffusivities': [1.5, 5.0, 10.0],
    # 'diffusivities': [1.5, 5.0, 20.0, 50.0, 100.0],
    'diffusivities': [100.0], 
    # 'diffusivities': [1.5],
    # 'diffusivity_cap':1.5, 
    'name': 'circle_low_res',
    'time_step':1e-2, 
    'num_time_steps':100, 
    'total_data_num': 4800, 
    'data_features':None, 
    'num_inits':1,
    'split': 'train'
}

heat_test_config = {
    'diffusivities':[1.5],
    'name': 'circle_low_res',
    'time_step':1e-2, 
    'num_time_steps':100, 
    'total_data_num':100,
    'data_features':None, 
    'num_inits':1,
    'split': 'test'
}

inviscidflow_train_config={
    'num_densities':None, 
    'time_step':1e-2,
    'name': 'cyn_low_res',
    'num_time_steps':100, 
    'total_data_num': 4800, 
    'data_features':None, 
    'num_inits':1, 
    'split': 'train',
    'generation': False,
    'density': [0.001]
}

inviscidflow_test_config={
    'num_densities':None, 
    'time_step':1e-2,
    'name': 'cyn_low_res',
    'num_time_steps':100, 
    'total_data_num':100,
    'data_features':None, 
    'num_inits':1, 
    'split': 'test',
    'generation':False,
    'density': [0.001]
}



wave_train_config={
    'num_speed':None, 
    'time_step':2e-3,
    'name': 'circle_low_res',
    'num_time_steps':100, 
    'total_data_num': 3000, 
    'data_features':None, 
    'num_inits':1, 
    'split': 'train',
    'generation': False,
    'speed': [0.1]
}

wave_test_config={
    'num_speed':None, 
    'time_step':2e-3, # DO NOT CHANGE THIS!
    'name': 'circle_low_res',
    'num_time_steps':100, 
    'total_data_num':100,
    'data_features':None, 
    'num_inits':1, 
    'split': 'test',
    'generation':False,
    'speed': [0.1]
}

poisson3d_train_config={
    'time_step':1e-2,
    'name': 'armadillo_low_res_init1',
    'num_time_steps':100, 
    'total_data_num': 3000, 
    'data_features':None, 
    'num_inits':1, 
    'split': 'train',
    'generation': False,
}

poisson3d_test_config={
    'time_step':1e-2, # DO NOT CHANGE THIS!
    'name': 'armadillo_low_res_init1',
    'num_time_steps':100, 
    'total_data_num':100,
    'data_features':None, 
    'num_inits':1, 
    'split': 'test',
    'generation':False,
}
