#Hyperparameter
def Hyperparameter():
    Hyperparameter={
    'resized_shape': (256,512),
    'test_size': 0.20,
    'n_filter': 8,
    'activation' : 'relu',
    'kernel_initializer' : 'he_normal',
    'learning_rate': 0.5e-3, 
    'decay_steps': 25,
    'decay_rate': 0.25,    
    'beta_1': 0.9,
    'beta_2':0.999,
    'epsilon': 1e-07,
    'Batch_Size': 32,
    'Epoch': 100,
    'Workers': 3,
    'dropout': 0.5,
    'output_activation': 'sigmoid'
    }
    return Hyperparameter