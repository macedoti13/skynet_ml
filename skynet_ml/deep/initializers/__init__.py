from skynet_ml.deep.initializers.random import Random
from skynet_ml.deep.initializers.he import He
from skynet_ml.deep.initializers.xavier import Xavier


initializers_map = {
    "random": Random(),
    "he": He(),
    "xavier": Xavier()
}

activation_to_initializer_map = {
    'sigmoid': 'xavier',
    'tanh': 'xavier',
    'relu': 'he',
    'softmax': 'xavier', 
    'linear': 'random' 
}