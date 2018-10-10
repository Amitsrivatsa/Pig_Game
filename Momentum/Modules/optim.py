import numpy as np

def adam(w, dw, config=None):
 

 
    if config is None: 
        config = {}

    
    config.setdefault('learning_rate', 1e-2)

    config.setdefault('momentum', 0.9)

    v = config.get('velocity', np.zeros_like(w))

  

    next_w = None

  #############################################################################

  # TODO: Implement the momentum update formula. Store the updated value in   #

  # the next_w variable. You should also use and update the velocity v.       #

  #############################################################################

    mu = config['momentum']

    lr = config['learning_rate'] 

    v = mu * v - lr * dw

    next_w = w + v

  #############################################################################

  #                             END OF YOUR CODE                              #

  #############################################################################

    config['velocity'] = v



    return next_w, config

