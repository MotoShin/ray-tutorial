from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import ray
import time

ray.init(num_cpus=9, ignore_reinit_error=True)

@ray.remote
def compute_gradient(data, current_model):
    time.sleep(0.03)
    return 1

@ray.remote
def train_model(hyperparameters):
    current_model = 0
    # Iteratively improve the current model. This outer loop cannot be parallelized.
    for _ in range(10):
        # EXERCISE: Parallelize the list comprehension in the line below. After you
        # turn "compute_gradient" into a remote function, you will need to call it
        # with ".remote". The results must be retrieved with "ray.get" before "sum"
        # is called.
        gradients = ray.get([compute_gradient.remote(j, current_model) for j in range(2)])
        total_gradient = sum(gradients)
        current_model += total_gradient

    return current_model

assert hasattr(compute_gradient, 'remote'), 'compute_gradient must be a remote function'
assert hasattr(train_model, 'remote'), 'train_model must be a remote function'

time.sleep(2.0)
start_time = time.time()

# Run some hyperparaameter experiments.
results = []
for hyperparameters in [{'learning_rate': 1e-1, 'batch_size': 100},
                        {'learning_rate': 1e-2, 'batch_size': 100},
                        {'learning_rate': 1e-3, 'batch_size': 100}]:
    results.append(train_model.remote(hyperparameters))

# EXERCISE: Once you've turned "results" into a list of Ray ObjectIDs
# by calling train_model.remote, you will need to turn "results" back
# into a list of integers, e.g., by doing "results = ray.get(results)".
results = ray.get(results)

end_time = time.time()
duration = end_time - start_time

assert all([isinstance(x, int) for x in results]), \
    'Looks like "results" is {}. You may have forgotten to call ray.get.'.format(results)

assert results == [20, 20, 20]
assert duration < 0.5, ('The experiments ran in {} seconds. This is too '
                         'slow.'.format(duration))
assert duration > 0.3, ('The experiments ran in {} seconds. This is too '
                        'fast.'.format(duration))

print('Success! The example took {} seconds.'.format(duration))

ray.timeline(filename="timeline03.json")
