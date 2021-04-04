from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import ray
import time

ray.init(num_cpus=4, ignore_reinit_error=True)

@ray.remote
class Foo(object):
    def __init__(self):
        self.counter = 0

    def reset(self):
        self.counter = 0

    def increment(self):
        time.sleep(0.5)
        self.counter += 1
        return self.counter

assert hasattr(Foo, 'remote'), 'You need to turn "Foo" into an actor with @ray.remote.'

# Create two Foo objects.
f1 = Foo.remote()
f2 = Foo.remote()

# Sleep a little to improve the accuracy of the timing measurements below.
time.sleep(2.0)
start_time = time.time()

# Reset the actor state so that we can run this cell multiple times without
# changing the results.
f1.reset.remote()
f2.reset.remote()

# We want to parallelize this code. However, it is not straightforward to
# make "increment" a remote function, because state is shared (the value of
# "self.counter") between subsequent calls to "increment". In this case, it
# makes sense to use actors.
results = []
for _ in range(5):
    results.append(f1.increment.remote())
    results.append(f2.increment.remote())
results = ray.get(results)

end_time = time.time()
duration = end_time - start_time

assert not any([isinstance(result, ray.ObjectID) for result in results]), 'Looks like "results" is {}. You may have forgotten to call ray.get.'.format(results)

assert results == [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]

assert duration < 3, ('The experiments ran in {} seconds. This is too '
                      'slow.'.format(duration))
assert duration > 2.5, ('The experiments ran in {} seconds. This is too '
                        'fast.'.format(duration))

print('Success! The example took {} seconds.'.format(duration))
