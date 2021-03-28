from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ray
import time

ray.init(num_cpus=4, ignore_reinit_error=True)

@ray.remote
def slow_function(i):
    time.sleep(1)
    return i

time.sleep(10.0)
start_time = time.time()

results = []
for i in range(4):
    result = slow_function.remote(i)
    results.append(ray.get(result))

end_time = time.time()
duration = end_time - start_time

print('The results are {}. This took {} seconds. Run the next cell to see '
      'if the exercise was done correctly.'.format(results, duration))

assert results == [0, 1, 2, 3], 'Did you remember to call ray.get?'
assert duration < 1.1, ('The loop took {} seconds. This is too slow.'
                        .format(duration))
assert duration > 1, ('The loop took {} seconds. This is too fast.'
                      .format(duration))

print('Success! The example took {} seconds.'.format(duration))

ray.timeline(filename="timeline01.json")
