from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict
import ray
import time

ray.init(num_cpus=4, ignore_reinit_error=True)

@ray.remote
class LoggingActor(object):
    def __init__(self):
        self.logs = defaultdict(lambda: [])
    
    def log(self, index, message):
        self.logs[index].append(message)
    
    def get_logs(self):
        return dict(self.logs)


assert hasattr(LoggingActor, 'remote'), ('You need to turn LoggingActor into an '
                                         'actor (by using the ray.remote keyword).')

logging_actor = LoggingActor.remote()

# Some checks to make sure this was done correctly.
assert hasattr(logging_actor, 'get_logs')


@ray.remote
def run_experiment(experiment_index, logging_actor):
    for i in range(60):
        time.sleep(1)
        # Push a logging message to the actor.
        logging_actor.log.remote(experiment_index, 'On iteration {}'.format(i))

experiment_ids = [run_experiment.remote(i, logging_actor) for i in range(3)]

for _ in range(5):
    time.sleep(1)
    logs = logging_actor.get_logs.remote()
    logs = ray.get(logs)

    assert isinstance(logs, dict), ("Make sure that you dispatch tasks to the "
                                    "actor using the .remote keyword and get the results using ray.get.")
    print(logs)
