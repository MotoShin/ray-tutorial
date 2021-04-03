from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import ray
import time
from bs4 import BeautifulSoup
import requests

import pandas as pd

ray.init(num_cpus=4, ignore_reinit_error=True)

@ray.remote
def fetch_commits(repo):
    url = 'https://github.com/{}/commits/master'.format(repo)
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    results = []
    for commit_elt in soup.find_all('li', class_='commit'):
        title = commit_elt.find_all('a', class_='message')[0].attrs.get('aria-label').split('\n')[0]
        link_elts = commit_elt.find_all('a', class_='issue-link')
        link = None
        for le in link_elts:
            if 'issue' in le.attrs['href'].lower():
                link = le.attrs['href']
        results.append(dict(title=title, link=link))

    df = pd.DataFrame(results)
    df['repository'] = repo
    return df

# Sleep a little to improve the accuracy of the timing measurements below.
time.sleep(2.0)
start = time.time()
repos = ["ray-project/ray", "modin-project/modin", "tensorflow/tensorflow", "apache/arrow"]
results = []
for repo in repos:
    df = fetch_commits.remote(repo)
    results.append(df)
    
df = pd.concat(ray.get(results), sort=False)
duration = time.time() - start
print("Constructing the dataframe took {} seconds.".format(duration))
