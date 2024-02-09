from jax import random

def key_iter(seed=0):
    key = random.PRNGKey(seed)
    while True:
        key, key_ = random.split(key)
        yield key_
