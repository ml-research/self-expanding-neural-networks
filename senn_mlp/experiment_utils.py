import os
import confuse
import argparse
from rtpt import RTPT
from datetime import datetime
from tensorflow import summary

def get_rtpt(name, max_iter=1):
    rtpt = RTPT(name_initials='RM', experiment_name=name, max_iterations=max_iter)
    rtpt.start()
    return rtpt

def get_cfg(script_name, args=[]):
    """args: [('--argname', type, 'dest.var'), ...]"""
    
    cfg = confuse.Configuration('experiment')
    cfg.set_file(f'./{script_name}/default_config.yaml', base_for_paths=True)
    OVERRIDE = './config.yaml'
    if os.path.isfile(OVERRIDE):
        cfg.set_file(OVERRIDE, base_for_paths=True)
    parser = argparse.ArgumentParser()    
    base_args = [
        ('--name', str, 'meta.name'),
        ('--seed', int, 'meta.seed'),
        ('--epochs', int, 'opt.max_epochs'),
        ('--thresh', str, 'evo.thresh')
    ]
    for n, t, d in base_args + args:
        parser.add_argument(n, type=t, dest=d)    
    cfg.set_args(parser.parse_args(), dots=True)
    return cfg
    
def set_writer(cfg):
    now = datetime.now()
    exp_name = cfg['meta']['name'].get(f"{now.date()}_{now.time()}")
    logdir = cfg['meta']['logdir'].get()
    writer = summary.create_file_writer(f"{logdir}/{exp_name}")
    writer.set_as_default()
    return writer
