import datetime
import os
import logging
import time

def set_logger(args):
    """Write logs to checkpoint and console"""
    log_name = 'train_' + time.strftime('%Y_%m_%d') + '_' + time.strftime('%H:%M:%S') + '.log'
    log_file = os.path.join(args.save_path, log_name)

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

def log_metrics(mode, step, metrics):
    """Print the evaluation logs"""
    for metric in metrics:
        logging.info('%s %s at epoch %d: %f' % (mode, metric, step, metrics[metric]))