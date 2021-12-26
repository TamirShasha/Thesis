import click
import os
import warnings
import numpy as np
import logging

from src.constants import ROOT_DIR
from src.experiments.experiments_2d import Experiment2D, EstimationMethod
from src.utils.micrograph import Micrograph, NoiseNormalizationMethod
from multiprocessing import Process

warnings.filterwarnings('ignore')

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.group(chain=False, context_settings=CONTEXT_SETTINGS)
@click.option('--debug/--no-debug', default=False, help="Default is --no-debug.")
@click.option('-v', '--verbosity', default=0, help='Verbosity level (0-3).')
def simple_cli(debug, verbosity):
    return


@simple_cli.command('estimate_particle_size', short_help='Estimates particle size')
@click.option('--mrc_path', type=str)
@click.option('--signal_length_by_percentage', nargs=3, type=int, default=None)
@click.option('--estimation_method', type=click.Choice(['vws', 'curves'], case_sensitive=False), default='vws')
@click.option('--num_of_instances_range', type=(int, int), default=(50, 100))
@click.option('--noise_params', type=(float, float), default=(None, None))
@click.option('--down_sample_size', type=int, default=-1)
@click.option('--filter_basis_size', type=int, default=7)
@click.option('--particles_margin', type=float, default=0.03)
@click.option('--save_statistics', is_flag=True)
@click.option('--plot', is_flag=True)
@click.option('--save', is_flag=True)
@click.option('--save_dir', type=str, default=os.path.join(ROOT_DIR, f'src/experiments/plots/'))
@click.option('--logs', is_flag=True)
@click.option('--logs-debug', is_flag=False)
@click.option('--random_seed', type=int, default=500)
def estimate(mrc_path,
             signal_length_by_percentage,
             estimation_method,
             num_of_instances_range,
             noise_params,
             down_sample_size,
             filter_basis_size,
             particles_margin,
             save_statistics,
             plot,
             save,
             save_dir,
             logs,
             logs_debug,
             random_seed):
    np.random.seed(random_seed)

    # build length options
    if signal_length_by_percentage is not None:
        start, end, step = signal_length_by_percentage
        signal_length_by_percentage = np.arange(start, end + 1, step)
    else:
        signal_length_by_percentage = [1.5, 2, 3, 4, 6, 8, 10]

    # estimation method
    estimation_method = estimation_method.lower()
    if estimation_method == 'vws':
        estimation_method = EstimationMethod.VeryWellSeparated
    else:
        estimation_method = EstimationMethod.Curves

    # Should estimate noise parameters
    estimate_noise_parameters = noise_params == (None, None)

    log_level = logging.NOTSET
    if logs:
        log_level = logging.INFO
    if logs_debug:
        log_level = logging.DEBUG

    file_name, file_ext = os.path.splitext(mrc_path)
    if file_ext == '.txt':
        file = open(mrc_path, 'r')
        paths = [path.rstrip() for path in file.readlines()]
    else:
        paths = [mrc_path]

    for path in paths:
        if path == '':
            continue

        name = os.path.basename(path)
        if '.' in name:
            name = name.split('.')[0]
        mrc_save_dir = os.path.join(save_dir, name)

        micrograph = Micrograph(path,
                                downsample=down_sample_size,
                                load_micrograph=True,
                                noise_mean=noise_params[0],
                                noise_std=noise_params[1])

        experiment = Experiment2D(name=name,
                                  mrc=micrograph,
                                  signal_length_by_percentage=signal_length_by_percentage,
                                  filter_basis_size=filter_basis_size,
                                  down_sample_size=down_sample_size,
                                  num_of_instances_range=num_of_instances_range,
                                  estimation_method=estimation_method,
                                  estimate_noise=estimate_noise_parameters,
                                  save_statistics=save_statistics,
                                  particles_margin=particles_margin,
                                  plot=plot,
                                  save=save,
                                  log_level=log_level,
                                  save_dir=mrc_save_dir)
        # experiment.run()
        Process(target=experiment.run).start()


if __name__ == "__main__":
    # estimate(['--mrc_path', r'C:\Users\tamir\Desktop\Thesis\data\paths.txt',
    #           '--down_sample_size', '1000',
    #           '--plot'])
    estimate()
