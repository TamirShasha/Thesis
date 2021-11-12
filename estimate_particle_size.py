import click
import os
import warnings
import numpy as np

from src.constants import ROOT_DIR
from src.experiments.experiments_2d import Experiment2D, EstimationMethod
from src.utils.micrograph import Micrograph, NoiseNormalizationMethod

warnings.filterwarnings('ignore')

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.group(chain=False, context_settings=CONTEXT_SETTINGS)
@click.option('--debug/--no-debug', default=False, help="Default is --no-debug.")
@click.option('-v', '--verbosity', default=0, help='Verbosity level (0-3).')
def simple_cli(debug, verbosity):
    return


@simple_cli.command('estimate_particle_size', short_help='Estimates particle size')
@click.option('--name', type=str)
@click.option('--mrc_path', type=str)
@click.option('--length_options', nargs=3, type=int)
@click.option('--estimation_method', type=click.Choice(['vws', 'curves'], case_sensitive=False), default='vws')
@click.option('--num_of_instances_range', type=(int, int), default=(20, 100))
@click.option('--normalize_noise_method', default='simple',
              type=click.Choice(['none', 'simple', 'whitening'], case_sensitive=False))
@click.option('--noise_mean', type=int, default=0)
@click.option('--noise_std', type=int, default=1)
@click.option('--down_sample_size', type=int, default=-1)
@click.option('--filter_basis_size', type=int, default=20)
@click.option('--particles_margin', type=float, default=0.01)
@click.option('--estimate_locations_and_num_of_instances', type=bool, default=False)
@click.option('--plot', type=bool, default=False)
@click.option('--save', type=bool, default=True)
@click.option('--save_dir', type=str, default=os.path.join(ROOT_DIR, f'src/experiments/plots/'))
@click.option('--logs', type=bool, default=True)
@click.option('--random_seed', type=int, default=500)
def estimate(name,
             mrc_path,
             length_options,
             estimation_method,
             num_of_instances_range,
             normalize_noise_method,
             noise_mean,
             noise_std,
             down_sample_size,
             filter_basis_size,
             particles_margin,
             estimate_locations_and_num_of_instances,
             plot,
             save,
             save_dir,
             logs,
             random_seed):
    # load micrograph
    normalize_noise_method = normalize_noise_method.lower()
    if normalize_noise_method == 'none':
        normalize_noise_method = NoiseNormalizationMethod.NoNormalization
    elif normalize_noise_method == 'simple':
        normalize_noise_method = NoiseNormalizationMethod.Simple
    else:
        normalize_noise_method = NoiseNormalizationMethod.Whitening
    micrograph = Micrograph(mrc_path,
                            downsample=down_sample_size,
                            load_micrograph=True,
                            noise_normalization_method=normalize_noise_method,
                            noise_mean=noise_mean,
                            noise_std=noise_std)

    # build length options
    start, end, step = length_options
    length_options = np.arange(start, end + 1, step)

    # estimation method
    estimation_method = estimation_method.lower()
    if estimation_method == 'vws':
        estimation_method = EstimationMethod.VeryWellSeparated
    else:
        estimation_method = EstimationMethod.Curves

    experiment = Experiment2D(name=name,
                              mrc=micrograph,
                              length_options=length_options,
                              filter_basis_size=filter_basis_size,
                              down_sample_size=down_sample_size,
                              num_of_instances_range=num_of_instances_range,
                              estimation_method=estimation_method,
                              estimate_locations_and_num_of_instances=estimate_locations_and_num_of_instances,
                              particles_margin=particles_margin,
                              plot=plot,
                              save=save,
                              save_dir=save_dir,
                              logs=logs)
    experiment.run()


if __name__ == "__main__":
    estimate(['--name', 'Tamir',
              '--mrc_path', 'C:\\Users\\tamir\\Desktop\\Thesis\\simulated_data\\1000x1000_33occ_sphere_80_N_0_10.npy',
              '--length_options', '20', '30', '10',
              '--plot', 'True'])
    # estimate()
