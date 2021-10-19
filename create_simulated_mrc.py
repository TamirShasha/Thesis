import click
import os
import warnings
import numpy as np
import pathlib

from src.constants import ROOT_DIR
from src.experiments.data_simulator_2d import DataSimulator2D, Shapes2D

warnings.filterwarnings('ignore')

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.group(chain=False, context_settings=CONTEXT_SETTINGS)
@click.option('--debug/--no-debug', default=False, help="Default is --no-debug.")
@click.option('-v', '--verbosity', default=0, help='Verbosity level (0-3).')
def simple_cli(debug, verbosity):
    return


@simple_cli.command('create_simulated_mrc', short_help='Creates Simulated Micrograph')
@click.option('--name', type=str)
@click.option('--size', type=int, default=1000)
@click.option('--signal_size', type=int, default=80)
@click.option('--signal_power', type=float, default=1)
@click.option('--signal_fraction', type=float, default=1 / 6)
@click.option('--signal_shape', type=click.Choice(['disk', 'sphere'], case_sensitive=False), default='sphere')
@click.option('--noise_mean', type=int, default=0)
@click.option('--noise_std', type=int, default=10)
@click.option('--apply_ctf', type=bool, default=False)
@click.option('--save_to', type=str, default=os.path.join(ROOT_DIR, f'../simulated_data/'))
def create_simulated_mrc(name, size, signal_size, signal_power, signal_fraction, signal_shape, noise_mean, noise_std,
                         apply_ctf, save_to):
    pathlib.Path(save_to).mkdir(parents=True, exist_ok=True)

    signal_gen = Shapes2D.sphere
    if signal_shape == 'disk':
        signal_gen = Shapes2D.disk

    simulator = DataSimulator2D(rows=size,
                                columns=size,
                                signal_length=signal_size,
                                signal_power=signal_power,
                                signal_fraction=signal_fraction,
                                signal_gen=signal_gen,
                                noise_std=noise_std,
                                noise_mean=noise_mean,
                                apply_ctf=apply_ctf)

    mrc = simulator.simulate()
    file_name = f'{name}_{size}x{size}_{simulator.occurrences}occ_{signal_shape}_{signal_size}_N_{noise_mean}_{noise_std}'
    file_path = os.path.join(save_to, file_name)
    np.save(file_path, mrc)


if __name__ == "__main__":
    create_simulated_mrc()
