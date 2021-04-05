import numpy as np
from enum import Enum


class SignalPowerEstimator(Enum):
    FirstMoment = 1
    SecondMoment = 2


def estimate_signal_power(y, noise_std, noise_mean, method: SignalPowerEstimator):
    if method == SignalPowerEstimator.FirstMoment:
        return _estimate_signal_power_using_first_moment(y, noise_mean)
    elif method == SignalPowerEstimator.SecondMoment:
        return _estimate_signal_power_using_second_moment(y, noise_std, noise_mean)
    else:
        raise Exception("Invalid method for estimating signal power")


def _estimate_signal_power_using_second_moment(y, noise_std, noise_mean):
    y_power = np.sum(np.power(y, 2))
    noise_power = (noise_std ** 2 - noise_mean ** 2) * y.shape[0]
    signal_power = y_power - noise_power
    return signal_power


def _estimate_signal_power_using_first_moment(y, noise_mean):
    signal_power = np.sum(y) - noise_mean * y.shape[0]
    return signal_power
