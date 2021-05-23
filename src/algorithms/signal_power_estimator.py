import numpy as np


class SignalPowerEstimator:
    FirstMoment = "First Moment"
    SecondMoment = "Second Moment"


def estimate_signal_power(data, noise_std, noise_mean, method: SignalPowerEstimator):
    if method == SignalPowerEstimator.FirstMoment:
        return _estimate_signal_power_using_first_moment(data, noise_mean)
    elif method == SignalPowerEstimator.SecondMoment:
        return _estimate_signal_power_using_second_moment(data, noise_std, noise_mean)
    else:
        raise Exception("Invalid method for estimating signal power")


def _estimate_signal_power_using_second_moment(data, noise_std, noise_mean):
    data_power = np.sum(np.power(data, 2))
    noise_power = (noise_std ** 2 - noise_mean ** 2) * data.size
    signal_power = data_power - noise_power
    return signal_power


def _estimate_signal_power_using_first_moment(data, noise_mean):
    signal_power = np.sum(data) - noise_mean * data.size
    return signal_power
