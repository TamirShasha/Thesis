import sys

from src.experiments.experiments_2d import EstimationMethod
from src.algorithms.length_estimator_2d_very_well_separated import LengthEstimator2DVeryWellSeparated
from src.algorithms.length_estimator_2d_curves_method import LengthEstimator2DCurvesMethod

arguments = sys.argv[1:]
mrc_path = arguments[0]

if len(arguments) == 1:
    method_type = EstimationMethod.VeryWellSeparated
else:
    method_type = EstimationMethod.VeryWellSeparated if arguments[1] == 'vws' else EstimationMethod.Curves

