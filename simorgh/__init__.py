"""SIMORGH — amortized, calibrated atmospheric retrieval for uniform
sub-Neptune surveys, with a population-inference contract.

Core objects:
  Spectrum            simorgh.spectra
  BoxPrior            simorgh.priors (subneptune_prior)
  ToySubNeptune       simorgh.simulate.toy
  AmortizedPosterior  simorgh.models.npe
  train_npe           simorgh.train
  SBC / TARP          simorgh.diagnostics
  hyper_log_likelihood simorgh.population.reweight
  DensityRatioGate    simorgh.gate.ood
"""

from .priors import BoxPrior, subneptune_prior
from .spectra import Spectrum

__version__ = "0.1.0"
__all__ = ["BoxPrior", "subneptune_prior", "Spectrum", "__version__"]
