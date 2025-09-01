import numpy as np
import torch
from sunbird.emulators import FCN
from .base import BaseObservable
import logging


class GalaxyOverdensityPDF(BaseObservable):
    """
    Class for the Emulator's Mock Challenge galaxy overdensity PDF.
    """
    def __init__(self, phase_correction=False, r=None, **kwargs):
        self.logger = logging.getLogger(self.__class__.__name__)
        if r is not None:
            self.stat_name = 'pdf_r{:d}'.format(r)
        else:
            self.stat_name = 'pdf'
        self.sep_name = 'delta'

        if phase_correction and hasattr(self, 'compute_phase_correction'):
            self.logger.info('Computing phase correction.')
            self.phase_correction = self.compute_phase_correction()

        super().__init__(**kwargs)

    @property
    def lhc_indices(self):
        """
        Indices of the Latin hypercube samples, including variations in cosmology and HOD parameters.
        """
        return {
            'cosmo_idx': list(range(0, 5)) + list(range(13, 14)) + list(range(100, 127)) + list(range(130, 182)),
            'hod_idx': list(range(350)),
        }

    @property
    def test_set_indices(self):
        """
        Indices of the test set samples, including variations in cosmology and HOD parameters.
        """
        return {
            'cosmo_idx': list(range(0, 5)) + list(range(13, 14)),
            'hod_idx': list(range(350)),
        }

    @property
    def coordinates(self):
        """
        Coordinates of the data and model vectors.
        """
        return{
            self.sep_name: self.separation,
        }

    @property
    def coordinates_indices(self):
        """
        Indices of the (flat) coordinates of the data and model vectors.
        """
        return{'bin_idx': list(range(len(self.separation)))}
    
    @property
    def model_fn(self):
        return f'/pscratch/sd/e/epaillas/emc/v1.1/trained_models/best/GalaxyOverdensityPDF/last.ckpt'
        # if '10' in self.stat_name:
        #     return f'/pscratch/sd/m/mpinon/density/trained_models/pdf/r10/cosmo+hod/optuna/asinh/last-v10.ckpt'
        # elif '15' in self.stat_name:
        #     return f'/pscratch/sd/m/mpinon/density/trained_models/pdf/r15/cosmo+hod/optuna/asinh/last-v3.ckpt'
        # elif '20' in self.stat_name:
        #     return f'/pscratch/sd/m/mpinon/density/trained_models/pdf/r20/cosmo+hod/optuna/asinh/last-v2.ckpt'


class GalaxyOverdensityVariance(BaseObservable):
    """
    Class for the Emulator's Mock Challenge galaxy overdensity variance.
    """
    def __init__(self, phase_correction=False, **kwargs):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.stat_name = 'variance'
        self.sep_name = 'r'

        if phase_correction and hasattr(self, 'compute_phase_correction'):
            self.logger.info('Computing phase correction.')
            self.phase_correction = self.compute_phase_correction()

        super().__init__(**kwargs)

    @property
    def lhc_indices(self):
        """
        Indices of the Latin hypercube samples, including variations in cosmology and HOD parameters.
        """
        return {
            'cosmo_idx': list(range(0, 5)) + list(range(13, 14)) + list(range(100, 127)) + list(range(130, 182)),
            'hod_idx': list(range(350)),
        }

    @property
    def test_set_indices(self):
        """
        Indices of the test set samples, including variations in cosmology and HOD parameters.
        """
        return {
            'cosmo_idx': list(range(0, 5)) + list(range(13, 14)),
            'hod_idx': list(range(350)),
        }

    @property
    def coordinates(self):
        """
        Coordinates of the data and model vectors.
        """
        return{
            self.sep_name: self.separation,
        }

    @property
    def coordinates_indices(self):
        """
        Indices of the (flat) coordinates of the data and model vectors.
        """
        return{'bin_idx': list(range(len(self.separation)))}
    
    @property
    def model_fn(self):
        return f'/pscratch/sd/e/epaillas/emc/v1.1/trained_models/best/GalaxyOverdensityPDF/last.ckpt'

    def get_model_prediction(self, x, batch=True, return_tensor=False, no_grad=True):
        """
        Get model prediction for a given x.

        Args:
            x (np.ndarray): Input features.

        Returns:
            np.ndarray: Model prediction.
        """
        if no_grad:
            with torch.no_grad():
                prediction = self.checkpoint.get_prediction(torch.Tensor(x))
        else:
            with torch.enable_grad():
                prediction = self.checkpoint.get_prediction(torch.Tensor(x))
        # take the variance of each distribution
        delta = np.load('/pscratch/sd/m/mpinon/acm/training_sets/cosmo+hod/pdf.npy', allow_pickle=True).item()['delta']
        prediction_r10 = variance_from_pdf(prediction[..., :125], torch.Tensor(delta[:125])) # R = 10
        prediction_r15 = variance_from_pdf(prediction[..., 125:199], torch.Tensor(delta[125:199])) # R = 15
        prediction_r20 = variance_from_pdf(prediction[..., 199:], torch.Tensor(delta[199:])) # R = 20
        prediction = torch.cat([prediction_r10, prediction_r15, prediction_r20], dim=-1)
        if return_tensor:
            return prediction
        prediction = prediction.numpy()
        if hasattr(self, 'phase_correction'):
            prediction = self.apply_phase_correction(prediction)
        coords = self.coordinates_indices if self.select_indices else self.coordinates
        coords_shape = tuple(len(v) for k, v in coords.items())
        from sunbird.data.data_utils import convert_to_summary
        if len(prediction.shape) > 1: # batch query
            dimensions = ["batch"] + list(coords.keys())
            coords["batch"] = range(len(prediction))
            prediction = prediction.reshape((len(prediction), *coords_shape))
            return convert_to_summary(
                data=prediction, dimensions=dimensions, coords=coords,
                select_filters=self.select_filters, slice_filters=self.slice_filters
            ).values.reshape(len(prediction), -1)
        else:
            coords_shape = tuple(len(v) for k, v in coords.items())
            prediction = prediction.reshape(coords_shape)
            dimensions = list(coords.keys())
            return convert_to_summary(
                data=prediction, dimensions=dimensions, coords=coords,
                select_filters=self.select_filters, slice_filters=self.slice_filters
            ).values.reshape(-1)

def variance_from_pdf(pdf, x):
    """Compute variance from normalized pdf values"""
    mean = torch.trapezoid(pdf * x, x=x, dim=-1)
    var = torch.trapezoid((x[None, ...] - mean[..., None])**2 * pdf, x=x, dim=-1)
    if len(var) > 1:
        var = var[..., None]
    return var