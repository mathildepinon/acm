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
        self.sep_name = 'bin_idx'

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

    