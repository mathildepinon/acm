from abc import ABC, abstractmethod
from sunbird.data.data_utils import convert_to_summary
from .paths import emc_paths
from pathlib import Path
import numpy as np
import torch


class BaseModel(ABC):
    """
    Base class for the Emulator's Mock Challenge models.
    """
    def __init__(self, observable):
        self.observable = observable

    def get_prediction(self, x, return_tensor=False, no_grad=True,):
        return self.observable.get_model_prediction(x, return_tensor=return_tensor, no_grad=no_grad)


class BaseObservable(ABC):
    """
    Base class for the Emulator's Mock Challenge observables.
    """
    def __init__(
        self,
        select_mocks: dict = {},
        select_indices: list = [],
        select_coordinates: dict = {},
        slice_coordinates: dict = {},
        train: bool = False,
    ):
        if select_indices:
            assert type(select_indices) == list, "select_indices should be a list of indices"
        self.select_indices = {'bin_idx': select_indices} if select_indices else {}
        self.select_mocks = select_mocks
        self.select_coordinates = select_coordinates
        self.slice_coordinates = slice_coordinates

        if bool((self.select_coordinates or self.slice_coordinates) and self.select_indices):
            raise ValueError("You can only select either coordinates or indices, not both.")
        self.select_filters = {}

        # Add each dictionary only if it's not None
        if self.select_mocks is not None:
            self.select_filters.update(self.select_mocks)
        if self.select_coordinates is not None:
            self.select_filters.update(self.select_coordinates)
        if self.select_indices is not None:
            self.select_filters.update(self.select_indices)

        self.slice_filters = self.slice_coordinates

        if not train:
            self.checkpoint = self.load_model_checkpoint()
        self.separation = self.load_separation()
        
    @property
    def nparams(self):
        """
        Number of parameters in the Latin hypercube samples
        (equal to the number of input parameters for the emulators.
        """
        return len(self.lhc_x_names)

    def lhc_fname(self):
        """
        File containing Latin hypercube samples.
        """
        lhc_dir = Path(emc_paths['lhc_dir'])
        return lhc_dir / f'{self.stat_name}.npy'
        #data_dir = f'/pscratch/sd/m/mpinon/density/training_sets/cosmo+hod/ph000/seed0/use'
        #return Path(data_dir) / f'{self.stat_name}.npy'

    def emulator_error_fname(self):
        """
        File containing the emulator error.
        """
        emulator_error_dir = Path(emc_paths['emulator_error_dir'])
        #emulator_error_dir =  Path('/pscratch/sd/m/mpinon/density/emc/emulator_error/')
        return emulator_error_dir / f'{self.stat_name}.npy'

    def small_box_fname(self):
        """
        File containing the output features from the small AbacusSummit box.
        """
        covariance_dir = Path(emc_paths['covariance_dir']) 
        #covariance_dir = Path('/pscratch/sd/m/mpinon/density/cov/use/')
        return covariance_dir / f'{self.stat_name}.npy'

    def diffsky_fname(self, phase_idx, sampling):
        """
        File containing the measurements from Diffsky simulations.
        """
        base_dir = Path(emc_paths['diffsky_dir']) 
        #base_dir = Path('/pscratch/sd/m/mpinon/density/diffsky/use')
        diffsky_dir = base_dir / f'galsampled_67120_fixedAmp_{phase_idx:03}_{sampling}_v0.3'
        return diffsky_dir / f'{self.stat_name}.npy'

    @property
    @abstractmethod
    def model_fn(self):
        pass

    @property
    def lhc_x(self):
        """
        Latin hypercube of input features (cosmological and/or HOD parameters)
        """
        fn = self.lhc_fname()
        lhc_x = np.load(fn, allow_pickle=True).item()['lhc_x']
        coords = self.lhc_indices
        coords.update({'param_idx': list(range(self.nparams))})
        coords_shape = tuple(len(v) for k, v in coords.items())
        dimensions = list(coords.keys())
        lhc_x = lhc_x.reshape(*coords_shape)
        return convert_to_summary(
            data=lhc_x, dimensions=dimensions, coords=coords,
            select_filters=self.select_filters, slice_filters=self.slice_filters
        ).values.reshape(-1)

    @property
    def lhc_x_names(self):
        """
        Names of the input features (cosmological and/or HOD parameters)
        """
        fn = self.lhc_fname()
        return np.load(fn, allow_pickle=True).item()['lhc_x_names']
    
    @property
    def lhc_y(self):
        """
        Latin hypercube of output features (tpcf, power spectrum, etc).
        """
        fn = self.lhc_fname()
        lhc_y = np.load(fn, allow_pickle=True).item()['lhc_y']
        coords = self.lhc_indices
        if self.select_indices:
            coords.update(self.coordinates_indices)
        else:
            coords.update(self.coordinates)
        coords_shape = tuple(len(v) for k, v in coords.items())
        dimensions = list(coords.keys())
        lhc_y = lhc_y.reshape(*coords_shape)
        return convert_to_summary(
            data=lhc_y, dimensions=dimensions, coords=coords,
            select_filters=self.select_filters, slice_filters=self.slice_filters
        ).values.reshape(-1)

    @property
    def small_box_indices(self):
        """
        Indices of the covariance samples, including variations in phase and HOD parameters.
        """
        return {
            'phase_idx': range(1786),
        }

    @property
    def small_box_y(self):
        """
        Output features from the small AbacusSummit box for covariance
        estimation.
        """
        #TODO: Get rid of the outlier stuff after void measurements are fixed
        fn = self.small_box_fname()
        small_box_y = np.load(fn, allow_pickle=True).item()['cov_y']
        coords = self.small_box_indices
        if self.select_indices:
            coords.update(self.coordinates_indices)
        else:
            coords.update(self.coordinates)
        coords_shape = tuple(len(v) for k, v in coords.items())
        dimensions = list(coords.keys())
        small_box_y = small_box_y.reshape(*coords_shape)
        small_box_y = convert_to_summary(
            data=small_box_y, dimensions=dimensions, coords=coords,
            select_filters=self.select_filters, slice_filters=self.slice_filters
        )
        small_box_y = small_box_y.sel(phase_idx= [i for i in list(range(1786))])
        return small_box_y.values.reshape(len(small_box_y), -1)

    def diffsky_y(self, phase_idx=1, sampling='mass_conc'):
        """
        Measurements from Diffsky simulations.
        """
        fn = self.diffsky_fname(phase_idx=phase_idx, sampling=sampling)
        diffsky_y = np.load(fn, allow_pickle=True).item()['diffsky_y']
        coords = self.coordinates_indices if self.select_indices else self.coordinates
        coords_shape = tuple(len(v) for k, v in coords.items())
        dimensions = list(coords.keys())
        diffsky_y = diffsky_y.reshape(*coords_shape)
        return convert_to_summary(
            data=diffsky_y, dimensions=dimensions, coords=coords,
            select_filters=self.select_filters, slice_filters=self.slice_filters
        ).values.reshape(-1)

    # def test_set_indices(self):


    @property
    def model(self):
        """
        Theory model of the observable.
        """
        return BaseModel(self)

    def load_model_checkpoint(self):
        """
        Load trained theory model from checkpoint file.
        """
        from sunbird.emulators import FCN
        try:
            model = FCN.load_from_checkpoint(self.model_fn, strict=True)
        except:
            raise ValueError(
            f"Model {self.model_fn} not found. If you are training a "
            "new model, please set the `train` argument to True at "
            "class initialization."
            )
        model = model.eval().to('cpu')
        if self.stat_name.startswith('minkowski'):
            from sunbird.data.transforms_array import WeiLiuInputTransform, WeiLiuOutputTransForm
            model.transform_output = WeiLiuOutputTransForm()
            model.transform_input = WeiLiuInputTransform()
        return model

    def get_emulator_error(self, method: ['median', 'std'] = 'median'):
        """
        Emulator error.
        """
        res = self.get_model_residuals()  # pred_y - true_y
        if method == 'median':
            error = np.median(np.abs(res), axis=0)
            return error
        elif method == 'std':
            error = np.std(res, axis=0)
            return error
        # fn = self.emulator_error_fname()
        # error = np.load(fn, allow_pickle=True).item()['emulator_error']
        # coords = self.coordinates_indices if self.select_indices else self.coordinates
        # coords_shape = tuple(len(v) for k, v in coords.items())
        # dimensions = list(self.coords.keys())
        # error = error.reshape(*coords_shape)
        # return convert_to_summary(
        #     data=error, dimensions=dimensions, coords=coords,
        #     select_filters=self.select_filters, slice_filters=self.slice_filters
        # ).values.reshape(-1)

    def get_chi2(self, residuals):
        covariance_data = self.get_covariance_matrix()
        precision_data = np.linalg.inv(covariance_data)
        chi2 = np.einsum('ij,jk,ik->i', residuals, precision_data, residuals)
        return np.sqrt(chi2) / residuals.shape[-1]

    def get_emulator_error_matrix(self, select_mocks=None, diagonalize=True,
        method: ['median', 'std'] = 'median'):
        """
        Get the covariance matrix of the emulator error.
        """
        res =  self.get_model_residuals(select_mocks=select_mocks)  # pred_y - true_y
        if method == 'median':
            error = np.median(np.abs(res), axis=0)
            if diagonalize is False:
                raise ValueError('Method "median" only works with diagonalize=True')
            return np.diag(error ** 2)
        elif method == 'std':
            cov = np.cov(res.T)
        elif method == 'std_chi2_5sigma':
            data_residuals = self.get_model_residuals_data(select_mocks=select_mocks)
            chi2 = self.get_chi2(residuals=data_residuals,)
            mask = chi2 < 5.
            cov = np.cov(res[mask].T)
        elif method == 'std_chi2_weighted':
            data_residuals = self.get_model_residuals_data(select_mocks=select_mocks)
            chi2 = self.get_chi2(residuals=data_residuals,)
            weights = 1.0 / (1.0 + chi2)
            weights = weights / np.sum(weights)
            cov = np.cov(res.T, aweights=weights)
        if diagonalize is False:
            return cov
        else:
            return np.diag(np.diag(cov))

    def get_model_residuals(self, select_mocks=None):
        """
        Calculate the emulator error from the test set of the Latin hypercube. 
        
        We make a new instance of the class with the test set filters and
        compare the emulator prediction to the true values.

        Args:
            method (str): Method to compute the error. Options are 'mae' for
                mean absolute error and 'cov' to get the emulator error covariance,
                which also accounts for correlations among bins.

        Returns:
            np.ndarray: Emulator error.
        """
        import numpy as np
        if select_mocks is None:
            select_mocks = self.test_set_indices
        if self.select_indices:
            select_indices = self.select_indices['bin_idx']
        else:
            select_indices = {}
        if hasattr(self, 'r'):
            kwargs = {'r': self.r}
        else:
            kwargs = {}
        observable = self.__class__(
            select_mocks=select_mocks,
            select_indices=select_indices,
            select_coordinates=self.select_coordinates,
            slice_coordinates=self.slice_coordinates,
            **kwargs)
        test_x = observable.lhc_x
        test_y = observable.lhc_y
        # reshape to (n_samples, n_features)
        n_samples = len(select_mocks['cosmo_idx']) * len(select_mocks['hod_idx'])
        test_x = test_x.reshape(n_samples, -1)
        test_y = test_y.reshape(n_samples, -1)
        pred_y = observable.get_model_prediction(test_x, batch=True, return_tensor=False)
        return test_y - pred_y


    def get_model_residuals_data(self, select_mocks=None):
        """
        Calculate the residuals between the data and the test set of the Latin hypercube. 
        
        We make a new instance of the class with the test set filters 

        Returns:
            np.ndarray: Data residuals.
        """
        import numpy as np
        if select_mocks is None:
            select_mocks = self.test_set_indices
        if self.select_indices:
            select_indices = self.select_indices['bin_idx']
        else:
            select_indices = {}
        if hasattr(self, 'r'):
            kwargs = {'r': self.r}
        else:
            kwargs = {}
        observable = self.__class__(
            select_mocks=select_mocks,
            select_indices=select_indices,
            select_coordinates=self.select_coordinates,
            slice_coordinates=self.slice_coordinates,
            **kwargs)
        test_x = observable.lhc_x
        test_y = observable.lhc_y
        # reshape to (n_samples, n_features)
        n_samples = len( self.test_set_indices['cosmo_idx']) * len( self.test_set_indices['hod_idx'])
        test_x = test_x.reshape(n_samples, -1)
        test_y = test_y.reshape(n_samples, -1)
        return test_y - self.lhc_y 

    def get_model_residuals_list(self):
        import numpy as np
        residuals = []
        if type(self.test_set_indices) == dict:
            select_mocks = [self.test_set_indices]
        else:
            select_mocks = self.test_set_indices
        if self.select_indices:
            select_indices = self.select_indices['bin_idx']
        else:
            select_indices = {}
        
        for i in range(len(select_mocks)):
            observable = self.__class__(
                select_mocks=select_mocks,
                select_indices=select_indices,
                select_coordinates=self.select_coordinates,
                slice_coordinates=self.slice_coordinates)
            test_x = observable.lhc_x
            test_y = observable.lhc_y
            # reshape to (n_samples, n_features)
            n_samples = len(select_mocks['cosmo_idx']) * len(select_mocks['hod_idx'])
            test_x = test_x.reshape(n_samples, -1)
            test_y = test_y.reshape(n_samples, -1)
            pred_y = observable.get_model_prediction(test_x, batch=True)
            residuals.append(test_y - pred_y)
        return np.asarray(residuals)


    def load_separation(self):
        """
        Separation values (s for the correlation function, k for power spectrum, etc.)
        """
        fn = self.lhc_fname()
        return np.load(fn, allow_pickle=True).item()[self.sep_name]

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
        if return_tensor:
            return prediction
        prediction = prediction.numpy()
        if hasattr(self, 'phase_correction'):
            prediction = self.apply_phase_correction(prediction)
        coords = self.coordinates_indices if self.select_indices else self.coordinates
        coords_shape = tuple(len(v) for k, v in coords.items())
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

    def get_covariance_matrix(self, divide_factor=64, nsim=None):
        """
        Covariance matrix of the combination of observables.

        Args:
            divide_factor (int): Divide the covariance matrix by this value
            to account for the volume difference between the small boxes and the target
            simulation.
            nsim (int): Number of simulations to use for the covariance matrix.
                If None, use all simulations.
        Returns:
            np.ndarray: Covariance matrix.
        """
        sims = self.small_box_y if nsim is None else self.small_box_y[:nsim]
        cov = np.cov(sims.T) / divide_factor
        return np.atleast_2d(cov)

    def get_phase_matrix(self):
        """
        Phase error matrix to be used instead of the phase correction.
        This is simpler to calculate but might excesively increase the 
        error budget.
        """
        return np.cov(self.small_box_y().T) / 64

    def get_covariance_correction(self, n_s, n_d, n_theta=None, method='percival'):
        """
        Correction factor to debias de inverse covariance matrix.

        Args:
            n_s (int): Number of simulations.
            n_d (int): Number of bins of the data vector.
            n_theta (int): Number of free parameters.
            method (str): Method to compute the correction factor.

        Returns:
            float: Correction factor
        """
        assert method in ['percival', 'percival-fisher', 'hartlap'], \
            "Method should be one of 'percival', 'percival-fisher' or 'hartlap'"
        if method == 'percival':
            B = (n_s - n_d - 2) / ((n_s - n_d - 1)*(n_s - n_d - 4))
            return (n_s - 1)*(1 + B*(n_d - n_theta))/(n_s - n_d + n_theta - 1)
        elif method == 'percival-fisher':
            return (n_s - 1)/(n_s - n_d + n_theta - 1)
        elif method == 'hartlap':
            return (n_s - 1)/(n_s - n_d - 2)