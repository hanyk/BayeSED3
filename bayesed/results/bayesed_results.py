"""
BayeSEDResults public facade.

Delegates to ResultsProducts and ResultsPresentation (ADR-0003).
"""

from pathlib import Path
from typing import List, Dict, Optional, Union, Any

from .results_products import ResultsProducts
from .results_presentation import ResultsPresentation


class BayeSEDResults:
    """
    BayeSEDResults facade with the historical public interface.

    Internals: ResultsProducts (HDF5/table/evidence/samples) and
    ResultsPresentation (GetDist, plots, correlations/statistics).
    """

    def __init__(self, output_dir: Union[str, Path], catalog_name: Optional[str] = None,
                 model_config: Optional[Union[str, int]] = None, object_id: Optional[str] = None):
        self._products = ResultsProducts(
            output_dir, catalog_name=catalog_name,
            model_config=model_config, object_id=object_id,
        )
        self._presentation = ResultsPresentation(self._products)

    # --- attribute compatibility ---
    @property
    def output_dir(self):
        return self._products.output_dir

    @output_dir.setter
    def output_dir(self, value):
        self._products.output_dir = value

    @property
    def catalog_name(self):
        return self._products.catalog_name

    @catalog_name.setter
    def catalog_name(self, value):
        self._products.catalog_name = value

    @property
    def model_config(self):
        return self._products.model_config

    @model_config.setter
    def model_config(self, value):
        self._products.model_config = value

    @property
    def object_id(self):
        return self._products.object_id

    @object_id.setter
    def object_id(self, value):
        self._products.object_id = value

    @property
    def parameters(self):
        return self._products.parameters

    @parameters.setter
    def parameters(self, value):
        self._products.parameters = value


    def __repr__(self):
        return repr(self._products)

    def __str__(self):
        return str(self._products)

    def get_bestfit_spectrum(self, *args, **kwargs):
        return self._products.get_bestfit_spectrum(*args, **kwargs)

    def get_configuration_summary(self, *args, **kwargs):
        return self._products.get_configuration_summary(*args, **kwargs)

    def get_derived_parameters(self, *args, **kwargs):
        return self._products.get_derived_parameters(*args, **kwargs)

    def get_evidence(self, *args, **kwargs):
        return self._products.get_evidence(*args, **kwargs)

    def get_free_parameters(self, *args, **kwargs):
        return self._products.get_free_parameters(*args, **kwargs)

    def get_parameter_names(self, *args, **kwargs):
        return self._products.get_parameter_names(*args, **kwargs)

    def get_parameter_values(self, *args, **kwargs):
        return self._products.get_parameter_values(*args, **kwargs)

    def get_posterior_samples(self, *args, **kwargs):
        return self._products.get_posterior_samples(*args, **kwargs)

    def list_model_components(self, *args, **kwargs):
        return self._products.list_model_components(*args, **kwargs)

    def list_model_configurations(self, *args, **kwargs):
        return self._products.list_model_configurations(*args, **kwargs)

    def list_objects(self, *args, **kwargs):
        return self._products.list_objects(*args, **kwargs)

    def load_hdf5_results(self, *args, **kwargs):
        return self._products.load_hdf5_results(*args, **kwargs)

    def print_summary(self, *args, **kwargs):
        return self._products.print_summary(*args, **kwargs)

    def rename_parameters(self, *args, **kwargs):
        return self._products.rename_parameters(*args, **kwargs)

    def validate_model_config(self, *args, **kwargs):
        return self._products.validate_model_config(*args, **kwargs)

    def compute_parameter_correlations(self, *args, **kwargs):
        return self._presentation.compute_parameter_correlations(*args, **kwargs)

    def get_getdist_samples(self, *args, **kwargs):
        return self._presentation.get_getdist_samples(*args, **kwargs)

    def get_parameter_statistics(self, *args, **kwargs):
        return self._presentation.get_parameter_statistics(*args, **kwargs)

    def plot_bestfit(self, *args, **kwargs):
        return self._presentation.plot_bestfit(*args, **kwargs)

    def plot_parameter_scatter(self, *args, **kwargs):
        return self._presentation.plot_parameter_scatter(*args, **kwargs)

    def plot_posterior(self, *args, **kwargs):
        return self._presentation.plot_posterior(*args, **kwargs)

    def plot_posterior_derived(self, *args, **kwargs):
        return self._presentation.plot_posterior_derived(*args, **kwargs)

    def plot_posterior_free(self, *args, **kwargs):
        return self._presentation.plot_posterior_free(*args, **kwargs)

    def set_parameter_labels(self, *args, **kwargs):
        return self._presentation.set_parameter_labels(*args, **kwargs)

    # Statics used by callers/tests via the public class name
    _to_str = staticmethod(ResultsProducts._to_str)
    _match_strings = staticmethod(ResultsProducts._match_strings)
