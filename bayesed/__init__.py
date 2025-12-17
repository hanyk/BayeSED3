"""
BayeSED3 Python Interface

This package provides a Python interface to BayeSED3, a Bayesian SED fitting tool.
"""

__version__ = "3.0.0"

# Import core classes from core module
from .core import (
    BayeSEDParams,
    BayeSEDInterface,
    BayeSEDValidationError,
    BayeSEDExecutionError,
)

# Import enhanced BayeSEDResults from results module
from .results import BayeSEDResults

# Import data classes
from .data import (
    SEDObservation,
    PhotometryObservation,
    SpectrumObservation,
)

# Import model classes
from .model import (
    SEDModel,
)

# Import inference classes
from .inference import (
    SEDInference,
)

# Import plotting functions
from .plotting import (
    plot_bestfit,
)

# Import utility functions
from .utils import (
    _to_array,
    create_input_catalog,
    create_filters_from_svo,
    create_filters_selected,
    infer_filter_itype_icalib,
    FILTER_TYPE_ENERGY,
    FILTER_TYPE_PHOTON,
    FILTER_CALIB_STANDARD,
    FILTER_CALIB_SPITZER_IRAC,
    FILTER_CALIB_SUBMM,
    FILTER_CALIB_BLACKBODY,
    FILTER_CALIB_SPITZER_MIPS,
    FILTER_CALIB_SCUBA,
    FILTER_TYPE_NAMES,
    FILTER_CALIB_NAMES,
)

# Import all parameter classes from params module
from .params import (
    FANNParams,
    AGNParams,
    BlackbodyParams,
    BigBlueBumpParams,
    GreybodyParams,
    AKNNParams,
    LineParams,
    LuminosityParams,
    NPSFHParams,
    PolynomialParams,
    PowerlawParams,
    RBFParams,
    SFHParams,
    SSPParams,
    SEDLibParams,
    SysErrParams,
    ZParams,
    NNLMParams,
    NdumperParams,
    OutputSFHParams,
    MultiNestParams,
    SFROverParams,
    SNRmin1Params,
    SNRmin2Params,
    GSLIntegrationQAGParams,
    GSLMultifitRobustParams,
    KinParams,
    LineListParams,
    MakeCatalogParams,
    CloudyParams,
    CosmologyParams,
    DALParams,
    RDFParams,
    TemplateParams,
    RenameParams,
)

__all__ = [
    # Core classes
    'BayeSEDParams',
    'BayeSEDInterface',
    'BayeSEDResults',  # Enhanced version from results module
    'BayeSEDValidationError',
    'BayeSEDExecutionError',

    # Data classes
    'SEDObservation',
    'PhotometryObservation',
    'SpectrumObservation',
    # Model classes
    'SEDModel',
    # Inference classes
    'SEDInference',
    # Plotting
    'plot_bestfit',
    # Utilities
    '_to_array',
    'create_input_catalog',
    'create_filters_from_svo',
    'create_filters_selected',
    'infer_filter_itype_icalib',
    # Filter constants
    'FILTER_TYPE_ENERGY',
    'FILTER_TYPE_PHOTON',
    'FILTER_CALIB_STANDARD',
    'FILTER_CALIB_SPITZER_IRAC',
    'FILTER_CALIB_SUBMM',
    'FILTER_CALIB_BLACKBODY',
    'FILTER_CALIB_SPITZER_MIPS',
    'FILTER_CALIB_SCUBA',
    'FILTER_TYPE_NAMES',
    'FILTER_CALIB_NAMES',
    # Parameter classes
    'FANNParams',
    'AGNParams',
    'BlackbodyParams',
    'BigBlueBumpParams',
    'GreybodyParams',
    'AKNNParams',
    'LineParams',
    'LuminosityParams',
    'NPSFHParams',
    'PolynomialParams',
    'PowerlawParams',
    'RBFParams',
    'SFHParams',
    'SSPParams',
    'SEDLibParams',
    'SysErrParams',
    'ZParams',
    'NNLMParams',
    'NdumperParams',
    'OutputSFHParams',
    'MultiNestParams',
    'SFROverParams',
    'SNRmin1Params',
    'SNRmin2Params',
    'GSLIntegrationQAGParams',
    'GSLMultifitRobustParams',
    'KinParams',
    'LineListParams',
    'MakeCatalogParams',
    'CloudyParams',
    'CosmologyParams',
    'DALParams',
    'RDFParams',
    'TemplateParams',
    'RenameParams',
]
