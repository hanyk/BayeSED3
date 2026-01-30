"""
Prior management for BayeSED3.

This module provides classes for managing parameter priors in BayeSED3 analysis.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Prior:
    """
    Represents a single parameter prior with support for all BayeSED3 prior types.
    
    BayeSED3 supports prior types from -9 to 9, where negative values indicate log10 space.
    Different prior types require different numbers of hyperparameters:
    
    - Type 0: Mirror prior (3 hyperparameters: model_id, par_type, par_id)
      References other parameters in the model:
      * par_type=0: Other information in input file (model_id must be -1)
      * par_type=1: Free parameters of model with given model_id
      * par_type=2: Derived parameters of model with given model_id
    - Types 1, 2, 3: Standard priors (no hyperparameters)
    - Type 4: Truncated Gaussian (0-2 hyperparameters with defaults)
    - Type 5: Gaussian (2 hyperparameters: mu, sigma)
    - Type 6: Gamma (2 hyperparameters: alpha, beta)
    - Type 7: Student's-t (3 hyperparameters: mu, sigma, nu)
    - Type 8: Beta (2 hyperparameters: a, b)
    - Type 9: Weibull (2 hyperparameters: a, b)
    
    Parameters
    ----------
    name : str
        Parameter name (e.g., "log(age/yr)", "Av_2")
    prior_type : int
        Prior type (-9 to 9, negative means log10 space)
    is_age : int
        Age-dependent flag (0 or 1)
    min_val : float
        Minimum value
    max_val : float
        Maximum value
    nbin : int
        Number of bins
    hyperparameters : List[float], optional
        Hyperparameters for different prior distributions (flexible list)
    component : str, optional
        Model component (e.g., "ssp", "agn")
    description : str, optional
        Parameter description
        
    Examples
    --------
    >>> # Create a uniform prior
    >>> prior = Prior(name="log(age/yr)", prior_type=1, is_age=1, 
    ...               min_val=8.0, max_val=10.0, nbin=40)
    
    >>> # Create a Gaussian prior
    >>> prior = Prior(name="log(M_*/Msun)", prior_type=5, is_age=0,
    ...               min_val=8.0, max_val=12.0, nbin=40,
    ...               hyperparameters=[10.0, 1.0])  # mu=10.0, sigma=1.0
    """
    name: str
    prior_type: int
    is_age: int
    min_val: float
    max_val: float
    nbin: int
    hyperparameters: List[float] = field(default_factory=list)
    component: Optional[str] = None
    description: Optional[str] = None
    source_file: Optional[str] = None  # Track which .iprior file this came from
    
    def get_required_columns(self) -> int:
        """
        Get the number of columns required for this prior type in .iprior file format.
        
        Returns
        -------
        int
            Number of columns (5-8 depending on prior type)
            
        Examples
        --------
        >>> prior = Prior(name="test", prior_type=5, is_age=0, 
        ...               min_val=0, max_val=1, nbin=10)
        >>> prior.get_required_columns()
        7
        """
        abs_type = abs(self.prior_type)
        if abs_type == 9:  # Weibull: [iprior_type is_age min max nbin a b]
            return 7
        elif abs_type == 8:  # Beta: [iprior_type is_age min max nbin a b]
            return 7
        elif abs_type == 7:  # Student's-t: [iprior_type is_age min max nbin mu sigma nu]
            return 8
        elif abs_type == 6:  # Gamma: [iprior_type is_age min max nbin alpha beta]
            return 7
        elif abs_type == 5:  # Gaussian: [iprior_type is_age min max nbin mu sigma]
            return 7
        elif abs_type == 4:  # Truncated Gaussian: [iprior_type is_age min max nbin mu sigma]
            return 7  # Can be 5, 6, or 7 columns with defaults
        elif self.prior_type == 0:  # Special case: 8 columns
            return 8
        else:  # Types 1, 2, 3: [iprior_type is_age min max nbin]
            return 5
    
    def get_required_hyperparameters(self) -> int:
        """
        Get the number of hyperparameters required for this prior type.
        
        Returns
        -------
        int
            Number of required hyperparameters (0-3 depending on prior type)
            
        Examples
        --------
        >>> prior = Prior(name="test", prior_type=5, is_age=0,
        ...               min_val=0, max_val=1, nbin=10)
        >>> prior.get_required_hyperparameters()
        2
        """
        # Base columns are: type, is_age, min, max, nbin
        return self.get_required_columns() - 5
    
    def validate(self) -> List[str]:
        """
        Validate prior parameters and return list of errors.
        
        Returns
        -------
        List[str]
            List of validation error messages (empty if valid)
            
        Examples
        --------
        >>> prior = Prior(name="test", prior_type=1, is_age=0,
        ...               min_val=2.0, max_val=1.0, nbin=10)
        >>> errors = prior.validate()
        >>> len(errors) > 0
        True
        >>> "min_val" in errors[0]
        True
        """
        errors = []
        
        # Basic validation - allow min_val == max_val (fixed parameter)
        if self.min_val > self.max_val:
            errors.append(f"min_val ({self.min_val}) must be <= max_val ({self.max_val})")
        if not -9 <= self.prior_type <= 9:
            errors.append(f"prior_type ({self.prior_type}) must be between -9 and 9")
        if self.nbin <= 0:
            errors.append(f"nbin ({self.nbin}) must be positive")
            
        # Log space validation (negative prior types)
        if self.prior_type < 0:
            if self.min_val <= 0 or self.max_val <= 0:
                errors.append(f"min_val and max_val must be positive for log10 space (prior_type < 0)")
        
        # Hyperparameter validation
        required_hyper = self.get_required_hyperparameters()
        if len(self.hyperparameters) != required_hyper:
            type_name = self.get_type_name()
            param_names = self.get_hyperparameter_names()
            error_msg = f"Prior type '{type_name}' (type={self.prior_type}) requires {required_hyper} hyperparameters, but {len(self.hyperparameters)} provided"
            if param_names:
                error_msg += f". Required: {', '.join(param_names)}"
            errors.append(error_msg)
        
        # Type-specific hyperparameter validation
        abs_type = abs(self.prior_type)
        if abs_type == 5 and len(self.hyperparameters) >= 2:  # Gaussian: mu, sigma
            if self.hyperparameters[1] <= 0:  # sigma must be positive
                errors.append("Gaussian prior sigma (hyperparameter[1]) must be positive")
        elif abs_type == 6 and len(self.hyperparameters) >= 2:  # Gamma: alpha, beta
            if self.hyperparameters[0] <= 0 or self.hyperparameters[1] <= 0:
                errors.append("Gamma prior alpha and beta (hyperparameters[0,1]) must be positive")
        elif abs_type == 7 and len(self.hyperparameters) >= 3:  # Student's-t: mu, sigma, nu
            if self.hyperparameters[1] <= 0:  # sigma must be positive
                errors.append("Student's-t prior sigma (hyperparameter[1]) must be positive")
            if self.hyperparameters[2] <= 0:  # nu must be positive
                errors.append("Student's-t prior nu (hyperparameter[2]) must be positive")
        elif abs_type == 8 and len(self.hyperparameters) >= 2:  # Beta: a, b
            if self.hyperparameters[0] <= 0 or self.hyperparameters[1] <= 0:
                errors.append("Beta prior a and b (hyperparameters[0,1]) must be positive")
        elif abs_type == 9 and len(self.hyperparameters) >= 2:  # Weibull: a, b
            if self.hyperparameters[0] <= 0 or self.hyperparameters[1] <= 0:
                errors.append("Weibull prior a and b (hyperparameters[0,1]) must be positive")
                
        return errors
    
    def to_iprior_line(self) -> str:
        """
        Convert prior to .iprior file line format.
        
        Returns
        -------
        str
            Formatted line for .iprior file
            
        Examples
        --------
        >>> prior = Prior(name="log(age/yr)", prior_type=1, is_age=1,
        ...               min_val=8.0, max_val=10.0, nbin=40)
        >>> prior.to_iprior_line()
        '1 1 8.0 10.0 40'
        
        >>> prior = Prior(name="test", prior_type=5, is_age=0,
        ...               min_val=8.0, max_val=12.0, nbin=40,
        ...               hyperparameters=[10.0, 1.0])
        >>> prior.to_iprior_line()
        '5 0 8.0 12.0 40 10.0 1.0'
        """
        base = f"{self.prior_type} {self.is_age} {self.min_val} {self.max_val} {self.nbin}"
        
        if self.hyperparameters:
            hyper_str = " ".join(str(h) for h in self.hyperparameters)
            return f"{base} {hyper_str}"
        else:
            return base
    
    def get_hyperparameter_names(self) -> List[str]:
        """
        Get the names of hyperparameters for this prior type.
        
        Returns
        -------
        List[str]
            List of hyperparameter names (e.g., ["mu", "sigma"] for Gaussian)
            
        Examples
        --------
        >>> prior = Prior(name="test", prior_type=5, is_age=0,
        ...               min_val=0, max_val=1, nbin=10)
        >>> prior.get_hyperparameter_names()
        ['mu', 'sigma']
        
        >>> prior = Prior(name="test", prior_type=1, is_age=0,
        ...               min_val=0, max_val=1, nbin=10)
        >>> prior.get_hyperparameter_names()
        []
        """
        abs_type = abs(self.prior_type)
        if abs_type == 5:  # Gaussian
            return ["mu", "sigma"]
        elif abs_type == 6:  # Gamma
            return ["alpha", "beta"]
        elif abs_type == 7:  # Student's-t
            return ["mu", "sigma", "nu"]
        elif abs_type == 8:  # Beta
            return ["a", "b"]
        elif abs_type == 9:  # Weibull
            return ["a", "b"]
        elif self.prior_type == 0:  # Mirror prior
            return ["model_id", "par_type", "par_id"]
        else:
            return []
    
    def get_type_name(self) -> str:
        """
        Get the human-readable name for this prior type.
        
        Returns
        -------
        str
            Human-readable type name (e.g., "Gaussian", "Log10_Linear-Inc")
            
        Examples
        --------
        >>> prior = Prior(name="test", prior_type=5, is_age=0,
        ...               min_val=0, max_val=1, nbin=10)
        >>> prior.get_type_name()
        'Gaussian'
        
        >>> prior = Prior(name="test", prior_type=-2, is_age=0,
        ...               min_val=0.1, max_val=1, nbin=10)
        >>> prior.get_type_name()
        'Log10_Linear-Inc'
        """
        abs_type = abs(self.prior_type)
        type_names = {
            0: "Mirror",
            1: "Uniform",
            2: "Linear-Inc",
            3: "Linear-Dec",
            4: "TruncGaussian",
            5: "Gaussian",
            6: "Gamma",
            7: "StudentT",
            8: "Beta",
            9: "Weibull"
        }
        name = type_names.get(abs_type, f"Type{abs_type}")
        # Add Log10_ prefix for negative types
        if self.prior_type < 0:
            return f"Log10_{name}"
        return name
    
    @staticmethod
    def type_name_to_int(type_name: str) -> int:
        """
        Convert a string type name to integer type code.
        
        Parameters
        ----------
        type_name : str
            Type name (e.g., 'Gaussian', 'Linear-Inc', 'Log10_Gaussian')
            
        Returns
        -------
        int
            Integer type code (-9 to 9)
            
        Raises
        ------
        ValueError
            If the type name is not recognized
            
        Examples
        --------
        >>> Prior.type_name_to_int('Gaussian')
        5
        >>> Prior.type_name_to_int('Log10_Linear-Inc')
        -2
        >>> Prior.type_name_to_int('Uniform')
        1
        """
        # Check for Log10_ prefix
        is_log10 = type_name.startswith('Log10_')
        if is_log10:
            type_name = type_name[6:]  # Remove 'Log10_' prefix
        
        # Map names to integers
        name_to_type = {
            "Mirror": 0,
            "Uniform": 1,
            "Linear-Inc": 2,
            "Linear-Dec": 3,
            "TruncGaussian": 4,
            "Gaussian": 5,
            "Gamma": 6,
            "StudentT": 7,
            "Beta": 8,
            "Weibull": 9
        }
        
        if type_name not in name_to_type:
            valid_names = ', '.join(name_to_type.keys())
            raise ValueError(f"Unknown prior type name: '{type_name}'. Valid names: {valid_names}")
        
        type_int = name_to_type[type_name]
        return -type_int if is_log10 else type_int
