import warnings

from dolfin import parameters
from ffc.quadrature.deprecation import QuadratureRepresentationDeprecationWarning

warnings.simplefilter("ignore", QuadratureRepresentationDeprecationWarning)

parameters["form_compiler"]["representation"] = "quadrature"
