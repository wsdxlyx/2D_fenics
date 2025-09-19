import numpy as np

in718 = np.array([1.0, 0.0, 0.0])
invar = np.array([0.0, 1.0, 0.0])
mk500 = np.array([0.0, 0.0, 1.0])

materials = (("IN718", "INVAR", "MK500"), np.array([in718, invar, mk500]))



__all__ = ["materials"]
