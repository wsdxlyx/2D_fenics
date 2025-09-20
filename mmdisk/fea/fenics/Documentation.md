# OVERVIEW
This is a documentation for important functions 


## CONTENTS
1. incremental_cyclic_fea( ) in [*incremental.py*](incremental.py)
- ...



## 1. incremental_cyclic_fea( )
### Overview
what does this function do

### Parameters
- ___mesh___: _dolfin Mesh object_
- ___properties___: _tuple_
    A tuple containing 7 material properties ($\rho$, $C_p$, $k$, $E$, $\sigma_{y0}$, $\nu$, $\alpha$), with each element a dolfin Function object.
- ___load___: _callable_
    callable function of _get_loadfunc(T_load, cycle)_
- ___omega___: _float_
    Disk rotational speed [rad/s]
- ___t_list___: _np.ndarray_
    time sequence defining thermal cycles
- ___t_output___: _np.ndarray_
    time sequence at which results are output
- ___period___: _float_
    thermal cycle period [s]
- ___Displacement_BC___: _str="fix_free"_
    disk hub and rim boundary conditions, either "fix-free" or "free-free"
- ___plastic_inverval___: _int = 1_
    time increment inverval for plasticity update 
- ___tol___: _float = 1e-5_
    convergence tolerance
- ___Nitermax___: _int = 50_
    maximum number of iterations for convergence
- ___skip_cold___: _bool = False_,
    If True, skips the cold dwell for the thermal problem.
- ___verbose___: _bool = False_,
    If True, enables verbose logging.
- ___outputs___: _Union [None, list, dict] = None_
    Outputs to be collected during the analysis. If None, defaults to collecting "PEEQ". If a list, it will be converted to a dictionary with empty arrays. If a dictionary， it will be used to store the outputs with preallocated arrays. Available outputs: "PEEQ" (equivalent plastic strain), "T" (temperature), "u" (displacement), "sig" (stress), "eps_vector" (strain vector [$\varepsilon_{rr}$, $\varepsilon_{\theta\theta}$, $\varepsilon_{zz}$, $\varepsilon_{rz}$]).



### Returns
- ___t_output___: _np.ndarray_
    the same as input parameters
- ___outputs___: _dict_
    the result variables requested
- ___flag___: _int_
     - 0: normal completion
     - 1: early stopping due to elastic limits
     - 2: early stopping due to shakedown limits

### Line-by-line Explanation


### Example


### Notes
N/A

```python
hi
```