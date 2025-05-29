import condor.backends.casadi as backend
"""
vcat/vertcat -- combine multiple backend_symbol
flatten -- always used in cnojuction with vertcat? to make differently shaped
           backend_reprs 

vertsplit -- opposite of vertcat
wrap -- opposite of flatten?
--> these two pairs are really single functions of fields

Function -- create a numerical/symbolic callable

??? -- create a backend-compatible Op from functions for 0th derivative and more
currently CasadiFunctionCallback, but needs to be generalized

if_else -- symbolic control flow
??? -- something for symbolic array stuffing if forecasting JAX compatibility;
may work just to use elements everywhere? or use a wrapped JAX array as the
backend_repr? assume replacing with elements works and JAX backend element mixin can
include __setitem__ overwrite to replace backend_repr with .at[...].set(...) result
would be nice if numpy ufunc dispatch mechanism worked and could be applied to elements.

nlpsol -- should get moved to the solvers (uses casadi, scipy solvers)
ideally, backend-compatible Op would see nlpsol is already an op,
Interface >>> performance hit of nlpsol native vs wrapped, presumably





"""

"""
Need to keep in a separate file so backend.get_symbol_data (required by fields which
generate the symbols) can return filled dataclass without causing a circular import

"""

"""
Backend:
[x] provide symbol_generator for creating backend symbol repr
[x] symbol_class for isinstance(model_attr, backend.symbol_class
[x] name which is how backend options on model is identified
Do we allow more complicated datastructures? like models, etc.

optional implementations which has a __dict__ that allows assignment
Or, should the implementations just live in the main backend?


Backend Implementations
[x] must be able to flatten model symbols to backend arrays,
[x] wrap backend arrays to model symbol, matching shape -- 
[ ] wrap and flatten must handle model numerics (float/numpy array) and backend numerics (if
different, eg casadi DM) and backend symbols
[ ] ideally, handle special case symmetric and dynamic flags for FreeSymbol and
[ ] MatchedSymbol if matched to symmetric/diagonal FreeSymbol
setting the values for outputs and intermediates

Couples to Model types -- knows all fields

who is responsible for:
filling in fields/._dataclass?
filling in model input/output attrs?




"""

"""
Figure out how to add DB storage -- maybe expect that to be a user choice (a decorator
or something)? Then user defined initializer could use it. Yeah, and ORM aspect just
takes advantage of model attributes just like backend implementation


I assume a user model/library code could inject an implementation to the backend?
not sure how to assign special numeric stuff, probably an submodel class on the model
based on NPSS discussion it's not really needed if it's done right 

For injecting a default implementation (e.g., new backend) this does work:

import casadi_implementations
casadi_implementations.ODESystem = 'a reference to check exists'

import condor as co

but probably should just figure out hooks to do that? Could create local dict of backend
that gets updated with backend.implementations at the top of this file, then library/user
code could update it (add/overwrite)
"""

