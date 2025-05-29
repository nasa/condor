"""
This module uses the backend to process a Model with values for input fields to call
solvers. 
"""
from dataclasses import dataclass
from enum import Enum, auto

import condor as co
import numpy as np

# TODO: make SGM and SolverWithWarmStart (really, back-tracking solver and possibly only
# needed if broyden doesn't resolve it?) generic and figure out how to separate the
# algorithm from the casadi callback. I think SWWS already does this pretty well

# Implementations should definitely be the owners of expressions used to generate
# functions -- maybe own all intermediates used to interface with solver? I think some
# casadi built-in solvers+symbolic representation take either (or both) expressions and
# ca.Function`s (~ duck-typed functions that can operate on either symbols or numerics,
# kind of an arechetype of callback?) but even when they take expressions, I don't think
# they allow access to them after creation so it's useful for the implementation to keep
# it

# would like implementation to be the interface between model field and the callback,
# but some of the helpers (eg initializer, state setter, etc) generate functions, not
# just collect expressions.

# TODO figure out how to use names for casadi callback layer
# TODO function generation is primarily responsibility of callback (e.g., nlpsol takes
# expressions, not functions)
# TODO if we provide a backend reference to vertcat, implementations can be
# backend-agnostic and just provide appropriate flattening + binding of fields! may
# also/instead create callback even for OptimizationProblem, ExplicitSystem to provide
# consistent interface
# --> move some of the helper functions that are tightly coupled to backend to utils, ?
# and generalize, eg state setter
# TODO for custom solvers like SGM, table, does the get_jacobian arguments allow you to
# avoid computing wrt particular inputs/outputs if possible?

from condor.implementations.simple import DeferredSystem, ExplicitSystem, ExternalSolverModel
from condor.implementations.iterative import (
    AlgebraicSystem, CasadiNlpsolImplementation as OptimizationProblem, ScipyCG,
    ScipySLSQP, ScipyTrustConstr
)
from condor.implementations.sgm_trajectory import TrajectoryAnalysis



