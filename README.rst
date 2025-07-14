Condor
======

.. image:: https://github.com/nasa/simupy-flight/actions/workflows/docs.yml/badge.svg
   :target: https://nasa.github.io/condor
.. image:: https://img.shields.io/badge/License-NOSA-green.svg
   :target: https://github.com/nasa/condor/blob/master/license.pdf
.. image:: https://img.shields.io/github/release/nasa/condor.svg
   :target: https://github.com/nasa/condor/releases


Condor is a new mathematical modeling framework for Python, developed at
NASA's Ames Research Center. Initial development began in April 2023 to
address model implementation challenges for aircraft synthesis and
robust orbital trajectory design. The goal is for Condor to help
evaluate numerical models and then get out of the way.

One key aspect to achieve this goal was to create an API that looked as
much like the mathematical description as possible with as little
distraction from programming cruft as possible. To best understand
this approach, we can consider a simple benchmark problem which consists
of a set of coupled algebraic expressions. This can be represented as a
system of algebraic equations:

.. code-block:: python

  class Coupling(co.AlgebraicSystem):
      x = parameter(shape=3)
      y1 = variable(initializer=1.)
      y2 = variable(initializer=1.)

      residual(y1 == x[0] ** 2 + x[1] + x[2] - 0.2 * y2)
      residual(y2 == y1**0.5 + x[0] + x[1])

This parametric model can be evaluated by providing the values for the
parameters; the resulting object has values for its inputs and outputs
bound, so the solved values for ``y1`` and ``y2`` can be accessed easily:

.. code-block:: python

   coupling = Coupling([5., 2., 1]) # evaluate the model numerically
   print(coupling.y1, coupling.y2) # individual elements are bound numerically
   print(coupling.variable) # fields are bound as a dataclass

Models can also be seamlessly built-up, with parent models accessing any
input or output of models embedded within them. For example, we can
optimize this system of algebraic equations by embedding it within an
optimization problem:

.. code-block:: python

  class Sellar(co.OptimizationProblem):
      x = variable(shape=3, lower_bound=0, upper_bound=10)
      coupling = Coupling(x)
      y1, y2 = coupling

      objective = x[2]**2 + x[1] + y1 + exp(-y2)
      constraint(y1 >= 3.16)
      constraint(24. >= y2)

After the model is solved, the embedded model can be accessed directly:

.. code-block:: python

   Sellar.set_initial(x=[5,2,1])
   sellar = Sellar()
   print(sellar.objective) # scalar value
   print(sellar.constraint) # field
   print(sellar.coupling.y1) # sub-model element

NASA's Condor is a framework for mathematical modeling of engineering
systems in Python, written for engineers with a deadline.

Installation
------------

Condor is available on `PyPI <https://pypi.org/project/condor/>`_, so you can
install with pip:

.. code:: bash

   $ pip install condor

Contributing
------------

Please feel free to share any thoughts or opinions about the design and
implementation of this software by `opening an issue on GitHub
<https://github.com/nasa/condor/issues/new>`_. Constructive feedback is
welcomed and appreciated.

Bug fix pull requests are always welcome. For feature additions, breaking
changes, etc. check if there is an open issue discussing the change and
reference it in the pull request. If there isn't one, it is recommended to open
one with your rationale for the change before spending significant time
preparing the pull request.

Ideally, new/changed functionality should come with tests and documentation. If
you are new to contributing, it is perfectly fine to open a work-in-progress
pull request and have it iteratively reviewed.

For pull requests to be accepted, all contributors must have a contributor's
agreement on file with NASA. We will provide contributors with additional
information during the review process.

License
-------

This software is released under the `NASA Open Source Agreement Version 1.3 <https://github.com/nasa/condor/raw/main/license.pdf>`_.

Notices
-------

Copyright © 2024 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

Disclaimers
-----------

No Warranty: THE SUBJECT SOFTWARE IS PROVIDED "AS IS" WITHOUT ANY WARRANTY OF ANY KIND, EITHER EXPRESSED, IMPLIED, OR STATUTORY, INCLUDING, BUT NOT LIMITED TO, ANY WARRANTY THAT THE SUBJECT SOFTWARE WILL CONFORM TO SPECIFICATIONS, ANY IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, OR FREEDOM FROM INFRINGEMENT, ANY WARRANTY THAT THE SUBJECT SOFTWARE WILL BE ERROR FREE, OR ANY WARRANTY THAT DOCUMENTATION, IF PROVIDED, WILL CONFORM TO THE SUBJECT SOFTWARE. THIS AGREEMENT DOES NOT, IN ANY MANNER, CONSTITUTE AN ENDORSEMENT BY GOVERNMENT AGENCY OR ANY PRIOR RECIPIENT OF ANY RESULTS, RESULTING DESIGNS, HARDWARE, SOFTWARE PRODUCTS OR ANY OTHER APPLICATIONS RESULTING FROM USE OF THE SUBJECT SOFTWARE.  FURTHER, GOVERNMENT AGENCY DISCLAIMS ALL WARRANTIES AND LIABILITIES REGARDING THIRD-PARTY SOFTWARE, IF PRESENT IN THE ORIGINAL SOFTWARE, AND DISTRIBUTES IT "AS IS."

Waiver and Indemnity:  RECIPIENT AGREES TO WAIVE ANY AND ALL CLAIMS AGAINST THE UNITED STATES GOVERNMENT, ITS CONTRACTORS AND SUBCONTRACTORS, AS WELL AS ANY PRIOR RECIPIENT.  IF RECIPIENT'S USE OF THE SUBJECT SOFTWARE RESULTS IN ANY LIABILITIES, DEMANDS, DAMAGES, EXPENSES OR LOSSES ARISING FROM SUCH USE, INCLUDING ANY DAMAGES FROM PRODUCTS BASED ON, OR RESULTING FROM, RECIPIENT'S USE OF THE SUBJECT SOFTWARE, RECIPIENT SHALL INDEMNIFY AND HOLD HARMLESS THE UNITED STATES GOVERNMENT, ITS CONTRACTORS AND SUBCONTRACTORS, AS WELL AS ANY PRIOR RECIPIENT, TO THE EXTENT PERMITTED BY LAW.  RECIPIENT'S SOLE REMEDY FOR ANY SUCH MATTER SHALL BE THE IMMEDIATE, UNILATERAL TERMINATION OF THIS AGREEMENT.
