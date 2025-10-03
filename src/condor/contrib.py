"""Built-in model templates"""

import logging
from dataclasses import dataclass, field

import ndsplines
import numpy as np

from condor.backend import (
    expression_to_operator,
    is_constant,
    process_relational_element,
)
from condor.backend.operators import if_else, substitute
from condor.fields import (
    AssignedField,
    BoundedAssignmentField,
    Direction,
    FreeAssignedField,
    FreeField,
    InitializedField,
    MatchedField,
    TrajectoryOutputField,
    WithDefaultField,
    pass_through,
    zero_like,
)
from condor.models import (
    ModelTemplate,
    ModelTemplateType,
    ModelType,
    SubmodelMetaData,
    SubmodelTemplate,
    SubmodelType,
    check_attr_name,
)

log = logging.getLogger(__name__)


class DeferredSystem(ModelTemplate):
    """output is an explicit function of input"""

    input = FreeField()
    output = FreeField(Direction.output)


class ExplicitSystem(ModelTemplate):
    r"""Representation of a system of explicit equations

    Outputs are functions solely of inputs:

    .. math::
       \begin{align}
       y_1 &=& f_1(x_1, x_2, \dots, x_n) \\
       y_2 &=& f_2(x_1, x_2, \dots, x_n) \\
       & \vdots & \\
       y_m &=& f_m(x_1, x_2, \dots, x_n)
       \end{align}

    where each :math:`y_i` is a name-assigned expression on the ``output`` field and
    each :math:`x_i` is an element drawn from the ``input`` field.

    Each :math:`x_i` and :math:`y_j` may have arbitrary shape. Condor can automatically
    calculate the derivatives :math:`\frac{dy_j}{dx_i}` as needed for parent solvers,
    etc.
    """

    #: the inputs of the model (e.g., :math:`x_i`)
    input = FreeField()
    #: the outputs of the model (e.g., :math:`y_i = f_i(x)`)
    output = AssignedField()


class AlgebraicSystemType(ModelType):
    @classmethod
    def process_placeholders(cls, new_cls, attrs):
        super().process_placeholders(new_cls, attrs)
        for elem in new_cls.residual:
            process_relational_element(elem)


class AlgebraicSystem(ModelTemplate, model_metaclass=AlgebraicSystemType):
    r"""Representation of a system of algebraic equations

    An algebraic system with parameters :math:`u` and implicit variables :math:`x`,
    is driven to a solution at :math:`x^*`:

    .. math::
       \begin{align}
       R_1(u_1, \dots, u_m, x_1^*, \dots, x_n^*) &=& 0 \\
       \vdots & & \\
       R_n(u_1, \dots, u_m, x_1^*, \dots, x_n^*) &=& 0
       \end{align}

    Condor solves for the :math:`x_i^*` and can automatically calculate the derivatives
    :math:`\frac{dx_i}{du_j}` as needed for parent solvers, etc.

    Additional explicit outputs at the solution may also be included:

    .. math::
        \begin{align}
        y_1 &=& f_1(u_1, \dots, u_m, x_1^*, \dots, x_n^*) \\
        & \vdots & \\
        y_l &=& f_m(x_1, \dots, u_m, x_1^*, \dots, x_n^*)
        \end{align}

    """

    # uses AlgebraicSystem type as model metaclass to support relational declaration of
    # residual

    #: parameters held constant during solution of the system, :math:`u`
    parameter = FreeField()
    #: variables used to drive residuals to 0, :math:`x`
    variable = InitializedField(Direction.output)
    #: residuals to satisfy, :math:`R`
    residual = FreeAssignedField(Direction.internal)
    #: additional explicit outputs, :math:`y`
    output = AssignedField()

    # TODO: output is unmatched, but maybe a subclass or imp might check lengths of
    # residuals and implicit_outputs to ensure enough DOF?

    @classmethod
    def set_initial(cls, **kwargs):
        """Set initial values for the ``variable``\\s of the model"""
        for k, v in kwargs.items():
            var = getattr(cls, k)
            if var.field_type is not cls.variable:
                msg = (
                    "Use set initial to set the initialier for variables, attempting "
                    f"to set {k}"
                )
                raise ValueError(msg)
            var.initializer = v


class OptimizationProblemType(ModelType):
    @classmethod
    def process_placeholders(cls, new_cls, attrs):
        super().process_placeholders(new_cls, attrs)
        original_elements = list(new_cls.constraint)
        new_cls.constraint._elements = []
        for elem in original_elements:
            re_append_elem = True
            process_relational_element(elem)
            if not is_constant(elem.lower_bound):
                new_cls.constraint(
                    elem.backend_repr - elem.lower_bound, lower_bound=0.0
                )
                re_append_elem = False
            if not is_constant(elem.upper_bound):
                new_cls.constraint(
                    elem.backend_repr - elem.upper_bound, upper_bound=0.0
                )
                re_append_elem = False

            if re_append_elem:
                new_cls.constraint._elements.append(elem)

    @classmethod
    def bind_model_fields(cls, new_cls, attrs):
        super().bind_model_fields(new_cls, attrs)

        p = new_cls.parameter.flatten()
        x = new_cls.variable.flatten()
        g = new_cls.constraint.flatten()
        f = getattr(new_cls, "objective", 0.0)

        new_cls._meta.objective_func = expression_to_operator(
            [x, p],
            f,
            f"{new_cls.__name__}_objective",
        )
        new_cls._meta.constraint_func = expression_to_operator(
            [x, p],
            g,
            f"{new_cls.__name__}_constraint",
        )


class OptimizationProblem(ModelTemplate, model_metaclass=OptimizationProblemType):
    r"""Representation of a general optimization problem

    The problem is of the form:

    .. math::

       \begin{aligned}
       \operatorname*{minimize}_{x_{1}, \ldots, x_{n}} &  &  &
            f (x_1, \ldots, x_n, p_1, \ldots p_m ) \\
       \text{subject to} &  &  & l_{x_i} \le x_i \le u_{x_i} \\
       & & & l_{g_j} \le g_j (x_1, \ldots , x_n, p_1, \ldots p_m ) \le u_{g_j} \\
       \end{aligned}

    The variables :math:`x_i` are driven by the optimization algorithm to minimize the
    objective function :math:`f` while respecting bounds on the variables as well as
    constraint functions :math:`g_j`.

    If an objective isn't provided, the default is constant 0 (i.e. a feasibility
    problem).
    """

    #: variables driven to solve the problem, :math:`x`
    variable = InitializedField(Direction.output)
    #: parameters held constant during solution of the system, :math:`p`
    parameter = FreeField()
    #: constraints :math:`g` expressed as relationals
    constraint = BoundedAssignmentField(Direction.internal)
    #: scalar objective to minimize, :math:`f`
    objective = placeholder()

    # TODO: need validation for objective size == 1

    @classmethod
    def set_initial(cls, **kwargs):
        r"""Set initial values for the ``variable``\s of the model

        Overrides initial values provided to the ``variable`` field in the model
        declaration.
        """
        for k, v in kwargs.items():
            var = getattr(cls, k)
            if var.field_type is not cls.variable:
                msg = (
                    "Use set initial to set the initialier for variables, attempting "
                    f"to set {k}"
                )
                raise ValueError(msg)
            var.initializer = v

    @classmethod
    def from_values(cls, **kwargs):
        """Construct an instance of a solved model from variable and parameter values"""
        self = cls.__new__(cls)
        parameters = {}
        for elem in cls.parameter:
            val = kwargs.pop(elem.name)
            parameters[elem.name] = val
            setattr(self, elem.name, val)

        variables = {}
        for elem in cls.variable:
            val = kwargs.pop(elem.name)
            variables[elem.name] = val
            setattr(self, elem.name, val)

        if kwargs:
            msg = f"Extra arguments provided: {kwargs}"
            raise ValueError(msg)

        self.input_kwargs = parameters
        self.parameter = cls.parameter._dataclass(**parameters)
        self.variable = cls.variable._dataclass(**variables)

        args = [
            self.variable.flatten(),
            self.parameter.flatten(),
        ]
        constraints = cls._meta.constraint_func(*args)
        self.bind_field(cls.constraint.wrap(constraints))

        self.objective = cls._meta.objective_func(*args)

        self.bind_embedded_models()

        return self


class ODESystem(ModelTemplate):
    r"""Representation of a dynamical system

    The system is defined by a set of ordinary differential equations (ODEs) and
    optionally additional outputs :math:`y`:

    .. math::
       \begin{align}
       \dot{x}_i &= f(t,x_1,\ldots,x_n,p_1,\ldots,p_m) \\
       y_i &= h(t,x_1,\ldots,x_n,p_1,\ldots,p_m)
       \end{align}

    where :math:`x` are the states fully defining the evolution of the system, :math:`t`
    is the independent variable (typically time, but may be anything), and :math:`p` are
    constant parameters.

    Typically used in combination with one or more :class:`TrajectoryAnalysis`
    submodels. :class:`Mode` and :class:`Event` submodels may also be added.
    """

    """
    t - indepdendent variable of ODE, notionally time but can be used
    for anything. Used directly by subclasses (e.g., user code may use
    `u=DynamicsModel.t`, implementations will use this symbol
    directly for fields)

    parameter - auxilary variables (constant-in-time) that determine system behavior

    state - fully defines evolution of system driven by ODEs

    dot - derivative of state with respect to indepdnent variable, purely a function of
    state, independent_variable, parameters
    [DAE version will have a residual associated with state differential]

    output - dynamic output at particular state and value of independent variable
    (point-wise in independent variable)

    control - time-varying placeholder, useful for re-defining using mode, etc.
    set it with `make` field -- all controls MUST be set? or default to 0.

    make - set control a value/computation. Or `let`?

    note: no time-varying input to system. Assume dynamicsmodels "pull" what they need
    from other models.
    Need to augment state with "pulled" dynamicsmodels
    But do need a "control" that could be defined based on mode? automatically added to
    output? or is it always just a placeholder for functional space gradient?
    Or also allow it to define eg feedback control

    I guess block diagram is nice for simulating something like saturation block --
    create event and switch modes? but I guess that's re-creatable with modes. But maybe
    needs control/subsitution -- I guess what I'm calling "control" is really an
    (explicit) algebraic state? I guess this is really the same as an "output" and is in
    fact how simupy implements it. maybe convert output to a freefield that takes an
    expression (like constraint, I guess?) and then "make" is only in mode and adds
    conditional behavior -- defaults to value from creation?

    then keep control field but only for open loop control? not sure, but some way to
    mark an output as a control signal

    don't include specialty types like discrete control, etc? user must know that for
    discrete time signal it requires state with dot=0 and update

    dt - reserved keyword for DT?

    inner model "Event" (see below). For a MyODESyStem model, create a sub-class of
    MyODESystem.Event,
    inner classes of "Mode" subclass -- conditional overwrite of dot and make
    inner classes of TrajectoryAnalysis -- perform simulations with particular
    trajectory outputs. Local copy of parameters, initial to allow sim-specific



    """

    # TODO: indepdent var  needs its own descriptor type? OR do we want user classes to
    # do t = DynamicsModel.independent_variable ? That would allow leaving it like this
    # and consumers still know what it is or just set it to t and always use it? trying
    # this way...

    # TODO: Are initial conditions an attribute of the dynamicsmodel or the trajectory
    # analysis that consumes it?
    # Currently, ODESystem sets a default that TrajectoryAnalysis can override for
    # current sim. I think I like this

    # TODO: mode and corresponding FiniteState type?
    # Yes mode with condition, no FiniteState -- this is an idiom for a state with no
    # dot and only updated to integer values. Expect to use in conditions with ==. Can
    # python enum values get passed to backend?

    # TODO: convenience for setting max_step based on discrete time systems? And would
    # like event functions of form (t - te) to loop with a specified tf. For now can set
    # max_step in trajectoryanalysis options manually... OR figure out how to specify
    # periodic exact-time simulation and then current machinery should be fine

    # TODO: decide how controls work :) Ideally, easy to switch between continuous and
    # sampled feedback, open-loop control. Open-loop would need some type of
    # specification so adjoint gradients could get computed??

    # TODO: combine all events if functions are the same? maybe that happens
    # automatically and that's okay. Needs the new events API (mimicking sundials) to
    # happen --> VERIFY that SGM works correctly? Would require non-conflicting update
    # functions

    # TODO: inside another model, the callback gets captured into the "expression" and
    # will never hit the model's or implementation's __call__ methods. Would really like
    # to 1) bind the simulation result (as in the implementation call) and/or 2)
    # initialize the model since that is most likely where we would hook the
    # sweepable-like logging. This is also/really a question about how to inject Model
    # layer back into casadi expressions. Could be used for capturing dangling
    # computations which currently leaning away from doing but trajectory is a special
    # case... If the trajectory analysis only appears once in the other model could use
    # the caching of result object on callback to
    # Can set an attribute (`from_implementation`) on callback  assuming we make sure
    # new  instances of implementation (and callback) are created to ensure re-entrant.
    # Then refactor binding to create new model instance as appropriate. This would be
    # version #2, probably can't figure out what model called the trajectory analysis
    # and bind in which is probably more consistent anyway.

    # TODO: does the simulation result need to be included in the template somehow?
    # Or is this an implementation detail and current approach is fine? Or how to
    # indicate that time, state, and output fields are time varying and should be
    # written as an arrray and not to eg the database?

    # SimuPy-coupled TODO

    # TODO switch to scikits.odes, especially updating update function API to get an
    # array of length num_events w/ elements 0 for did not occur and +/-1 for direction
    # don't use nan's to terminate? althogh, can't code-gen termination?

    # --> MAKE SURE scikits.odes is re-entrant. AND important TODO: figure out how to
    # create new implementation instances as needed for parallelization. Not sure if
    # that would happen automatically with pickle, etc.

    # TODO: simupy currently requires dim_output > 0, which kinda makes sense for the
    # BlockDiagram case and kinda makes sense for disciplined dynamical system modeling,
    # but maybe doesn't make sense for single system simulation?
    # IF we were keeping intermediate computations, they would be good candidates

    # OR can/should trajectory output integrandds get added to simupy outputs, save a
    # for-loop?

    # what is a good output for adjoint system? gradient integrand term!

    # TODO: currently, to get a discrete time control need to augment state and provide
    # a separate initializer, even though it's  generally going to be the same
    # expression as the update. Should that be fixed in simupy? event funciton = 0 -> do
    # update? I can fix it in casadi shooting_gradient_method.py as well.

    # or is initial where we want it? consistent with initial conition processing for
    # SGM.

    # Are simupy outputs even good? need to be disciplined to keep system encapsolation
    # anyway, and control as discrete state prevents the re-computation. I guess
    # back-to-back computation of a "continuous" optimization (not sure if DAEs can do
    # this) is fine too? I think DAE version will be even more efficient, since
    # everything is a "state" and computed simultaneously. Can you do MAUD approach in
    # IDA to provide solvers for explicit expressions?

    # TODO: add event times/event channels to SimulationResult during detection

    # TODO: don't like hacks to simupy to make it work... especially the success
    # checking stuff -- is that neccessary anymore?

    #: independent variable :math:`t`
    t = placeholder(default=None)
    #: state variables :math:`x`
    state = FreeField(Direction.internal)
    #: initial values for the states; may also/instead be specified at the
    #: :class:`TrajectoryAnalysis` level
    initial = MatchedField(state, default_factory=zero_like)
    #: constant (in time) parameters :math:`p`
    parameter = FreeField()
    #: derivatives of the state variables with respect to :math:`t`, :math:`\dot{x}`
    dot = MatchedField(state, default_factory=zero_like)
    #: elements with deferred behavior, for implementing things such as control inputs
    modal = WithDefaultField(Direction.internal)
    #: additional time-varying outputs :math:`y`
    dynamic_output = AssignedField(Direction.output)


@dataclass
class TrajectoryAnalysisMetaData(SubmodelMetaData):
    events: list = field(default_factory=list)
    modes: list = field(default_factory=list)


class TrajectoryAnalysisType(SubmodelType):
    """Handle kwargs for including/excluding events (also need to include/exlcude
    modes?), injecting bound events (event functions, updates) to model, etc.

    A common use case will be to bind the parameters then only update the state...


    """

    metadata_class = TrajectoryAnalysisMetaData

    @classmethod
    def __prepare__(
        cls,
        *args,
        include_events=None,
        exclude_events=None,
        include_modes=None,
        exclude_modes=None,
        **kwargs,
    ):
        cls_dict = super().__prepare__(*args, **kwargs)
        if exclude_events is not None and include_events is not None:
            msg = "Use only one of include or exclude events"
            raise ValueError(msg)

        if include_events is None:
            cls_dict.meta.events = list(cls_dict.meta.primary.Event)
        else:
            cls_dict.meta.events = include_events

        if exclude_events is not None:
            cls_dict.meta.events = [
                event for event in cls_dict.meta.events if event not in exclude_events
            ]

        if exclude_modes is not None and include_modes is not None:
            msg = "Use only one of include or exclude modes"
            raise ValueError(msg)

        if include_modes is None:
            cls_dict.meta.modes = list(cls_dict.meta.primary.Mode)
        else:
            cls_dict.meta.modes = include_modes

        if exclude_modes is not None:
            cls_dict.meta.modes = [
                mode for mode in cls_dict.meta.modes if mode not in exclude_modes
            ]

        return cls_dict

    def __new__(
        cls,
        *args,
        include_events=None,
        exclude_events=None,
        include_modes=None,
        exclude_modes=None,
        **kwargs,
    ):
        new_cls = super().__new__(cls, *args, **kwargs)
        return new_cls

    @classmethod
    def bind_model_fields(cls, new_cls, attrs):
        super().bind_model_fields(new_cls, attrs)
        p = new_cls.parameter.flatten()
        x = new_cls.state.flatten()

        ode_model = new_cls._meta.primary
        model = new_cls
        control_subs_pairs = {
            control.backend_repr: [control.default] for control in ode_model.modal
        }
        for mode in model._meta.modes:
            for act in mode.action:
                control_subs_pairs[act.match.backend_repr].insert(
                    -1, (mode.condition, act.backend_repr)
                )
        control_sub_expression = {}
        for k, v in control_subs_pairs.items():
            control_sub_expression[k] = substitute(if_else(*v), control_sub_expression)

        new_cls._meta.initial_condition_function = expression_to_operator(
            [p],
            substitute(new_cls.initial.flatten(), control_sub_expression),
            f"{new_cls.__name__}_initial_condition",
        )

        new_cls._meta.state_equation_function = expression_to_operator(
            [new_cls.t, x, p],
            substitute(new_cls.dot.flatten(), control_sub_expression),
            f"{new_cls.__name__}_state_equation",
        )
        new_cls._meta.output_equation_function = expression_to_operator(
            [new_cls.t, x, p],
            substitute(new_cls.dynamic_output.flatten(), control_sub_expression),
            f"{new_cls.__name__}_output_equation",
        )
        new_cls._meta.modal_eval = expression_to_operator(
            [new_cls.t, x, p],
            substitute(new_cls.modal.flatten(), control_sub_expression),
            f"{new_cls.__name__}_modal_evaluation",
        )


class TrajectoryAnalysis(
    # ModelTemplate,
    SubmodelTemplate,
    model_metaclass=TrajectoryAnalysisType,
    primary=ODESystem,
    copy_fields=True,
    copy_embedded_models=False,
):
    """Simulation of an :class:`ODESystem`

    The trajectory analysis specifies the parameters for numerical integration of the
    ODE system.

    The parameters, final states and state rates, and final time from the ODE system are
    all available for use in expressions for ``trajectory_output``.
    """

    #: additional output calculated from the terminal state and/or integrand terms
    trajectory_output = TrajectoryOutputField()
    #: final time; may not be reached if the system has a terminating :class:`Event`
    #: occuring before ``tf``
    tf = placeholder(default=np.inf)  # TODO use placeholder with default = None
    #: initial time (default 0)
    t0 = placeholder(default=0.0)

    # TODO: how to make trajectory outputs that depend on other state's outputs without
    # adding an accumulator state and adding the updates to each event? Maybe that
    # doesn't make sense...

    @classmethod
    def initial_condition(cls, *args, **kwargs):
        """should initial condition be x0(t, p) not just p?
        bind dynamic output, time, modal? or do point analysis there?
        but maybe still time?
        """
        self = cls._meta.primary.__new__(cls._meta.primary)
        pp = cls.function_call_to_fields([cls.parameter], *args, **kwargs)[0]
        p = pp.flatten()
        self.input_kwargs = pp.asdict()
        x0 = cls._meta.initial_condition_function(p)
        self.bind_field(cls.state.wrap(x0))
        return self

    @classmethod
    def point_analysis(cls, t, *args, **kwargs):
        """Compute the state rates for the ODESystems that were bound (at the time of
        construction).
        """
        self = cls._meta.primary.__new__(cls._meta.primary)
        xx, pp = cls.function_call_to_fields(
            [cls.state, cls.parameter], *args, **kwargs
        )
        x = xx.flatten()
        p = pp.flatten()

        self.input_kwargs = dict(t=t, **xx.asdict(), **pp.asdict())

        dot = cls._meta.state_equation_function(t, x, p)
        yy = cls._meta.output_equation_function(t, x, p)
        modals = cls._meta.modal_eval(t, x, p)

        self.bind_field(xx)
        self.bind_field(pp)
        self.bind_field(cls.dot.wrap(dot))
        self.bind_field(cls.dynamic_output.wrap(yy))
        self.bind_field(cls.modal.wrap(modals))
        # self.bind_embedded_models()
        return self

        # bind paramaeters, state, call implementation functions (dot, dynamic output)

        # apply to dynamic output, as well? but needs the embedded models? hmm...
        # and I guess should similarly bind the events? -- a dictionary for update,
        # OR maybe dot/dynamic output is more simple, it's useful for trimming /other
        # solvers and doesn't need additional use case? there are potentially lots of
        # places embedded models came from -- I guess just events and modals?


# TODO: need to exlcude fields, particularly dot, initial, etc.
# define which fields get lifted completely, which become "read only" (can't generate
# new state) etc.
# maybe allow creation of new parameters (into ODESystem parameter field), access to
# state, etc.


class EventType(SubmodelType):
    @classmethod
    def process_condor_attr(cls, attr_name, attr_val, new_cls):
        state_elem = new_cls._meta.primary.state.get(backend_repr=attr_val)
        if state_elem != []:
            primary_attr = getattr(new_cls._meta.primary, state_elem.name, None)
            if primary_attr is None:
                check_attr_name(state_elem.name, attr_val, new_cls._meta.primary)
                setattr(new_cls._meta.primary, state_elem.name, state_elem)
                new_cls._meta.primary._meta.user_set[state_elem.name] = attr_val
            elif primary_attr is not state_elem:
                msg = (
                    f"{new_cls} attempting to assign state {attr_name} = {attr_val}"
                    f" but {new_cls._meta.primary} already has {attr_name} ="
                    f"{primary_attr}"
                )
                raise NameError(msg)
            if state_elem.name != attr_name:
                super().process_condor_attr(attr_name, attr_val, new_cls)
        else:
            super().process_condor_attr(attr_name, attr_val, new_cls)


class Event(
    # ModelTemplate,
    SubmodelTemplate,
    model_metaclass=EventType,
    primary=ODESystem,
):
    """Instantaneous event for :class:`ODESystem` models.

    Events may be declared at a specific time (``at_time``) or as a function with
    zero-crossings dictating the event time (``function``).
    """

    """
    update for any state that needs it

    terminate is a boolean flag to indicate whether the event terminates the simulation
    (default is False)

    function defines when event occurs OR at_time is an expression or slice of
    expressions for when the event occurs. Slice assumes constant dt between start and
    stop. use stop=None for infinite oscillator. Differs from standard slice
    semantics in that both start and stop are inclusive, so slice(0, 1, 1) will generatw
    two events at 0 and 1. This means an at_time of a single value of t_e is equiavelent
    to slice(t_e, t_e, None)


    How to bind numerical evaluation of model?
    During trajectory analysis model construction, currently existing events get copied
    to a local namespace and implementations are created, <trajectory analysis model
    name>.Events.<event model name>.
    I guess this is done by defining new  __new__ on TA, and calling super? Or do I need
    those hooks after all?
    Event implementation is owned by TA implementation, maybe in sub-name space to avoid
    extraneous implementation constructions.
    then when trajectory analysis is evaluated and bound, add events... somewhere. Maybe
    the _res.e elements get replaced by the evaluated Event models (so one for each
    occurance of the event)??? function isn't necessary, so maybe get an index instead.

    """
    # TODO: singleton field event.function is very similar to objective in
    # OptimizationProblem. And at_time. Need to be able to define such singleton
    # assignment fields by name so there's no clash for repeated symbols.

    #: instantaneous updates for any states in the ODE system
    update = MatchedField(
        ODESystem.state,
        direction=Direction.output,
        default_factory=pass_through,
    )
    # make[mode_var] = SomeModeSubclass
    # actually, just update it
    # make = MatchedField(ODESystem.finite_state)
    # terminate = True -> return nan instead of update?

    #: flag to specify whether the event should terminate the simulation (overriding
    #: ``tf`` on the :class:`TrajectoryAnalysis`)
    terminate = placeholder(default=False)
    #: expression where zero-crossings indicate event times
    function = placeholder(default=np.nan)
    #: time at which the event occurs; periodic events may be specified with a slice
    #: object (stop=None for infinite) with both start and stop inclusive
    at_time = placeholder(default=np.nan)


# this should just provide the capabiility to overwrite make (or whatever sets control)
# and dot based on condition...
# needs to inject on creation? Or is TrajectoryAnalysis implementation expected to
# iterate Modes and inject? Then can add dot and make to copy_fields
class Mode(
    # ModelTemplate,
    SubmodelTemplate,
    primary=ODESystem,
):
    """Conditional behavior of dynamics and/or controls for :class:`ODESystem`"""

    """
    convenience for defining conditional behavior for state dynamics and/or controls
    depending on `condition`. No condition to over-write behavior, essentially a way to
    do inheritance for ODESystems which is otherwise hard? Can this be used instead of
    deferred subsystems? Yes but only for ODESystems..
    """

    #: expression for triggering the mode
    condition = placeholder(default=1.0)
    #: behaviors of the declared ``modal``\s in this mode
    action = MatchedField(
        ODESystem.modal,
        direction=Direction.internal,
        default_factory=zero_like,
    )


def copy_field(new_model_name, old_field, new_field=None):
    if new_field is None:
        new_field = old_field.inherit(new_model_name, field_type_name=old_field._name)
    new_field._elements = [sym for sym in old_field]
    return new_field


class ExternalSolverWrapperType(ModelTemplateType):
    """"""

    """
    since this is only one case, only need to bind input and output fields
    explicitly -- and possibly this is just syntax sugar. wrapper doesn't need
    much from the metclass, primarily handled by ExternalSolverModel which is
    automatically generated by wrapper's __create_model__, injected automatically
    by model sub-types. Could just as easily create decorator or other
    class-method (on Wrapper) that consumes singleton/collection of functions

    IO fields just a nice way to create simple object with IO metadata.
    """

    @classmethod
    def __prepare__(cls, model_name, bases, name="", **kwds):
        log.debug(
            f"ExternalSolverWrapperType prepare for cls={cls}, "
            f"model_name={model_name}, bases={bases}, name={name}, kwds={kwds}"
        )
        if name:
            model_name = name
        sup_dict = super().__prepare__(model_name, bases, **kwds)
        # TODO should check MRO, I guess?
        if cls.baseclass_for_inheritance is not None and ExternalSolverWrapper in bases:
            for field_name in ["input", "output"]:
                sup_dict[field_name] = copy_field(
                    model_name, getattr(ExternalSolverWrapper, field_name)
                )
        return sup_dict

    def __call__(self, *args, **kwargs):
        log.debug(
            f"ExternalSolverWrapperType __call__ for cls={self}, *args={args}, "
            f"**kwargs={kwargs}"
        )
        # gets called on instantiation of the user wrapper, so COULD return the
        # condor model instead of the wrapper class -- perhaps this is more condoric,
        # not sure what's preferable
        # actully, this can get used instead of create model with init wrapper? no,
        # don't have access to instance yet.
        # print(cls, "__call__")
        wrapper_object = super().__call__(*args, **kwargs)
        return wrapper_object.condor_model


class ExternalSolverModel(ModelTemplate):
    input = FreeField()
    output = FreeField(Direction.output)


class ExternalSolverWrapper(
    ModelTemplate,
    metaclass=ExternalSolverWrapperType,
):
    input = FreeField()
    output = FreeField(Direction.output)

    def __init_subclass__(cls, singleton=True, **kwargs):
        log.debug(
            f"ExternalSolverWrapper init subclass, cls={cls}, singleton={singleton}, "
            f"kwargs={kwargs}"
        )
        # at this point, fields  are already bound by ExternalSolverWrapperType.__new__
        # but modifications AFTER construction can be doen here
        # print("init subclass of", cls)
        cls.__original_init__ = cls.__init__
        cls.__init__ = cls.__create_model__

    def __create_model__(self, *args, condor_model_name="", **kwargs):
        log.debug(
            f"ExternalSolverWrapper create model self={self}, args={args}, "
            f"condor_model_name={condor_model_name}, kwargs={kwargs}"
        )
        # print("create model of", self, self.__class__)
        # copy field so that any field modification by __original_init__ is onto the
        # copy
        # print("copying IO fields to", condor_model_name)
        if not condor_model_name:
            # could use repr somehow? but won't exist yet...
            condor_model_name = self.__class__.__name__
        for field_name in ["input", "output"]:
            setattr(
                self,
                field_name,
                copy_field(condor_model_name, getattr(self, field_name)),
            )

        self.__original_init__(*args, **kwargs)
        # update and/or copy meta? -- no, create a __condor_model__ class which is the
        # actual model and call, etc. get mapped to that??

        attrs = ExternalSolverModel.__prepare__(
            condor_model_name, (ExternalSolverModel,)
        )
        # this copying feels slightly redundant...
        for field_name in ["input", "output"]:
            copy_field(
                condor_model_name,
                getattr(self, field_name),
                new_field=attrs[field_name],
            )

        self.condor_model = ExternalSolverModel.__class__(
            condor_model_name, (ExternalSolverModel,), attrs
        )
        self.condor_model._meta.external_wrapper = self
        self.input = self.condor_model.input
        self.output = self.condor_model.output


class TableLookup(ExternalSolverWrapper):
    """Spline interpolation for gridded data. Currently for structured, rectilinear
    data."""

    def __init__(self, xx, yy, degrees=3, bcs=(-1, 0)):
        """Construct a :class:`~condor.contrib.TableLookup` model from data

        Parameters
        ----------
        xx : dict
            a dictionary where keys are input element names and values are the grid
            points for data on that axis, so the collection of values can be passed to
            ``np.meshgrid`` to construct a rectilinear grid; input should be a single
            dimension.
        yy : dict
            a dictionary where keys are output element names and values are the values
            on the input grid to be interpolated
        degrees : float or array-like, optional
            Degree of interpolant for each axis (or broadcastable). Default is 3.
        bcs : array-like, optional
            Array of boundary conditions broadcastable to ``(xdim, 2, 2)``. ``bc[xdim, 0
            for left | 1 for right, :] = (order, value)`` for specifying derivative
            conditions. Use (-1, 0) to specify the not-a-knot boundary conditions
            (default), where the boundary polynomial is the same as it neighbor.
        """
        input_data = []
        for k, v in xx.items():
            self.input(name=k)
            input_data.append(v)
        output_data = []
        for k, v in yy.items():
            self.output(name=k)
            output_data.append(v)
        output_data = np.stack(output_data, axis=-1)
        self.interpolant = ndsplines.make_interp_spline(
            input_data,
            output_data,
            degrees=degrees,
            bcs=bcs,
        )
        self.jac_interps = [
            self.derivative_or_zero(self.interpolant, idx)
            for idx in range(self.interpolant.xdim)
        ]
        self.hess_interps = [
            [
                self.derivative_or_zero(interpolant, idx)
                for idx in range(interpolant.xdim)
            ]
            for interpolant in self.jac_interps
        ]

    @staticmethod
    def derivative_or_zero(interpolant, idx):
        """convenience function for creating a derivative in a particular direction or
        a 0-interpolant if the degree in that direction is 0"""
        if interpolant.degrees[idx] > 0:
            return interpolant.derivative(idx)
        return ndsplines.make_interp_spline(
            np.zeros(1), np.zeros((1, interpolant.ydim)), degrees=0
        )
        raise ValueError

    def function(self, inputs):
        """evaluate the table-interpolating spline at :attr:`xx`"""
        xx = inputs.flatten()
        return self.interpolant(np.array(xx).reshape(-1))[0, :]  # .T

    def jacobian(self, inputs):
        """evaluate the dense jacobian at :attr:`xx`"""
        xx = inputs.flatten()
        array_vals = [
            interp(np.array(xx).reshape(-1))[0, :] for interp in self.jac_interps
        ]
        # TODO -- original implementation did not have transpose, but generic version
        # needs it
        # EVEN WORSE, adding hessian capability makes it want to have transpose again??
        # some weird casadi issue I assume... :(
        # changing API of casadi's FunctionToOperator to return the value (and letting
        # casadi-specific do casadi-specific thing) means don't transpose?
        return_val = np.stack(array_vals, axis=1)
        return return_val

    def hessian(self, inputs):
        """evaluate the dense hessian at :attr:`xx`"""
        xx = inputs.flatten()
        array_vals = np.stack(
            [
                np.stack(
                    [interp(np.array(xx).reshape(-1))[0, :] for interp in interp_row],
                    axis=0,
                )
                for interp_row in self.hess_interps
            ],
            axis=1,
        )
        return array_vals
