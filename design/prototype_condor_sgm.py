class BaseLinCovCW:
    t = independent_variable() # maybe this is in the base class, can get over-written?

    # parameters defined in the class, hopefully accessible by scope (but maybe not
    # without scope magic)
    measurement_frequency, measurement_phase = parameters()

    # metaclass can definitely process the unpacking? or does unpacking automatically
    # create sequentially keyed items so it would just work to return 1?
    # so could make that feel really fun

    omega = constant(...) # define it as a constant? or... just set a value? lol

    # can figure out how to manage scope for parameters. stack overflow probably follows
    # similar mechanism for unpacking

    # for MDAO, parameters imply inputs?
    # 
    # demo GASPy with control flow type states (t_rotate)
    # demo orbital, iterator to create major/minor burns - 
    # how to specify state equations? maybe as a dict?? state generation defines order
    # special dict attribute state_rate[state_var] = ...
    # in this context, "output" is of trajectory?
    # could always correspond to integral of lagrange term (cost to go?) -- augmented
    # state with state rate = L (integrand)
    # can create state rate from overloading simupy but can't inherently inspect
    # interior signals, only IO which is very systems-y. May need symbolic vector to
    # dispatch operations? neat.
    # need to automate passing up of metadata from a simupy bd -- 
    # similar system output as attribute dict with state keysa, ?? o
    # can't have computational blocks?? unless they provide a derivative?
    # need to create own symbolic representation 



    # numerical representation: ~ simupy syntax
    # I guess not really needed since the whole point is to handle all the needed
    # partial derivatives
    dim_state = ...
    dim_output = ...
    dim_input = ...

    def state_equation(t, x, u):
        return ...

    # symbolic representation 1
    state = ...
    input = ...
    output = ...

    state_equation = ...


    # name the event by subclassing
    class measurement(event):
        # event.function is the event function?
        function = sin(measurement_frequency*t + measurement_phase)


        # need some symbol for returning back to "same" phase,
        # is it always the most specific subclass? probably.
        # then need a way to terminate?
        # maybe we assume there is never a "next phase" everything defined by events?
        # how to handle GASPy? 
        # just create new state, 
        next_phase = cyclic





class LinCovCW(SGM):
    t = independent_variable() # maybe this is in the base class, can get over-written?

    # in future work this maybe optimized -- not sure if there will be an API for
    # parameters we aren't optimizing over? 
    measurement_frequency, measurement_phase = constant(..., ...)

    # call state() to generate state variables. I think we can handle unpacking
    px, py, pz = state()
    vx, vy, vz = state()

    # state_rate[:] defines all currently defined states? can index like a numpy array
    # or by list of state references
    # similarly, state without a call gets all currently defined state?
    state_rate[...] = A_cw @ state
    AugCov = state(12, 12)
    state_rate[AugCov] = Acal @ AugCov + AugCov @ Acal + ...

    NavCov = state(6, 6)

    mu = state() # Delta v nominal accumulator
    sigma = state() # Delta v dispersion accumulator

    state_rate[mu] = 0. # actually a default, so don't need to do this/

    initial[state] = ....

    class measurement(Event):
        # event.function is the event function?
        function = sin(measurement_frequency*t + measurement_phase)

        # default/unset behavior for update is continuous
        update[NavCov] = ...

    class BurnTemplate(EventTemplate):
        tig = parameter_template() # don't need template suffix since declared in
        # template?
        tmf = parameter_template()
        burn_target = parameter_template(3)

        function = t - tig
        update[] = ...

    for maj_burn_idx, num_minor_burns = enumerate([0, 2, 5, 3]):

        # everything that needs to get automatically indexed gets indexed?
        # parameter with repeat names, event classes, etc.
        # parameters declared here won't have a persistent accessor by name on main
        # class; instead access throguh class.parameters

        # I guess parameters need initial guesses?
        tig_maj = parameter()
        tmf_maj = parameter()
        burn_target_maj = parameter(3)


        class MajorBurn(BurnTemplate, tig=tig_maj, tmf=tmf_maj, burn_target_maj):
            pass


        for min_burn_idx in range(num_minor_burns):
            tig_min = parameter()


            class MinorBurn(MajorBurn, tig=tig_min):
                pass



        # or...
        MajorBurn = BurnTemplate(tig=tig_maj, tmf=tmf_maj, burn_target_maj)
        for min_burn_idx in range(num_minor_burns):
            tig_min = parameter()
            MinorBurn = MajorBurn(tig=tig_min)




class AircraftMission(SGM):
    # control is ~ time-varying parameter? actually, here (and with a feedback
    # controller?) they're just deferred inputs... interesting.
    # state is time-varying
    alpha = control()
    throttle = control()
    weight, drange, altitude, gamma, airspeed  = state()


    initial[weight] = parameter()
    initial[altitude] = parameter() # take-off airport altitude
    # initial drange, gamma, airspeed = 0. by default

    # generally expressions are point-wise in time
    # implicitly or explicitly get time? 
    # aero_model I guess also needs t_gears, t_flaps if they aren't incremented
    # somewhere else?
    flight_condition = flight_condition_model(altitude, airspeed)
    lift, drag = aero_model(flight_condition, alpha)
    fuel_flow_rate, thrust = propulsion(flight_condition, throttle)

    rate[weight] = -fuel_flow_rate
    rate[airspeed] = ...

    # these states will automatically not matter since they don't directly appear in
    # output integrand of terminal terms?
    t_rot, t_gear, t_flaps = state()

    initial[t_rot, t_gear, t_flaps] = inf

    mode = finite_state()
    # logic can only depend on finite_state, mode can only be updated by events
    # can only only take on discrete values. classes inside a dynamic system model that
    # inherit from mode provide a convenience -- auto number/name like enum,
    # adds logic operator to any output thats overwritten


    class groundroll(mode):
        alpha = 0.
        throttle = propulsion.takeoff

    initial[mode] = groundroll

    class rotation(mode):
        alpha = rot_rate * (t - t_rot)
        throttle = propulsion.takeoff

    v_rot = parameter()
    class rotation_trigger(Event):
        function = airspeed - v_rot
        # time and state on RHS of update means at event time
        update[t_rot] = t
        update[mode] = rotation

    class liftoff(Event):
        function = normal_force
        update[mode] = rotating_ascent

    class rotating_ascent(rotation):
        pass

    class ascent_constraint_trigger(Event):
        function = min(
            TAS_constraint,
            xlf_constraint,
            pitch_constraint
        )
        update[mode] = rotation

    gear_retraction_altitude = parameter(50*ft) # default value??
    # TODO: defaults should get added to input table along with bounds?
    # this will be helpful for running specific problems
    class gear_retraction(Event):
        function = altitude - gear_retraction_altitude
        update[t_gear] = t

    flaps_retraction_altitude = parameter(400*ft) # default value??
    class gear_retraction(Event):
        function = altitude - flaps_retraction_altitude
        update[t_flaps] = t

    class constrained_ascent(mode):
        throttle = propulsion.takeoff
        # Actually, this is best as an OptimizationProblem :)
        alpha = min(
            solve(TAS_constraint),
            solve(xlf_constraint),
            solve(pitch_constraint)
        )

    # similar events for gear and flap retraction time
    # alpha for rotation, aero for gear/flap retraction are easy with this formulation

    # want fully determined controls? or allow functional space deriv? no, can put
    # spline with finite parameters unless study shows its worse?





class trajectory_analysis(model=some_ode_class,):
    class output1(output):
        integrand_cost = expression(model.state, control, etc)
        terminal_cost = expression(...)

    input = ....
    model.p = input
    # so trajectory analysis distinguishes between parameters (inputs?) and sets
    # constants, not the ODE!!

# really it needs to be
class MyTrajectory(TrajectoryModel):
    model = some_ode_class # not sure if this is an API or convention

    # add outputs that depend on either integral or terminal terms
    integrand_term.output1 = ... #some expressing
    terminal_term.output2 = ... # some expression

    # or both
    integrand_term.output0 = ...
    terminal_term.output0 = ...

    # in general, need to make the backend repr's available! Easiest to make model an
    # inner_to, it's just the way it gets used it's not really an inner, it's a consumer
    # really, inner mechanism is an "Attached" mechanism? 
    # TODO: rename inner to attached


# wrapping an numeric python method (no solvers)
class Wrapped(CondorModel):
    inp1 = input()
    inp2 = input()

    out = my_function_to_wrap(inp1, inp2)

    output.outname0 = out[0]
    output.outname1 = out[1]
    output.outname2 = out[2]


# for external "codes" -- depends on backend/imp API
# I guess also need to define a Model subclass?


# some kind of inheritable descriptor?
# fine for A here, don't know how to deal with x below?

class LTI(DynamicsModel):
    A = _inheritable_descriptor
    B = ...
    x = state(A.shape[0]) ???
    u = parameter(B.shape[1])
    dot[x] = A@x + B@u

class double_integrator(LTI):
    A = ...

# then Aviary becomes

class my_TTBW_analysis(Aviary):
    propulsion = ...


# probably needs to be a function that generate's what's needed?
# actually, can it be in new? or call? then just need to figure out return container
def LTI():

    def __call__(A, B=None, C=None):
        x = LTI.state(A.shape[0])
        xdot = A@x
        if B is not None:
            u = LTI.input(B.shape[1])
            xdot += B@u

        LTI.dot[x] = xdot
        if C is not None:
            output = C @ x


"""
How deferred subsystems work
"""

class propulsion_interface(Deferred):
    throttle = input()
    TAS = input()
    alt = input()
    # what if a different trajectory model wants a different IO? for example, adding
    # electric split? or required thrust? just re-type the trajectory? 
    thrust = output()
    fuel_rate = output()

class aero_interface(Deferred):
    alpha = input()
    TAS = input()
    alt = input()
    CD = output()
    CL = output()



@WithDeferredSubSystems
class GASPVehicleDynamics(ODESystem):
    propulsion = subsystem(propulsion_interface)
    aero = subsystem(aero_interface)

    distance = state()
    alt = state()
    TAS = state()
    gamma = state()

    # see dynamics model above with events etc

# Doesn't need deferral decorator since it's an attached model
class GASPTrajectory(GASPVehicleDynamics.TrajectoryModel):

    terminal.fuel_burn = weight - weight_intial
    terminal.range_flown = downrange


class CoupledSizingClosure(OptimizationModel):

    # variables can get numeric attributes like initial, bounds, scaling?
    sls = variable()
    gtow = variable()
    cruise_range = variable()
    target_range = parameter()

    # static analysis
    # several of thsese return objects that trajectory can use
    static_prop = RubberizedEngineStatic(some_fixed_deck_ref, sls)
    fuel_capacity, something_that_feeds_to_aero = GASPWeightsSizing(GTOW)
    static_aero = GASPAeroModelStatic(something_that_feeds_to_aero)

    # trajectory has the most important performance metrics
    # realize the trajectory model
    trajectory = GASPTrajectory(
        propulsion=partial(RubberizedEngineDynamic, **static_prop.parameters_to_dynamic),
        aero=partial(GASPAeroModelDynamic, **static_aero.parameters_to_static),
        initial_weight = gtow
    )

    # this doesn't even need to be wrapped, I think this can just get duck-typed and
    # turned into the right kind of expression
    total_fuel = fuel_burned_with_margin_computation(trajectory.fuel_burn)

    constraint.add(total_fuel == fuel_capcity)
    constraint.add(target_range == trajectory.range_flown)

closed_vehicle = CoupledSizingClosure(target_range=3500)
# what kind of stuff gets reported?
print(
closed_vehicle.total_fuel,
)
plot(
    closed_vehicle.trajectory.alt,
    closed_vehicle.trajectory.range,
    ...
)


# TODO: is it possible to have an accumulator term? Would need to attach to (each)
# event? A trajectory could interact with each event class, but not clear what the
# API is. basically want to add an update to each event? Could be a case for a
# MatchedField that takes two matches, an output name and an event. how to create
# accumulator ? I guess something like this?
accumulator = FreeField()
accumulate = MatchedField(accumulator, ODESystem.Event)

# then behavior is
sigma = accumulator()
accumulate[sigma, MyEvent] = ... 
# some expression of ODESystem.state and/or MyEvent.update[...]? May need to access
# backend repr if using MyEvent directly; could just re-write any expressions as
# needed and that would probably be fine

