import condor as co


class LotkaVolterra(co.ODESystem):
    prey_population = state()
    predator_population = state()

    prey_birth_rate = parameter()
    predation_rate = parameter()
    predator_death_rate = parameter()
    predator_growth_from_predation = parameter()

    dot[prey_population] = (
        prey_birth_rate * prey_population
        - predation_rate * prey_population * predator_population
    )

    dot[predator_population] = (
        -predator_death_rate * predator_population
        + predator_growth_from_predation * prey_population * predator_population
    )


class Sim(LotkaVolterra.TrajectoryAnalysis):
    t0 = 0
    tf = 10

    initial[prey_population] = 1
    initial[predator_population] = 1

    class Options:
        solver = co.implementations.sgm_trajectory.TrajectoryAnalysis.Solver.CVODE
        rtol = 1e-10
        atol = 1e-14


sim = Sim(1.5, 1.0, 3.0, 1.0)
