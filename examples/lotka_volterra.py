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

import matplotlib.pyplot as plt
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

t_vals = sim.t
u_vals = sim.prey_population.squeeze()
v_vals = sim.predator_population.squeeze()

# Left plot: Time series of prey and predator populations
ax1.plot(t_vals, u_vals, "b-", label="Prey (u)")
ax1.plot(t_vals, v_vals, "r--", label="Predator (v)")
ax1.set_xlabel("Time")
ax1.set_ylabel("Population")
ax1.set_title("Lotka-Volterra: Population vs Time")
ax1.legend()
ax1.grid(alpha=0.3)

# Right plot: Phase portrait (predator vs prey)
ax2.plot(u_vals, v_vals, "g-")
ax2.plot(u_vals[0], v_vals[0], "ko", label="Start")
ax2.plot(u_vals[-1], v_vals[-1], "ks", label="End")
ax2.set_xlabel("Prey (u)")
ax2.set_ylabel("Predator (v)")
ax2.set_title("Phase Portrait")
ax2.legend()
ax2.grid(alpha=0.3)

print(sim._res.x[-1,:])
plt.tight_layout()
plt.show()
