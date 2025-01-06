from rockit import *
from casadi import *

from rockit import *
from numpy import cos, random
import matplotlib.pyplot as plt

import csv

# Define constants
Nsim  = 10 # number of simulation steps
nx    = 2 # the system is composed of 2 states
nu    = 1 # the system has 1 input
power = 0.0015
p_min, p_max = -1.2, 0.6
v_min, v_max = -0.07, 0.07
goal_position = 0.45
N = 50  # number of control intervals

# Logging variables

time_hist = np.zeros((Nsim + 1, N + 1))
p_hist = np.zeros((Nsim + 1, N + 1))
v_hist = np.zeros((Nsim + 1, N + 1))
F_hist = np.zeros((Nsim + 1, N + 1))

tracking_error = np.zeros((Nsim + 1, 1))

# Set OCP
#ocp = Ocp(T=10.0)
ocp = Ocp(T=FreeTime(100.0))

# Define states
p = ocp.state() # position
v = ocp.state() # velocity

# Define controls
F = ocp.control(order=0)

# Specify ODE
ocp.set_der(p, v)
ocp.set_der(v, F * power - 0.0025 * cos(3 * p))

# Set initial position parameter
initial_position = random.uniform(-0.6, -0.4)
initial_velocity = 0.0  # starting at rest

# Apply initial conditions
ocp.set_initial(p, initial_position)
ocp.set_initial(v, initial_velocity)

# Ensure the initial conditions are enforced
ocp.subject_to(ocp.at_t0(p) == initial_position)
ocp.subject_to(ocp.at_t0(v) == initial_velocity)

# Path constraints
ocp.subject_to(-1 <= (F<= 1))
ocp.subject_to(p_min <= (p <= p_max))
ocp.subject_to(v_min <= (v <= v_max))

# Objective
ocp.add_objective(ocp.integral(0.1 * F**2 + (p - goal_position)**2))
#ocp.add_objective(ocp.T)

# Terminal constraints
ocp.subject_to(ocp.at_tf(p) >= goal_position)
ocp.subject_to(ocp.at_tf(v) == 0)

# Pick a solver
ocp.solver('ipopt')
#options = {"ipopt": {"print_level": 0}}
#options["expand"] = True
#options["print_time"] = False
#ocp.solver('ipopt',options)

# Make it concrete for this ocp
ocp.method(MultipleShooting(N=N, M=1, intg='rk',
                            grid=FreeGrid(min=0.05, max=2)))

# Solve the optimization problem
sol = ocp.solve()

# Get discretised dynamics as CasADi function to simulate the system
Sim_system_dyn = ocp._method.discrete_system(ocp)

# Log data for post-processing
t_sol, p_sol = sol.sample(p, grid='control')
t_sol, v_sol = sol.sample(v, grid='control')
t_sol, F_sol = sol.sample(F, grid='control')

time_hist[0, :] = t_sol
p_hist[0, :] = p_sol
v_hist[0, :] = v_sol
F_hist[0, :] = F_sol

tracking_error[0] = sol.value(ocp.objective)

# CSV file to log position, velocity, and force
with open('ocp_simulation_data.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Position", "Velocity", "Force"])

  # Simulate the MPC solving the OCP (with the updated state) several times

    for i in range(Nsim):
      print("timestep", i + 1, "of", Nsim)

      # Use the first control input directly
      current_U = F_sol[0]

      # Simulate dynamics (applying the first control input) and
      # update the current state
      current_X = Sim_system_dyn(x0=vertcat(p_sol[0], v_sol[0]), u=F_sol[0],
                                  T=t_sol[1] - t_sol[0])["xf"]

      # Update initial condition for the next MPC step
      ocp.set_initial(p, current_X[0])
      ocp.set_initial(v, current_X[1])

      # Solve the optimization problem
      sol = ocp.solve()

      # Log data for post-processing
      t_sol, p_sol = sol.sample(p, grid='control')
      t_sol, v_sol = sol.sample(v, grid='control')
      t_sol, F_sol = sol.sample(F, grid='control')

      time_hist[i + 1, :] = t_sol
      p_hist[i + 1, :] = p_sol
      v_hist[i + 1, :] = v_sol
      F_hist[i + 1, :] = F_sol

      tracking_error[i + 1] = sol.value(ocp.objective)
      print('Tracking error f', tracking_error[i + 1])

      # CSV file to log position, velocity, and force
      for j in range(len(p_sol)):
        writer.writerow([p_sol[j], v_sol[j], F_sol[j]])

      print('Tracking error f', tracking_error[i + 1])

      ocp.set_initial(p, p_sol)
      ocp.set_initial(v, v_sol)
      ocp.set_initial(F, F_sol)

# Plot the results

# Create subplots
plt.figure(figsize=(12, 10))

# Plot position
plt.subplot(3, 1, 1)
plt.plot(time_hist[-1], p_hist[-1], 'b-', label="Position (p)", linewidth=2)
plt.axhline(goal_position, color='r', linestyle='--', label="Goal Position",
            linewidth=2)
plt.title("Position vs Time")
plt.xlabel("Time")
plt.ylabel("Position")
plt.legend()
plt.grid(True)

# Plot velocity
plt.subplot(3, 1, 2)
plt.plot(time_hist[-1], v_hist[-1], 'g-', label="Velocity (v)", linewidth=2)
plt.title("Velocity vs Time")
plt.xlabel("Time")
plt.ylabel("Velocity")
plt.legend()
plt.grid(True)

# Plot Force
plt.subplot(3, 1, 3)
plt.plot(time_hist[-1], F_hist[-1], 'm-', label="Control Force (F)",
         linewidth=2)
plt.title("Control Force vs Time")
plt.xlabel("Time")
plt.ylabel("Control Force")
plt.legend()
plt.grid(True)

# Adjust layout
plt.tight_layout()
plt.show()

# Plot tracking error in a separate figure
plt.figure(figsize=(10, 6))
plt.plot(range(Nsim + 1), tracking_error, marker='o', color='red',
         linestyle='-', label='Tracking Error', linewidth=2)
plt.title('Tracking Error over Time')
plt.xlabel('MPC Step')
plt.ylabel('Tracking Error')
plt.grid(True)
plt.show()
