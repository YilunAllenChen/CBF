'''
This program uses the standard consensus program provided in the example code as a template, and aims to verify the validity
of a custom-implemented CBF function.

In the program, 12 agents are deployed. Half of agents will be moving in a circle performing cyclic pursuit, and are considered
moving obstacles. The other half, with the CBF function enabled, will attempt to rendezvous inside the circle without running into any
obstacle agents.
'''

# Robotarium imports
import rps.robotarium as robotarium
from rps.utilities.graph import *
from rps.utilities.transformations import *
from rps.utilities.barrier_certificates import *
from rps.utilities.misc import *
from rps.utilities.controllers import *

# other third party imports
import numpy as np

# Custom CBF function
from cbf import ControlBarrierFunction

# Instantiate the CBF instance and corresponding static dynamics.
c = ControlBarrierFunction(0.5)
robot_dynamics = np.array([[0,0],[0,0]])
obstacle_dynamics = np.array([[1,0],[0,1]])


# Instantiate Robotarium object
N = 12
h = 0.5



# Create initial conditions. The dimensions are (number of agents, 3(x, y, heading))
# 6 obstacle agents form a smaller circle around the center. 6 free agents form a bigger circle of the same origin.
initial_conditions = np.zeros((12, 3))
center = np.array([0, 0, 0])
num_agents_in_circles = 6
diameter = 0.8
for i in range(6):
    theta = i * (2 * np.pi / num_agents_in_circles)
    initial_conditions[i] = center + [diameter * np.cos(theta), diameter * np.sin(theta), theta + (2/3 * np.pi)]
for i in range(6, 12):
    theta = i * (2 * np.pi / num_agents_in_circles)
    initial_conditions[i] = center + [2 * diameter * np.cos(theta), 2 * diameter * np.sin(theta), theta + (2/3 * np.pi)]
# transpose the initial condition array to comply with robotarium's requirement.
initial_conditions = initial_conditions.transpose()


# instantiate robotarium.
r = robotarium.Robotarium(number_of_robots=N, show_figure=True, sim_in_real_time=True, initial_conditions=initial_conditions)

# How many iterations do we want (about N*0.033 seconds)
iterations = 1000

# We're working in single-integrator dynamics, and we don't want the robots
# to collide or drive off the testbed.  Thus, we're going to use barrier certificates
si_barrier_cert = create_single_integrator_barrier_certificate_with_boundary(safety_radius=0.12)

# Create SI to UNI dynamics tranformation
si_to_uni_dyn, uni_to_si_states = create_si_to_uni_mapping()


# Laplacian used by the obstacle agents in cyclic pursuit.
L1 = np.array([
    [-1,  1,  0,  0,  0,  0],
    [ 0, -1,  1,  0,  0,  0],
    [ 0,  0, -1,  1,  0,  0],
    [ 0,  0,  0, -1,  1,  0],
    [ 0,  0,  0,  0, -1,  1],
    [ 1,  0,  0,  0,  0, -1],
])

# Laplacian used by free agents to rendezvous.
L2 = completeGL(int(N/2))


for k in range(iterations):

    # Get the poses of the robots and convert to single-integrator poses
    x = r.get_poses()
    x_si = uni_to_si_states(x)

    # Initialize the single-integrator control inputs
    si_velocities = np.zeros((2, N))

    # For obstacle robots...
    for i in range(int(N/2)):
        # Get the neighbors of robot 'i' (encoded in the graph Laplacian)
        j = topological_neighbors(L1, i)
        theta = - np.pi / num_agents_in_circles

        # rotate the agent's desired direction so that the circle is maintained (doesn't converge).
        rotation = np.array([
            [np.cos(theta), np.sin(theta)],
            [-np.sin(theta), np.cos(theta)]
        ]) 
        si_velocities[:, i] = np.sum(x_si[:, j] - x_si[:, i, None], 1) @ rotation

    # For free agents...
    for i in range(int(N/2), N):
        # Get the neighbors of robot 'i' (encoded in the graph Laplacian)
        j = topological_neighbors(L2, i-int(N/2)) + int(N/2)
        # Compute the consensus algorithm
        si_velocities[:, i] = np.sum(x_si[:, j] - x_si[:, i, None], 1)



    
    # Use the barrier certificate to avoid collisions (Robotarium Official)
    # si_velocities = si_barrier_cert(si_velocities, x_si)


    # Test snippet: let agent 10 (bottom left) be equipped with custom-implemented CBF.
    i = 10

    # obtain and divide the states.
    states = x[:2, :].transpose()
    robot_state = states[i]
    obstacle_states = states[:6]

    # Find the nearest obstacle. I'm only doing this because solving QP is too slow.
    # In production, we should feed in states of all other agents, rather than just the 
    # nearest agent's state.
    nearest_obstacle_state = [999, 999]
    nearest_distance = 999
    for obstacle_state in obstacle_states:
        distance = np.sqrt(sum((obstacle_state - robot_state)**2))
        if distance < nearest_distance:
            nearest_obstacle_state = obstacle_state
            nearest_distance = distance
            
    # Only trigger safe control when the robot comes too close to the obstacle. Again,
    # this is purely for performance purposes. If QP can be solved fast, we should remove
    # the if condition and always trigger the safe control function.
    if nearest_distance < 0.25:
        safe_control = c.get_safe_control(robot_state, nearest_obstacle_state, robot_dynamics, obstacle_dynamics, np.array([si_velocities[0][i], si_velocities[1][i]]))
        # override si_velocities with safe control inputs.
        si_velocities[0][i] = safe_control[0]
        si_velocities[1][i] = safe_control[1]



    # Transform single integrator to unicycle
    dxu = si_to_uni_dyn(si_velocities, x)

    # Set the velocities of agents 1,...,N
    r.set_velocities(np.arange(N), dxu)
    # Iterate the simulation
    r.step()




#Call at end of script to print debug information and for your script to run on the Robotarium server properly
r.call_at_scripts_end()
