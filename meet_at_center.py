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
c = ControlBarrierFunction(15)
robot_dynamics_fx = 0.1*np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]])
robot_dynamics_gx = 0.1*np.array([[1,0],[0,1],[0,0],[0,0]])


# Instantiate Robotarium object
N = 10
h = 0.5


# Create initial conditions. The dimensions are (number of agents, 3(x, y, heading))
# 6 obstacle agents form a smaller circle around the center. 6 free agents form a bigger circle of the same origin.
initial_conditions = np.zeros((N, 3))
center = np.array([0, 0, 0])
num_agents_in_circles = 5
diameter = 0.7
for i in range(5):
    theta = i * (2 * np.pi / num_agents_in_circles)
    initial_conditions[i] = center + [diameter * np.cos(theta), diameter * np.sin(theta), theta + (2/3 * np.pi)]
for i in range(5, N):
    theta = i * (2 * np.pi / num_agents_in_circles) + np.pi/5
    initial_conditions[i] = center + [1.5 * diameter * np.cos(theta), 1.5 * diameter * np.sin(theta), theta + (2/3 * np.pi)]
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
    [-1,  1,  0,  0,  0],
    [ 0, -1,  1,  0,  0],
    [ 0,  0, -1,  1,  0],
    [ 0,  0,  0, -1,  1],
    [ 1,  0,  0,  0, -1],
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


    # Test snippet: let all agents be equipped with custom-implemented CBF.
    # obtain and divide the states.
    states = np.concatenate((x[:2, :], si_velocities), axis=0).transpose()
    obstacle_states = states[:5]
    agent_states = states[5:]
    safety_distance = 0.2
    for i in range(int(N/2), N):
        
        danger_obstacle_states = []
        robot_state = states[i]

        # keep safe distance to obstacles
        for obstacle_state in obstacle_states:
            distance = np.sqrt(sum((obstacle_state[:2] - robot_state[:2])**2))
            if distance < safety_distance:
                danger_obstacle_states.append(obstacle_state)   

        # keep safe distance between other agents
        for agent_state in agent_states:
            distance = np.sqrt(sum((agent_state[:2] - robot_state[:2])**2))
            if distance < safety_distance and distance > 0:
                danger_obstacle_states.append(agent_state)   

        # Only trigger safe control when the robot comes too close to others
        if (len(danger_obstacle_states) > 0):    
            danger_obstacle_states = np.array(danger_obstacle_states)
            safe_control = c.get_safe_control(robot_state, danger_obstacle_states, robot_dynamics_fx, robot_dynamics_gx, np.array([si_velocities[0][i], si_velocities[1][i]]))
            #print(f"robot_state {robot_state}, nearest_obstacle_state {nearest_obstacle_state}, u0 {np.array([si_velocities[0][i], si_velocities[1][i]])}, safe_control {safe_control}")
            #input()
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
