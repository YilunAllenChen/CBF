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
from matplotlib.animation import FFMpegWriter

# other third party imports
import numpy as np

# Custom CBF function
from cbf import ControlBarrierFunction

# Set up formatting for the movie files
metadata = dict(title='Simulation')
writer = FFMpegWriter(fps=30, metadata=metadata)

# Instantiate the CBF instance and corresponding static dynamics.
c = ControlBarrierFunction(15)
robot_dynamics_fx = 0.1*np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]])
robot_dynamics_gx = 0.1*np.array([[1,0],[0,1],[0,0],[0,0]])


# Instantiate Robotarium object
N_robots = 4
N_obs = 6
h = 0.5


# Create initial conditions. The dimensions are (number of agents, 3(x, y, heading))
# 6 obstacle agents form a smaller circle around the center. 6 free agents form a bigger circle of the same origin.
initial_conditions_robots = np.zeros((N_robots, 3))
initial_conditions_obs = np.zeros((N_obs, 2))
center_obs = np.array([0, 0])
center_robots = np.array([0, 0, 0])
diameter = 0.6
for i in range(N_obs):
    theta = i * (2 * np.pi / N_obs)
    initial_conditions_obs[i] = center_obs + [diameter * np.cos(theta), diameter * np.sin(theta)]
for i in range(N_robots):
    theta = i * (2 * np.pi / N_robots)
    initial_conditions_robots[i] = center_robots + [0.6 * diameter * np.cos(theta) - 1.15, 0.6 * diameter * np.sin(theta), theta + (2/3 * np.pi)]
# transpose the initial condition array to comply with robotarium's requirement.
initial_conditions_robots = initial_conditions_robots.transpose()
initial_conditions_obs = initial_conditions_obs.transpose()
obs_pos=initial_conditions_obs
# instantiate robotarium.
r = robotarium.Robotarium(number_of_robots=N_robots, show_figure=True, sim_in_real_time=True, initial_conditions=initial_conditions_robots)
# Plot virtual obstacles
safety_radius = 0.06
safety_radius_marker_size = determine_marker_size(r,safety_radius) # Will scale the plotted markers to be the diameter of provided argument (in meters)
g = r.axes.scatter(initial_conditions_obs[0,:], initial_conditions_obs[1,:], s=np.pi/4*safety_radius_marker_size, marker='o', facecolors='none',edgecolors=[1,0,0],linewidth=3)
h = r.axes.scatter(1.5, 0, s=np.pi/4*safety_radius_marker_size, marker='o', facecolors=[0,1,0],edgecolors=[0,1,0],linewidth=3)
f = r.axes.scatter(0, 0, s=np.pi/4*safety_radius_marker_size, marker='o', facecolors='none',edgecolors=[1,0,0],linewidth=3)
# How many iterations do we want (about N*0.033 seconds)
iterations = 3000
T = 1/30

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
L2 = np.array([
    [-1, 0, 0, 0, 1],
    [1, -2, 0, 1, 0],
    [1, 1, -2, 0, 0],
    [1, 0, 1, -2, 0],
    [0, 0, 0, 0, 0],
])
with writer.saving(r.figure, "simulation.mp4", 100):
    for k in range(iterations):
        writer.grab_frame()
        # Get the poses of the robots and convert to single-integrator poses
        x = r.get_poses()
        x_si = uni_to_si_states(x)
        x_si = np.concatenate((x_si,np.array([[1.5],[0]])),axis=1)
        # Initialize the single-integrator control inputs
        si_velocities = np.zeros((2, N_robots))
        obs_velocities = np.zeros((2, N_obs))
        
        # For obstacle robots...
        for i in range(int(N_obs)):
            # Get the neighbors of robot 'i' (encoded in the graph Laplacian)
            j = topological_neighbors(L1, i)
            theta = - np.pi / N_obs

            # rotate the agent's desired direction so that the circle is maintained (doesn't converge).
            rotation = np.array([
                [np.cos(theta), np.sin(theta)],
                [-np.sin(theta), np.cos(theta)]
            ]) 
            obs_velocities[:, i] = np.sum(obs_pos[:, j] - obs_pos[:, i, None], 1) @ rotation * 0.05

        # For free agents...
        for i in range(int(N_robots)):
            # Get the neighbors of robot 'i' (encoded in the graph Laplacian)
            j = topological_neighbors(L2, i)
            # Compute the consensus algorithm
            si_velocities[:, i] = np.sum(x_si[:, j] - x_si[:, i, None], 1)


        # Test snippet: let all agents be equipped with custom-implemented CBF.
        # obtain and divide the states.
        obs_pos=np.concatenate((obs_pos,np.zeros((2,1))),axis=1)
        obs_velocities=np.concatenate((obs_velocities,np.zeros((2,1))),axis=1)
        obstacle_states = np.concatenate((obs_pos, obs_velocities), axis=0).transpose()
        agent_states = np.concatenate((x[:2, :], si_velocities), axis=0).transpose()
        safety_distance = 0.2
        for i in range(N_robots):
            
            danger_obstacle_states = []
            robot_state = agent_states[i]

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
        
        # Use the barrier certificate to avoid collisions (Robotarium Official)
        si_velocities = si_barrier_cert(si_velocities, x_si[:,:4])


        # Transform single integrator to unicycle
        dxu = si_to_uni_dyn(si_velocities, x)

        # Set the velocities of agents 1,...,N
        r.set_velocities(np.arange(N_robots), dxu)
        # Update positions of obstacles
        g.set_offsets(obs_pos.T)
        obs_pos=obs_pos[:,:N_obs]+T*obs_velocities[:,:N_obs]
        # Iterate the simulation
        r.step()




#Call at end of script to print debug information and for your script to run on the Robotarium server properly
r.call_at_scripts_end()
# generate movie