import numpy as np
import cvxopt
import sys

class ControlBarrierFunction():
    def __init__(self, max_speed, dmin = 0.12, k = 1):
        """
        Args:
            dmin: dmin for bx
            yita: yita for bx
        """
        self.dmin = dmin
        self.k = k
        # limitation of robot speed
        self.max_speed = max_speed 
        self.gamma = 0.5

    def get_safe_control(self, robot_state, obs_states, f, g, u0):
        """
        Args:
            robot_state: np array current robot state <x, y>
            obs_state: np array dynamic obstacle state <x, y>, you may only need to focus on these too close
            f: np array, the dynamic of robot state
            g: np array, the dynamic of robot state
            u0: the original action (maybe unsafe)
            [dynamics : x_dot = f(x) + g(x)*u]
            [bx: barrier function -- h = p*(robot_state-obs_states)+q]
        Returns:
            u: the cbf modified safe control
        """
        u0 = np.array(u0).reshape((2,1))
        L_gs = []
        L_fs = []
        obs_dots = []
        reference_control_laws = []
        is_safe = True

        for i, obs_state in enumerate(obs_states):
            d = np.array(robot_state - obs_state)
            d_pos = d[:2] # pos distance
            d_abs = np.linalg.norm(d_pos)
            # the goal is to make the value of h always greater than 0
            # h measure the relative distance between robot and obstacle
            # you can think other robots as dynamic obstacles 
            # here I calculate the positive distance between them
            hs_p = np.array([1,1])
            if (d_pos[0] < 0):
                hs_p[0] = -1
            if (d_pos[1] < 0):
                hs_p[1] = -1
            
            L_f = hs_p @ (f @ d.reshape((-1,1))) # shape (1, 1)
            L_g = -hs_p @ g # shape (1, 2) g contains x information
            L_gs.append(L_g)
            reference_control_laws.append(hs_p @ d.reshape((-1,1)) - self.dmin + \
                                            L_f + np.dot(hs_p, np.dot(g, u0)))
        
        # Solve safe optimization problem
        # min_x ||x||^2   s.t. Ax <= b
        u0 = u0.reshape(-1,1)
        Q = cvxopt.matrix(np.eye(2))
        p = cvxopt.matrix(np.zeros(2).reshape(-1,1))
        G = cvxopt.matrix(np.vstack([np.eye(2), -np.eye(2)]))
        S_saturated = cvxopt.matrix(np.array([self.max_speed-u0[0][0], self.max_speed+u0[0][0], \
                                                self.max_speed-u0[1][0], self.max_speed+u0[1][0]]).reshape(-1, 1))

        L_gs = np.array(L_gs).reshape(-1, 2)
        reference_control_laws = np.array(reference_control_laws).reshape(-1,1)
        A = cvxopt.matrix([[cvxopt.matrix(L_gs), G]])
        cvxopt.solvers.options['show_progress'] = False
        cvxopt.solvers.options['maxiters'] = 600

        while True:
            try:
                b = cvxopt.matrix([[cvxopt.matrix(reference_control_laws), S_saturated]])
                sol = cvxopt.solvers.qp(Q, p, A, b)
                u = sol["x"]
                break
            except ValueError:
                for i in range(len(reference_control_laws)):
                    reference_control_laws[i][0] += 1

        u = np.array([u[0]+u0[0][0], u[1]+u0[1][0]])
        u[0] = max(min(u[0], self.max_speed), -self.max_speed)
        u[1] = max(min(u[1], self.max_speed), -self.max_speed)
        return u
        
if __name__ == "__main__":
    # max_speed is depend on robot, and dmin is user define the safety distance
    # I suggest to make dmin larger than collision distance
    cbf_controller = ControlBarrierFunction(max_speed = 0.2, dmin = 0.12) 
    robot_state = np.array([0.1, 0.1])
    unsafe_obstacles = np.array([[0.08, 0.14], [0.12, 0.09], [0.12, 0.12]])
    # dynamic model parameters, this is double integrator model
    fx = np.array([[0,0],[0,0]])
    gx = np.array([[1,0],[0,1]])
    # action generated from leader-follower control
    # we pass this action to cbf to make it safe
    # action is the acceleration on x-axis and y-axis
    action = [-0.01, 0.03]
    action = cbf_controller.get_safe_control(robot_state, unsafe_obstacles, fx, gx, action)
    print(f"action {action}")