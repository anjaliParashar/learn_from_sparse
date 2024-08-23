"""

Path tracking simulation with LQR speed and steering control

author Atsushi Sakai (@Atsushi_twi)

"""
import sys
sys.path.append('/home/anjali/work/SQP/nla_falsifier/LQR')
import numpy as np
import matplotlib.pyplot as plt
import cubic_spline
from dynamics import Toy
# === Parameters =====

# LQR parameter
lqr_Q = np.eye(5)
lqr_R = np.eye(2)
dt = 0.001  # time tick[s]
L = 0.5  # Wheel base of the vehicle [m]
max_steer = np.deg2rad(45.0)  # maximum steering angle[rad]


class LQR():
    def __init__(self,lqr_Q = np.eye(5),lqr_R = np.eye(2)):
        self.Q = 10*lqr_Q
        self.R = lqr_R

    def solve_dare(self,A, B, Q, R):
        """
        solve a discrete time_Algebraic Riccati equation (DARE)
        """
        x = Q
        x_next = Q
        max_iter = 150
        eps = 0.01

        for i in range(max_iter):
            x_next = A.T @ x @ A - A.T @ x @ B @ \
                    np.linalg.inv(R + B.T @ x @ B) @ B.T @ x @ A + Q
            if (abs(x_next - x)).max() < eps:
                break
            x = x_next
        return x_next

    def dlqr(self,A, B, Q, R):
        """Solve the discrete time lqr controller.
        x[k+1] = A x[k] + B u[k]
        cost = sum x[k].T*Q*x[k] + u[k].T*R*u[k]
        # ref Bertsekas, p.151
        """
        # first, try to solve the ricatti equation
        X = self.solve_dare(A, B, Q, R)

        # compute the LQR gain
        K = np.linalg.inv(B.T @ X @ B + R) @ (B.T @ X @ A)
        eig_result = np.linalg.eig(A - B @ K)
        return K, X, eig_result[0]
    
    def pi_2_pi(self,angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def global_to_curvilinear(self,state,state_ref):
        print('state_ref:',state_ref.x)
        print('state:',state.x)
        rho = 0.01
        dx = np.power((state_ref.x-state.x),2) + np.power((state_ref.y-state.y),2)
        min_d = -np.logsumexp(-rho*dx,0)/rho
        ind = list(dx).index(min_d)
        dist = np.sqrt(min_d)
        dxl = state_ref.x[ind] - state.x
        dyl = state_ref.y[ind] - state.y
        return ind, min_d
    
    def calc_nearest_index(self,state, cx, cy, cyaw):
        
        dx = [state.x - icx for icx in cx]
        dy = [state.y - icy for icy in cy]

        d = [idx ** 2 + idy ** 2 for (idx, idy) in zip(dx, dy)]

        mind = min(d)

        ind = d.index(mind)

        mind = np.sqrt(mind)

        dxl = cx[ind] - state.x
        dyl = cy[ind] - state.y

        angle = self.pi_2_pi(cyaw[ind] - np.arctan2(dyl, dxl))
        if angle < 0:
            mind *= -1
        return ind, mind
    
    def lqr_speed_steering_control(self,state,state_ref, v_ref, pe, pth_e,k):
        e = np.power((state_ref.x-state.x),2) + np.power((state_ref.y-state.y),2)
        th_e = self.pi_2_pi(state.yaw - state_ref.yaw)
        v = state.v
        A = np.zeros((5, 5))
        A[0, 0] = 1.0
        A[0, 1] = dt
        A[1, 2] = v
        A[2, 2] = 1.0
        A[2, 3] = dt
        A[4, 4] = 1.0

        B = np.zeros((5, 2))
        B[3, 0] = v / L
        B[4, 1] = dt
        Q = self.Q
        R = self.R
        K, _, _ = self.dlqr(A, B, Q, R)

        x = np.zeros((5, 1))
        x[0, 0] = np.sqrt(e)
        x[1, 0] = (e - pe) / dt
        x[2, 0] = th_e
        x[3, 0] = (th_e - pth_e) / dt
        x[4, 0] = v - v_ref
        ustar = -K @ x

        # calc steering itorchut
        ff = np.arctan2((L * k), 1)  # feedforward steering angle
        fb = self.pi_2_pi(ustar[0, 0])  # feedback steering angle
        delta = ff + fb

        # calc accel itorchut
        accel = ustar[1, 0]

        return delta, e, th_e, accel


    def lqr_speed_steering_control_curv(self,state, cx, cy, cyaw, ck, pe, pth_e, sp):
        ind, e = self.calc_nearest_index(state, cx,cy,cyaw)
        tv = sp[ind]
        k = ck[ind]
        v = state.v
        th_e = self.pi_2_pi(state.yaw - cyaw[ind])

        # A = [1.0, dt, 0.0, 0.0, 0.0
        #      0.0, 0.0, v, 0.0, 0.0]
        #      0.0, 0.0, 1.0, dt, 0.0]
        #      0.0, 0.0, 0.0, 0.0, 0.0]
        #      0.0, 0.0, 0.0, 0.0, 1.0]
        A = np.zeros((5, 5))
        A[0, 0] = 1.0
        A[0, 1] = dt
        A[1, 2] = v
        A[2, 2] = 1.0
        A[2, 3] = dt
        A[4, 4] = 1.0

        # B = [0.0, 0.0
        #     0.0, 0.0
        #     0.0, 0.0
        #     v/L, 0.0
        #     0.0, dt]
        B = np.zeros((5, 2))
        B[3, 0] = v / L
        B[4, 1] = dt
        Q = self.Q
        R = self.R
        K, _, _ = self.dlqr(A, B, Q, R)

        # state vector
        # x = [e, dot_e, th_e, dot_th_e, delta_v]
        # e: lateral distance to the path
        # dot_e: derivative of e
        # th_e: angle difference to the path
        # dot_th_e: derivative of th_e
        # delta_v: difference between current speed and target speed
        x = np.zeros((5, 1))
        x[0, 0] = e
        x[1, 0] = (e - pe) / dt
        x[2, 0] = th_e
        x[3, 0] = (th_e - pth_e) / dt
        x[4, 0] = v - tv

        # itorchut vector
        # u = [delta, accel]
        # delta: steering angle
        # accel: acceleration
        ustar = -K @ x

        # calc steering itorchut
        ff = np.arctan2(L * k, 1)  # feedforward steering angle
        fb = self.pi_2_pi(ustar[0, 0])  # feedback steering angle
        delta = ff + fb

        # calc accel itorchut
        accel = ustar[1, 0]

        return delta, ind, e, th_e, accel, K





