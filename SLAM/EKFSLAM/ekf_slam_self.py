"""
Extended Kalman Filter SLAM example

author: Atsushi Sakai (@Atsushi_twi)
"""

import math

import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))

import matplotlib.pyplot as plt
import numpy as np
from utils.angle import angle_mod
from utils.plot import *

# EKF state covariance
Cx = np.diag([0.5, 0.5, np.deg2rad(30.0)]) ** 2
Cx = np.diag([0.01, 0.01, np.deg2rad(0.5)]) ** 2 # Testing with smaller values

#  Simulation parameter
Q_sim = np.diag([0.1, np.deg2rad(1.0)]) ** 2
R_sim = np.diag([0.1, np.deg2rad(10.0)]) ** 2
Q_sim = np.diag([0.01, np.deg2rad(.05)]) ** 2 # Testing with smaller values
R_sim = np.diag([0.05, np.deg2rad(.05)]) ** 2

#Original settings
DT = 0.1  # time tick [s]
SIM_TIME = 50.0  # simulation time [s]
MAX_RANGE = 15.0  # maximum observation range
MAX_ANGLE_RANGE = math.pi  # maximum angle observation range
M_DIST_TH = 2.0  # Threshold of Mahalanobis distance for data association.
STATE_SIZE = 3  # State size [x,y,yaw]
LM_SIZE = 2  # LM state size [x,y]


DT = 1  # time tick [s]
SIM_TIME = 10.0  # simulation time [s]
MAX_RANGE =3.5  # maximum observation range
MAX_ANGLE_RANGE = math.pi / 2.0  # maximum angle observation range
M_DIST_TH = 2  # Threshold of Mahalanobis distance for data association.
STATE_SIZE = 3  # State size [x,y,yaw]
LM_SIZE = 2  # LM state size [x,y]

show_animation = True
save_fig =  None


def ekf_slam(xEst, PEst, u, z):
    # Predict
    G, Fx = jacob_motion(xEst, u)
    # print(G)
    xEst[0:STATE_SIZE] = motion_model(xEst[0:STATE_SIZE], u)
    PEst = G.T @ PEst @ G + Fx.T @ Cx @ Fx
    initP = np.eye(2)
    
    #Intermediate matrices to store the predicted state and covariance matrix
    xPredicted = xEst
    pPredicted = PEst

    # Update
    for iz in range(len(z[:, 0])):  # for each observation
        min_id = search_correspond_landmark_id(xEst, PEst, z[iz, 0:2])

        nLM = calc_n_lm(xEst)
        if min_id == nLM:
            print("New LM")
            # Extend state and covariance matrix
            xAug = np.vstack((xEst, calc_landmark_position(xEst, z[iz, :])))
            PAug = np.vstack((np.hstack((PEst, np.zeros((len(xEst), LM_SIZE)))),
                              np.hstack((np.zeros((LM_SIZE, len(xEst))), initP))))
            
            # print("PEst:\n", PEst)
            
            # print("Right side:\n",np.hstack((PEst, np.zeros((len(xEst), LM_SIZE)))))
            # print("Underside:\n", np.hstack((np.zeros((LM_SIZE, len(xEst))), initP)))
            
            # print("PAug\n", PAug)
            
            xEst = xAug
            PEst = PAug
        lm = get_landmark_position_from_state(xEst, min_id)
        y, S, H = calc_innovation(lm, xEst, PEst, z[iz, 0:2], min_id)

        K = (PEst @ H.T) @ np.linalg.inv(S)
        xEst = xEst + (K @ y)
        PEst = (np.eye(len(xEst)) - (K @ H)) @ PEst

    xEst[2] = pi_2_pi(xEst[2])

    return xEst, PEst, xPredicted, pPredicted


def calc_input():
    v = 1.0  # [m/s]
    # yaw_rate = 0.1  # [rad/s]
    yaw_rate = 0.0  # [rad/s]
    u = np.array([[v, yaw_rate]]).T
    return u


def observation(xTrue, xd, u, RFID):
    xTrue = motion_model(xTrue, u)

    # add noise to gps x-y
    z = np.zeros((0, 3))
    zTrue = np.zeros((0, 3))

    for i in range(len(RFID[:, 0])):

        dx = RFID[i, 0] - xTrue[0, 0]
        dy = RFID[i, 1] - xTrue[1, 0]
        d = math.hypot(dx, dy)
        angle = pi_2_pi(math.atan2(dy, dx) - xTrue[2, 0])
        if d <= MAX_RANGE and (angle <= MAX_ANGLE_RANGE and angle >= -MAX_ANGLE_RANGE):
            zTrue = np.vstack((zTrue, np.array([d, angle, i])))
            dn = d + np.random.randn() * Q_sim[0, 0] ** 0.5  # add noise
            angle_n = angle + np.random.randn() * Q_sim[1, 1] ** 0.5  # add noise
            zi = np.array([dn, angle_n, i])
            z = np.vstack((z, zi))

    # add noise to input
    ud = np.array([[
        u[0, 0] + np.random.randn() * R_sim[0, 0] ** 0.5,
        u[1, 0] + np.random.randn() * R_sim[1, 1] ** 0.5]]).T

    xd = motion_model(xd, ud)
    return xTrue, z, xd, ud, zTrue


def motion_model(x, u):
    F = np.array([[1.0, 0, 0],
                  [0, 1.0, 0],
                  [0, 0, 1.0]])

    B = np.array([[DT * math.cos(x[2, 0]), 0],
                  [DT * math.sin(x[2, 0]), 0],
                  [0.0, DT]])

    x = (F @ x) + (B @ u)
    return x


def calc_n_lm(x):
    n = int((len(x) - STATE_SIZE) / LM_SIZE)
    return n


def jacob_motion(x, u):
    Fx = np.hstack((np.eye(STATE_SIZE), np.zeros(
        (STATE_SIZE, LM_SIZE * calc_n_lm(x)))))

    jF = np.array([[0.0, 0.0, -DT * u[0, 0] * math.sin(x[2, 0])],
                   [0.0, 0.0, DT * u[0, 0] * math.cos(x[2, 0])],
                   [0.0, 0.0, 0.0]], dtype=float)

    G = np.eye(len(x)) + Fx.T @ jF @ Fx

    return G, Fx,


def calc_landmark_position(x, z):
    zp = np.zeros((2, 1))

    zp[0, 0] = x[0, 0] + z[0] * math.cos(x[2, 0] + z[1])
    zp[1, 0] = x[1, 0] + z[0] * math.sin(x[2, 0] + z[1])

    return zp


def get_landmark_position_from_state(x, ind):
    lm = x[STATE_SIZE + LM_SIZE * ind: STATE_SIZE + LM_SIZE * (ind + 1), :]

    return lm


def search_correspond_landmark_id(xAug, PAug, zi):
    """
    Landmark association with Mahalanobis distance
    """

    nLM = calc_n_lm(xAug)

    min_dist = []

    for i in range(nLM):
        lm = get_landmark_position_from_state(xAug, i)  # Get i-th landmark position
        y, S, H = calc_innovation(lm, xAug, PAug, zi, i) # y: inovation->error, S: innovation covariance, H: Jacobian
        min_dist.append(y.T @ np.linalg.inv(S) @ y) # Mahalanobis distance is calculated using formula: d = y^T * S^-1 * y

    min_dist.append(M_DIST_TH)  # new landmark

    min_id = min_dist.index(min(min_dist))
    # If this index corresponds to one of the existing landmarks (i.e., its Mahalanobis distance is the smallest), the function returns the index of that landmark.
    # If the minimum distance is the threshold value M_DIST_TH, the function returns an index that signals a new landmark (typically the last index in the list).

    return min_id


def calc_innovation(lm, xEst, PEst, z, LMid):
    delta = lm - xEst[0:2]
    q = (delta.T @ delta)[0, 0]
    z_angle = math.atan2(delta[1, 0], delta[0, 0]) - xEst[2, 0]
    zp = np.array([[math.sqrt(q), pi_2_pi(z_angle)]])
    y = (z - zp).T
    y[1] = pi_2_pi(y[1])
    H = jacob_h(q, delta, xEst, LMid + 1)
    S = H @ PEst @ H.T + Cx[0:2, 0:2]

    return y, S, H


def jacob_h(q, delta, x, i):
    sq = math.sqrt(q)
    G = np.array([[-sq * delta[0, 0], - sq * delta[1, 0], 0, sq * delta[0, 0], sq * delta[1, 0]],
                  [delta[1, 0], - delta[0, 0], - q, - delta[1, 0], delta[0, 0]]])

    G = G / q
    nLM = calc_n_lm(x)
    F1 = np.hstack((np.eye(3), np.zeros((3, 2 * nLM))))
    F2 = np.hstack((np.zeros((2, 3)), np.zeros((2, 2 * (i - 1))),
                    np.eye(2), np.zeros((2, 2 * nLM - 2 * i))))

    F = np.vstack((F1, F2))

    H = G @ F

    return H


def pi_2_pi(angle):
    return angle_mod(angle)

def plot_landmark_covariance_ellipse(xEst, PEst, color="-r", ax=None):
    for i in range(calc_n_lm(xEst)):
        lm_x = xEst[STATE_SIZE + i * 2]
        lm_y = xEst[STATE_SIZE + i * 2 + 1]
        lm_P = PEst[STATE_SIZE + i * 2:STATE_SIZE + i * 2 + 2, STATE_SIZE + i * 2:STATE_SIZE + i * 2 + 2]
        # print("lm_P\n", lm_P)
        plot_ellipse(lm_x, lm_y, lm_P, color=color, ax=ax)
        
def plot_ellipse(x, y, cov, color="-r",ax=None):
    eigvals, eigvecs = np.linalg.eigh(cov)
    angle = np.arctan2(eigvecs[1, 0], eigvecs[0, 0])
    width, height = 2 * np.sqrt(eigvals)
    ellipse = plt.matplotlib.patches.Ellipse((x, y), width, height, angle=np.degrees(angle), edgecolor=color, fc='None', lw=1)
    # plt.gca().add_patch(ellipse)
    if ax is None:
        plt.gca().add_patch(ellipse)
    else:
        ax.add_patch(ellipse)
        # Plot the major and minor axes
        for i in range(2):
            axis_length = np.sqrt(eigvals[i])
            axis_vector = eigvecs[:, i] * axis_length
            ax.plot([x - axis_vector[0], x + axis_vector[0]], [y - axis_vector[1], y + axis_vector[1]], color=color, lw=1)
        
def plot_observation_line(x, z, ax=None):
    for i in range(len(z[:, 0])):
        x_l = x[0, 0] + z[i, 0] * math.cos(x[2, 0] + z[i, 1])
        y_l = x[1, 0] + z[i, 0] * math.sin(x[2, 0] + z[i, 1])
        if ax is None:
            plt.plot([x[0, 0], x_l], [x[1, 0], y_l], "--k", label="Lidar Observation")
        else:
            ax.plot([x[0, 0], x_l], [x[1, 0], y_l], "--k", label="Lidar Observation")


def main():
    print(__file__ + " start!!")

    time = 0.0
    save_fig = 0

    # RFID positions [x, y]
    RFID = np.array([[10.0, -2.0],
                     [15.0, 10.0],
                     [3.0, 15.0],
                     [-5.0, 20.0]])
    
    RFID = np.array([[4.0, -2.0],
                     [5.0, -3.0],
                     [10.0, 5.0],
                     [12.0, -6.0],
                     [4.0, 8.0]])
    
    RFID = np.array([[2.0, 1.0],
                    [3.0, -1.0],
                    [3.8, 1.0],
                    [4.0, -0.5],
                    [5.1, 0.8]])

    # State Vector [x y yaw v]'
    xEst = np.zeros((STATE_SIZE, 1))
    xTrue = np.zeros((STATE_SIZE, 1))
    PEst = np.eye(STATE_SIZE)

    xDR = np.zeros((STATE_SIZE, 1))  # Dead reckoning

    # history
    hxEst = xEst
    hxTrue = xTrue
    hxDR = xTrue
    
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    # Flag to control the loop execution
    running = True

    colorbar = None

    # hxEst = np.hstack((hxEst, hxEst))
    
    while SIM_TIME >= time:
        time += DT
        u = calc_input()

        xTrue, z, xDR, ud, zTrue = observation(xTrue, xDR, u, RFID)

        xEst, PEst, xPred, PPred = ekf_slam(xEst, PEst, ud, z)

        x_state = xEst[0:STATE_SIZE]

        # store data history
        
        hxEst = np.hstack((hxEst, x_state))
        hxDR = np.hstack((hxDR, xDR))
        hxTrue = np.hstack((hxTrue, xTrue))

        
        
        if show_animation:  # pragma: no cover
            
            # ax[1].clear()
            # fig.clf() 
            # print("z\n", z)
            
           
            # Plot the main SLAM results
            ax[0].plot(RFID[:, 0], RFID[:, 1], "*k", label="Landmarks", markersize=10)
            # ax[0].plot(xEst[0], xEst[1], ".r", label="Estimated Position")
            
            plot_observation_line(xTrue, zTrue, ax=ax[0])

            # plot landmark
            # for i in range(calc_n_lm(xEst)):
            #     ax[0].plot(xEst[STATE_SIZE + i * 2],
            #             xEst[STATE_SIZE + i * 2 + 1], "xg")

            ax[0].plot(hxTrue[0, :],
                    hxTrue[1, :], "-b", label="True Trajectory")
            ax[0].plot(hxDR[0, :],
                    hxDR[1, :], "-k", label="Dead Reckoning")
            ax[0].plot(hxEst[0, :],
                    hxEst[1, :], marker = "o", markersize = 4, color = "r", label="Estimated Trajectory")
           
            
            # ax[0].grid(True)
            # Set specific x and y axis ranges
            ax[0].set_xlim([-0.2, 7.2])
            ax[0].set_ylim([-1.5, 1.5])
            # Ensure equal scaling without overriding limits
            ax[0].set_aspect('equal')
            # ax[0].axis("equal")
           # Remove axis numbering (but keep the ticks)
            ax[0].tick_params(axis='both', which='both', labelbottom=False, labelleft=False, direction='in', length=3)

            # Make sure the ticks appear on all sides (top, right, bottom, left)
            ax[0].tick_params(axis='x', which='both', direction='in', length=3, top=True)
            ax[0].tick_params(axis='y', which='both', direction='in', length=3, right=True)
            
            #Plot predicted state covariance ellips
            plot_ellipse(xPred[0], xPred[1], PPred[:2, :2], color="b", ax=ax[0])
            # Plot update state covariance ellips
            plot_ellipse(xEst[0], xEst[1], PEst[:2, :2], color="r", ax=ax[0])
            # Plot the landmark covariance ellipses
            plot_landmark_covariance_ellipse(xEst, PEst, color = 'green', ax=ax[0])

            # Plot the covariance matrix
            # if colorbar:
            #     colorbar.remove()
            # alphabets = ['X', 'Y', 'Theta', 'Xm', 'Ym']

            # Min-max normalization
            min_val = np.min(PEst)
            max_val = np.max(PEst)
            normalized_matrix = (PEst - min_val) / (max_val - min_val)
            cax = ax[1].matshow(PEst, cmap='viridis',vmin=0, vmax=1)
            # cax = ax[1].matshow(normalized_matrix, cmap='viridis',vmin=0, vmax=1)
            # colorbar = fig.colorbar(cax, ax=ax[1])
            ax[1].set_title('Covariance Matrix')
            # ax[1].set_xticklabels(['']+alphabets)
            # ax[1].set_yticklabels(['']+alphabets)

            # ax[0].set_title('EKF SLAM')
            # ax[0].legend()

            if save_fig != 8:
                # Temporarily hide ax[1] to save only ax[0]
                ax[1].set_visible(False)

                # Save only ax[0]
                plt.savefig(f"SLAM/EKFSLAM/Plots/EKF_SLAM_{save_fig}.svg", format="svg", bbox_inches="tight")  
                save_fig += 1  # Increment the counter for the next iteration

                # Make ax[1] visible again after saving
                ax[1].set_visible(True)
    
            plt.pause(0.001)
            ax[0].cla()
            ax[1].cla()   

           



if __name__ == '__main__':
    main()
