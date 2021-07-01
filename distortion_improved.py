import numpy as np

import util

def calculate_lens_distortion(model, all_data, K, extrinsic_matrices):
    """Calculate least squares estimate of distortion coefficients.

    Args:
       model: Nx2 planar points in the world frame
       all_data: M-length list of Nx2 sensor frame correspondences
       K: 3x3 intrinsics matrix
       exrinsic_matrices: M-length list of 3x4 extrinsic matrices
    Returns:
       Radial distortion coefficients [k0, k1]
    """
    M = len(all_data)
    N = model.shape[0]


    model = util.to_homogeneous_3d(model)
    u_c, v_c = K[0,2], K[1,2]
    # Form radius vector
    # here it converted from world coord. to nomarlized coord. and calculated r
    r = np.zeros(2 * M * N)
    for e, E in enumerate(extrinsic_matrices):
        normalized_projection = np.dot(model, E.T)
        normalized_projection = util.to_inhomogeneous(normalized_projection)

        x_normalized_proj, y_normalized_proj = normalized_projection[:, 0], normalized_projection[:, 1]
        r_i = np.sqrt(x_normalized_proj**2 + y_normalized_proj**2)
        r[e*N:(e+1)*N] = r_i
    r[M*N:] = r[:M*N]

    # Form observation vector
    # obs=[u00 u01 u02... u10 u11 u12.....v00 v01 v02...v10 v11 v12....] 
    # unrolling all u and v values (v10)v of image 1 and point 0 on that image
    obs = np.zeros(2 * M * N)
    u_data, v_data = np.zeros(M * N), np.zeros(M * N)
    for d, data in enumerate(all_data):
        u_i, v_i = data[:, 0], data[:, 1]
        u_data[d*N:(d+1)*N] = u_i
        v_data[d*N:(d+1)*N] = v_i
    obs[:M*N] = u_data
    obs[M*N:] = v_data

    # Form prediction vector
    pred = np.zeros(2 * M * N)
    pred_centered = np.zeros(2 * M * N)
    pred_centered_n1 = np.zeros(2 * M * N)
    pred_centered_n2 = np.zeros(2 * M * N)

    u_pred, v_pred = np.zeros(M * N), np.zeros(M * N)
    for e, E in enumerate(extrinsic_matrices):
        P = np.dot(K, E)
        projection = np.dot(model, P.T)
        projection = util.to_inhomogeneous(projection)
        u_pred_i = projection[:, 0]
        v_pred_i = projection[:, 1]

        u_pred[e*N:(e+1)*N] = u_pred_i
        v_pred[e*N:(e+1)*N] = v_pred_i
    pred[:M*N] = u_pred
    pred[M*N:] = v_pred
    pred_centered[:M*N] = u_pred - u_c
    pred_centered[M*N:] = v_pred - v_c
    pred_centered_n1[:M*N]=2*np.square(u_pred)+np.square(r[:M*N]) 
    pred_centered_n1[M*N:]=u_pred*v_pred*2

    pred_centered_n2[:M*N]=u_pred*v_pred*2
    pred_centered_n2[M*N:]=2*np.square(v_pred)+np.square(r[M*N:]) 
    # Form distortion coefficient constraint matrix

    #[k1 k2 p1 p2 k3]
    D_new = np.zeros((2 * M * N, 4))#>>>>>>5
    D_new[:, 0] = pred_centered * r**2
    D_new[:, 1] = pred_centered * r**4
    D_new[:, 2] = pred_centered_n1
    D_new[:, 3] = pred_centered_n2

    # Form values (difference between sensor observations and predictions)
    b = obs - pred


    D_new_inv = np.linalg.pinv(D_new)
    k_n = np.dot(D_new_inv, b)
    return k_n