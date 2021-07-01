import argparse
import numpy as np
import os
import dataloader
import distortion
import distortion_improved
import distortion_improved2
import csv
import extrinsics
import homography
import intrinsics
import refinement
import refinement_improved
import refinement_improved2
import util
import visualize
import time
from scipy.optimize import curve_fit
import glob
import cv2 as cv2
import os
def K16P16_zhang_calibration(model, all_data):
    homographies = []
    for data in all_data:
        H = homography.calculate_homography(model, data)
        H = homography.refine_homography(H, model, data)
        homographies.append(H)

    # Compute intrinsics
    K = intrinsics.recover_intrinsics(homographies)

    model_hom_3d = util.to_homogeneous_3d(model)

    # Compute extrinsics based on fixed intrinsics
    extrinsic_matrices = []
    for h, H in enumerate(homographies):
        E = extrinsics.recover_extrinsics(H, K)
        extrinsic_matrices.append(E)

        # Form projection matrix
        P = np.dot(K, E)

        predicted = np.dot(model_hom_3d, P.T)
        predicted = util.to_inhomogeneous(predicted)
        data = all_data[h]
        nonlinear_sse_decomp = np.sum((predicted - data)**2)

    # Calculate radial distortion based on fixed intrinsics and extrinsics
    k = distortion_improved2.calculate_lens_distortion(model, all_data, K, extrinsic_matrices)

    # Nonlinearly refine all parameters(intrinsics, extrinsics, and distortion)
    K_opt, k_opt, extrinsics_opt = refinement_improved2.refine_all_parameters(model, all_data, K, k, extrinsic_matrices)

    return K_opt, k_opt, extrinsics_opt
def K1K2P1P2_zhang_calibration(model, all_data):
    homographies = []
    for data in all_data:
        H = homography.calculate_homography(model, data)
        H = homography.refine_homography(H, model, data)
        homographies.append(H)

    # Compute intrinsics
    K = intrinsics.recover_intrinsics(homographies)

    model_hom_3d = util.to_homogeneous_3d(model)

    # Compute extrinsics based on fixed intrinsics
    extrinsic_matrices = []
    for h, H in enumerate(homographies):
        E = extrinsics.recover_extrinsics(H, K)
        extrinsic_matrices.append(E)

        # Form projection matrix
        P = np.dot(K, E)

        predicted = np.dot(model_hom_3d, P.T)
        predicted = util.to_inhomogeneous(predicted)
        data = all_data[h]
        nonlinear_sse_decomp = np.sum((predicted - data)**2)

    # Calculate radial distortion based on fixed intrinsics and extrinsics
    k = distortion_improved.calculate_lens_distortion(model, all_data, K, extrinsic_matrices)

    # Nonlinearly refine all parameters(intrinsics, extrinsics, and distortion)
    K_opt, k_opt, extrinsics_opt = refinement_improved.refine_all_parameters(model, all_data, K, k, extrinsic_matrices)

    return K_opt, k_opt, extrinsics_opt

def K1K2_zhang_calibration(model, all_data):
    '''Perform camera calibration, including intrinsics, extrinsics,
       and distortion coefficients.

    Args:
       model: Nx2 collection of planar points in the world
       all_data: M-length list of Nx2 point sets of sensor correspondences
    Returns:
       Intrinsic matrix, distortion coefficients, and M-length list of 
       extrinsic matrices
    '''
    # model_hom = util.to_homogeneous(model)
    
    # Compute homographies for each image and run nonlinear refinement on each
    # homography
    homographies = []
    for data in all_data:
        H = homography.calculate_homography(model, data)
        H = homography.refine_homography(H, model, data)
        homographies.append(H)

    # Compute intrinsics
    K = intrinsics.recover_intrinsics(homographies)

    model_hom_3d = util.to_homogeneous_3d(model)

    # Compute extrinsics based on fixed intrinsics
    extrinsic_matrices = []
    for h, H in enumerate(homographies):
        E = extrinsics.recover_extrinsics(H, K)
        extrinsic_matrices.append(E)

        # Form projection matrix
        P = np.dot(K, E)

        predicted = np.dot(model_hom_3d, P.T)
        predicted = util.to_inhomogeneous(predicted)
        data = all_data[h]
        nonlinear_sse_decomp = np.sum((predicted - data)**2)

    # Calculate radial distortion based on fixed intrinsics and extrinsics
    k = distortion.calculate_lens_distortion(model, all_data, K, extrinsic_matrices)

    # Nonlinearly refine all parameters(intrinsics, extrinsics, and distortion)
    K_opt, k_opt, extrinsics_opt = refinement.refine_all_parameters(model, all_data, K, k, extrinsic_matrices)

    return K_opt, k_opt, extrinsics_opt



def create_vector_of_2d_points_for_each_chessboard_corner(CHECKERBOARD,chessboardsize):
    x=np.array([x/1000 for x in range(0,CHECKERBOARD[0]*chessboardsize,chessboardsize)]).astype(np.float32)
    y=np.zeros((1,CHECKERBOARD[0]),dtype=np.float32)
    t=np.zeros((1,CHECKERBOARD[0]),dtype=np.float32)
    z= np.append([x],y,axis=0).T
    z= np.array([z])
    for i in range(chessboardsize,CHECKERBOARD[1]*chessboardsize,chessboardsize):
        y2= np.ones((1,CHECKERBOARD[0]),dtype=np.float32)*i/1000
        z2= np.append([x],y2,axis=0).T
        z2= np.array([z2])
        z=np.append(z,z2,axis=0)

    z=np.reshape(z,(CHECKERBOARD[1]*CHECKERBOARD[0],2)).astype(np.float32)
    return z  

def import_all_data(location,CHECKERBOARD):
    cout=0
    vector_of_2d_img_points=[]
    images = glob.glob(location)
    for item in images:
        img = cv2.imread(item) 
        #img = img[:,280:1000]
        #img = cv2.resize(img, (920,720), interpolation = cv2.INTER_AREA)

        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray,CHECKERBOARD ,cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_NORMALIZE_IMAGE )
        if(ret==True):
            cout+=1
            corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1),criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30,0.001))
            corners2=corners2.reshape(CHECKERBOARD[0]*CHECKERBOARD[1],2)
            vector_of_2d_img_points.append(corners2)

    print("number of images", cout)
    return np.array(vector_of_2d_img_points)


def main():

    model=[]
    chessboard=(8,6)
    chessboardsize=108 # in millmeter

    #in_location='very_close/*.png'
    #in_location='mydata/*.png'
    #in_location='small_test/*.png'
    #in_location='chess_at_the_edge/*.png'
    #in_location='all/*.png'
    in_location='all_185/*.png'

    test_in_location='right/*.png'

    os.system("rm ./out/*")
    os.system("rm ./out_comp/*")

    out_location='out'
    out_comp_location='out_comp'

    model=create_vector_of_2d_points_for_each_chessboard_corner(chessboard,chessboardsize)

    tic=time.time()
    all_data=import_all_data(in_location,chessboard)
    print("time taken to import images = ",time.time()-tic, " s" )


    tic=time.time()
    #K_opt, k_opt, extrinsics_opt = K1K2_zhang_calibration(model, all_data)
    K_opt, k_opt, extrinsics_opt = K1K2P1P2_zhang_calibration(model, all_data)
    #K_opt, k_opt, extrinsics_opt = K16P16_zhang_calibration(model, all_data)

    print()
    print("time taken to calibrate = ",time.time()-tic, " s" )
    #'''
    cameraMatrix=K_opt
    print("cameraMatrix >> ",cameraMatrix)
    distCoeffs=k_opt
    print("distCoeffs >> ",distCoeffs)

    #distCoeffs[2]=k_opt[3]
    #distCoeffs[3]=k_opt[4]
    #distCoeffs[4]=k_opt[2]


    '''
    cameraMatrix=np.array([[806.33763, 0.0, 642.513432],
    [0.000000, 805.499873, 359.96411],
    [0.000000, 0.000000, 1.000000]])
    distCoeffs=np.array([[-0.391035 , 0.107098, 0.000320, 0.000612, 0.000000]])
    '''
    visualize.undistort_images(test_in_location,out_location, cameraMatrix,distCoeffs)

    visualize.undistort_images_compare(test_in_location,out_comp_location, cameraMatrix,distCoeffs)


    print('   Focal Length: [ {:.5f}  {:.5f} ]'.format(K_opt[0,0], K_opt[1,1]))
    print('Principal Point: [ {:.5f}  {:.5f} ]'.format(K_opt[0,2], K_opt[1,2]))
    print('           Skew: [ {:.7f} ]'.format(K_opt[0,1]))
    #print('     Distortion: [ {:.6f}  {:.6f} ]'.format(k_opt[0], k_opt[1]))

    #visualize.visualize_camera_frame(model, extrinsics_opt)
    #visualize.visualize_world_frame(model, extrinsics_opt)


if __name__ == '__main__':
    main()