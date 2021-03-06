import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as p3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import numpy as np
import cv2
import dataloader
import util
import glob
import time
from matplotlib import rc
from matplotlib.ticker import MultipleLocator

# From https://dawes.wordpress.com/2014/06/27/publication-ready-3d-figures-from-matplotlib/
rc('font',size=10)
rc('font',family='serif')
rc('axes',labelsize=10)
rc('lines', markersize=5)
rc('lines', markeredgewidth=10)

def make_axis_publishable(ax, major_x, major_y, major_z):
    # [t.set_va('center') for t in ax.get_yticklabels()]
    # [t.set_ha('left') for t in ax.get_yticklabels()]
    # [t.set_va('center') for t in ax.get_xticklabels()]
    # [t.set_ha('right') for t in ax.get_xticklabels()]
    # [t.set_va('center') for t in ax.get_zticklabels()]
    # [t.set_ha('left') for t in ax.get_zticklabels()]

    ax.grid(False)
    ax.xaxis.pane.set_edgecolor('black')
    ax.yaxis.pane.set_edgecolor('black')
    ax.zaxis.pane.set_edgecolor('black')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    ax.xaxis._axinfo['tick']['inward_factor'] = 0
    ax.xaxis._axinfo['tick']['outward_factor'] = 0.4
    ax.yaxis._axinfo['tick']['inward_factor'] = 0
    ax.yaxis._axinfo['tick']['outward_factor'] = 0.4
    ax.zaxis._axinfo['tick']['inward_factor'] = 0
    ax.zaxis._axinfo['tick']['outward_factor'] = 0.4
    ax.zaxis._axinfo['tick']['outward_factor'] = 0.4

    ax.xaxis.set_major_locator(MultipleLocator(major_x))
    ax.yaxis.set_major_locator(MultipleLocator(major_y))
    ax.zaxis.set_major_locator(MultipleLocator(major_z))


def visualize_camera_frame(model, extrinsics):
    model = util.to_homogeneous_3d(model)

    fig =plt.figure()

    ax = fig.add_subplot('111', projection='3d')

    make_axis_publishable(ax, 10, 10, 10)

    ax.set_title('Camera-Centric Extrinsics')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.set_xlim(-10,10)
    ax.set_ylim(-10,10)
    ax.set_zlim(0, 20)

    # From StackOverflow: https://stackoverflow.com/questions/39408794/python-3d-pyramid
    v = np.array([[-0.5, -0.5, 1], [0.5, -0.5, 1], [0.5, 0.5, 1],  [-0.5, 0.5, 1], [0, 0, 0]])

    verts = [ [v[0],v[1],v[4]], [v[0],v[3],v[4]],
     [v[2],v[1],v[4]], [v[2],v[3],v[4]], [v[0],v[1],v[2],v[3]]]

    ax.add_collection3d(Poly3DCollection(verts, 
     facecolors='cyan', linewidths=1, edgecolors='r', alpha=.25))

    for E in extrinsics:
      model_ext = np.dot(model, E.T)

      xs = model_ext[:,0]
      ys = model_ext[:,1]
      zs = model_ext[:,2]

      ax.plot_trisurf(xs, ys, zs)

    ax.invert_xaxis()

    plt.show()


def visualize_world_frame(model, extrinsics):
    model = util.to_homogeneous_3d(model)

    fig =plt.figure()

    ax = fig.add_subplot('111', projection='3d')

    make_axis_publishable(ax, 3, 3, 20)

    ax.set_title('World-Centric Extrinsics')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.set_xlim(-2,9)
    ax.set_ylim(-9,2)
    ax.set_zlim(-20, 0)

    # From StackOverflow: https://stackoverflow.com/questions/39408794/python-3d-pyramid
    v = np.array([[-0.5, -0.5, 1], [0.5, -0.5, 1], [0.5, 0.5, 1],  [-0.5, 0.5, 1], [0, 0, 0]])

    xs = model[:,0]
    ys = model[:,1]
    zs = model[:,2]

    ax.plot_trisurf(xs, ys, zs)

    v = util.to_homogeneous(v)

    for E in extrinsics:
        E = np.vstack((E, np.array([0.,0.,0.,1.])))
        E_inv = np.linalg.inv(E)
        E_inv = E_inv[:3]

        v_new = np.dot(v, E_inv.T)
 
        verts = [ [v_new[0],v_new[1],v_new[4]], [v_new[0],v_new[3],v_new[4]],
            [v_new[2],v_new[1],v_new[4]], [v_new[2],v_new[3],v_new[4]], [v_new[0],v_new[1],v_new[2],v_new[3]]]

        ax.add_collection3d(Poly3DCollection(verts, facecolors='cyan', linewidths=1, edgecolors='r', alpha=.25))

    ax.invert_xaxis()

    plt.show()

def undistort_images(in_location,out_location, cameraMatrix,distCoeffs):
    count=0
    images = glob.glob(in_location)
    for item in images:
        img = cv2.imread(item)
        img = img[:,280:1000]
        img = cv2.resize(img, (1280,720), interpolation = cv2.INTER_AREA)

        h,  w = img.shape[:2]
        # undistort
        tic=time.time()
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, (w,h), 1, (w,h))
        dst = cv2.undistort(img, cameraMatrix, distCoeffs, None, newcameramtx)
        # crop the image
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]
        dst = cv2.resize(dst, (1280,720), interpolation = cv2.INTER_AREA)

        if(count==0):
            print("time taken to undistort one single image = ",time.time()-tic, " s" )

        plt.imsave(out_location+"/out{}.jpg".format(count),dst);
        count+=1

def undistort_images_compare(in_location,out_location, cameraMatrix,distCoeffs):
    count=0
    images = glob.glob(in_location)
    for item in images:
        img = cv2.imread(item).copy()
        img = img[:,280:1000]
        img = cv2.resize(img, (1280,720), interpolation = cv2.INTER_AREA)

        h,  w = img.shape[:2]
        # undistort
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, (w,h), 1, (w,h))
        dst = cv2.undistort(img, cameraMatrix, distCoeffs, None, newcameramtx)     
         # crop the image
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]
        dst = cv2.copyMakeBorder( dst, int(abs(img.shape[0]-dst.shape[0])/2), 
                                    int(abs(img.shape[0]-dst.shape[0])/2+abs(img.shape[0]-dst.shape[0])%2), 
                                    int(abs(img.shape[1]-dst.shape[1])/2), 
                                    int(abs(img.shape[1]-dst.shape[1])/2+abs(img.shape[1]-dst.shape[1])%2),
                                    borderType=cv2.BORDER_CONSTANT)

        out=np.concatenate((img,dst),axis=1)
        plt.imsave(out_location+"/out_comp{}.jpg".format(count),out);
        count+=1
def main():
    print("calibrate.py")

if __name__ == '__main__':
    main()