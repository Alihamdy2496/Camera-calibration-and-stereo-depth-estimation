/*
 * Description: Produce undistorted image from raw image captured from the camera.
 * Input:
 *			    |fx 0  cx|
 *			K = |0  fy cy|
 *				|0  0   1|
 *
 *		dist_coef = |k1, k2, p1, p2|
 * Option#0 - Using cv::getOptimalNewCameraMatrix and cv::initUndistortRectifyMap, which automatically
 *			produce the undistorted image for us.
 * Option#1 - Perform the reverse distortion in which an undistorted image is created by interpolating
 *			the pixel values from the distorted (input/raw) image using cv::remap.
 *			
 */
#include <string>
#include <iostream>
#include <cstdlib>
#include <opencv2/opencv.hpp>
#include <time.h>
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "std_msgs/msg/string.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "cv_bridge/cv_bridge.h"
#include "opencv2/calib3d.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/ximgproc.hpp"
using namespace std;
using namespace cv;
using namespace cv::ximgproc;

class Node1 : public rclcpp::Node
{
public:
	cv::VideoCapture cap1;//Declaring an object to capture stream of frames from default camera//
	cv::VideoCapture cap2;//Declaring an object to capture stream of frames from default camera//
	cv::Mat myImageR;//Declaring a matrix to load the frames//
	cv::Mat myImageL;//Declaring a matrix to load the frames//
    Node1():Node("elp_calibrate")
    {
		RCLCPP_INFO(this->get_logger(),"elp_calibrate node has started");
        timer_ = this->create_wall_timer(std::chrono::milliseconds(1), std::bind(&Node1::timer_cb, this) );	
		namedWindow("Video Player");//Declaring the video to show the video//
		cap1.open(2);
		cap1.set(CAP_PROP_FRAME_WIDTH, 240);//Setting the width of the video
		cap1.set(CAP_PROP_FRAME_HEIGHT, 120);//Setting the height of the video//
		cap2.open(4);
		cap2.set(CAP_PROP_FRAME_WIDTH, 240);//Setting the width of the video
		cap2.set(CAP_PROP_FRAME_HEIGHT, 120);//Setting the height of the video//
		if (!cap2.isOpened() || !cap1.isOpened() ){ //This section prompt an error message if no video stream is found//
			cout << "No video stream detected" << endl;
			return ;
   		}		
    }
	~Node1(){
		cap1.release();//Releasing the buffer memory//
		cap2.release();//Releasing the buffer memory//
	}
private:
		double fx =799.21956;
		double fy =800.62757;
		double cx =633.32867;
		double cy =371.76627;
		double skew =-2.5550388;
		double k1 = -0.38202288;   
		double k2 = 0.10694649;
		double p1 = -0.00192528;
		double p2 = -0.00304159;
		cv::Mat disparity,depth;
		double baseline= 6.20; // distance between two cameras in cm
		double focal= (fx+fy)/2; // focal length from calibration two cameras in pixels

    void timer_cb()
    {
		/*just for thest*/
		/*
		cv::Mat leftt = cv::imread("left-0000.png", cv::IMREAD_COLOR);
		cv::Mat rightt = cv::imread("right-0000.png", cv::IMREAD_COLOR);
		std::cout<<leftt.size()<<std::endl;
		cvtColor(rightt,  rightt,  COLOR_RGB2GRAY);
		cvtColor(leftt,  leftt,  COLOR_RGB2GRAY);

		rightt = calibrate(rightt);
		leftt = calibrate(leftt);
    	disparity=Compute_disparity(rightt,leftt);
		depth = (baseline * (focal)) / (disparity);
		cv::imwrite("depth_map.jpg", depth);
		*/
		cap1 >> myImageR;
		cap2 >> myImageL;
		cout<< myImageL.size();
		cout<< myImageR.size();
		cv::resize(myImageR, myImageR, cv::Size(1280,720));
		cv::resize(myImageL, myImageL, cv::Size(1280,720));
		if (myImageR.empty() || myImageL.empty()){ //Breaking the loop if no video frame is detected//
			RCLCPP_INFO(this->get_logger(),"no video stream ");
         	rclcpp::shutdown() ;
		}	
		myImageR = calibrate(myImageR);
		myImageL = calibrate(myImageL);
    	disparity=Compute_disparity(myImageR,myImageL);
		//depth = baseline * focal / disparity ,,,, disparity in pixels
		depth = (baseline * (focal)) / (disparity);

		//cvtColor(depth,  depth,  COLOR_GRAY2RGB);
 		//cv::applyColorMap(depth,depth,COLORMAP_HOT); 
	    imshow("Video Player", depth);
      	char c = (char)waitKey(1);
      	if (c == 27){ 
         	rclcpp::shutdown() ;
      	}
		  
    }
	cv::Mat Compute_disparity(cv::Mat myImageR,cv::Mat myImageL){
		cv::Mat left_for_matcher, right_for_matcher , filtered_disp, filtered_disp_vis;
		cv::Mat left_disp,right_disp;

    	Ptr<DisparityWLSFilter> wls_filter;
		cv::Ptr<cv::StereoBM> stereo = cv::StereoBM::create();
		int max_disp = 160;//160
		int vis_mult = 4.0;//scale disparity map
    	int wsize=5;//must be positive odd, be within 5..255
		double lambda= 8000.0;//80000
		double sigma  = 5.0;//5.0

		max_disp/=2;
		if(max_disp%16!=0)
			max_disp += 16-(max_disp%16);
		resize(myImageL,left_for_matcher ,Size(),0.5,0.5, INTER_LINEAR_EXACT);
		resize(myImageR,right_for_matcher,Size(),0.5,0.5, INTER_LINEAR_EXACT);


		/* *bm algorithm*
		Ptr<StereoBM> left_matcher = StereoBM::create(max_disp,wsize);
		wls_filter = createDisparityWLSFilter(left_matcher);
		Ptr<StereoMatcher> right_matcher = createRightMatcher(left_matcher);

		cvtColor(left_for_matcher,  left_for_matcher,  COLOR_BGR2GRAY);
		cvtColor(right_for_matcher, right_for_matcher, COLOR_BGR2GRAY);
		left_matcher-> compute(left_for_matcher, right_for_matcher,left_disp);
		right_matcher->compute(right_for_matcher,left_for_matcher, right_disp);
		*/

		Ptr<StereoSGBM> matcher  = StereoSGBM::create(0,max_disp,wsize);
		matcher->setUniquenessRatio(0);//0
		matcher->setDisp12MaxDiff(1000000);//1000000
		matcher->setSpeckleWindowSize(0);//0
		matcher->setP1(2*wsize*wsize);//24
		matcher->setP2(2*wsize*wsize);//96
		matcher->setMode(StereoSGBM::MODE_SGBM_3WAY);
		wls_filter = createDisparityWLSFilterGeneric(false);
		wls_filter->setDepthDiscontinuityRadius((int)ceil(0.0005*wsize));//0.5
		matcher->compute(left_for_matcher,right_for_matcher,left_disp);

        wls_filter->setLambda(lambda);
        wls_filter->setSigmaColor(sigma);
        wls_filter->filter(left_disp,myImageL,filtered_disp,right_disp);
        getDisparityVis(filtered_disp,filtered_disp_vis,vis_mult);
		return filtered_disp_vis;
	}
		// Compute the image size of the undistorted image
	void computeUndistortedBoundary(/*in*/cv::Size &image_size, 
									/*in*/cv::Mat &K,
									/*in*/cv::Mat &distortionCoef,
									/*out*/cv::Mat &undistortedCorners,
									/*out*/double &dminx,
									/*out*/double &dminy,
									/*out*/double &dmaxx,
									/*out*/double &dmaxy)
	{
		#define LARGE_IMAGE_COORD_D	(999999999.0)

		// Define the 4 image corners (original);
		cv::Mat distortedCorners(4, 1, CV_64FC2);
		distortedCorners.at<cv::Vec2d>(0,0)[0] = 0;						// Top-left
		distortedCorners.at<cv::Vec2d>(0,0)[1] = 0;
		distortedCorners.at<cv::Vec2d>(1,0)[0] = image_size.width-1;	// Top-right
		distortedCorners.at<cv::Vec2d>(1,0)[1] = 0;
		distortedCorners.at<cv::Vec2d>(2,0)[0] = image_size.width-1;	// Bottom-right
		distortedCorners.at<cv::Vec2d>(2,0)[1] = image_size.height-1;
		distortedCorners.at<cv::Vec2d>(3,0)[0] = 0;						// Bottom-left
		distortedCorners.at<cv::Vec2d>(3,0)[1] = image_size.height-1;

		// Undistorted the image corners
		cv::undistortPoints(distortedCorners, undistortedCorners, K, distortionCoef);

		// Compute the extreme corners
		dminx = LARGE_IMAGE_COORD_D;	dmaxx = -LARGE_IMAGE_COORD_D;
		dminy = LARGE_IMAGE_COORD_D;	dmaxy = -LARGE_IMAGE_COORD_D;

		for (int cor = 0; cor < 4; cor++)
		{
			// Un-normalized point coordinates
			double normalized_x, normalized_y, non_normalized_x, non_normalized_y;
			normalized_x = undistortedCorners.at<cv::Vec2d>(cor,0)[0]; 
			normalized_y = undistortedCorners.at<cv::Vec2d>(cor,0)[1];
			non_normalized_x = normalized_x * K.at<double>(0, 0) + K.at<double>(0, 2);
			non_normalized_y = normalized_y * K.at<double>(1, 1) + K.at<double>(1, 2);

			// Assign the undistorted corners the non-normalized values
			undistortedCorners.at<cv::Vec2d>(cor,0)[0] = non_normalized_x;
			undistortedCorners.at<cv::Vec2d>(cor,0)[1] = non_normalized_y;

			// Determine the two extreme points as the top left and bottom right corners
			if (non_normalized_x < dminx)
				dminx = non_normalized_x;
			if (non_normalized_y < dminy)
				dminy = non_normalized_y;
			if (non_normalized_x > dmaxx)
				dmaxx = non_normalized_x;
			if (non_normalized_y > dmaxy)
				dmaxy = non_normalized_y;
		}
	}
	// Produce distorted point (out_x, out_y) from undistort (in_x, in_y) point
	// Produce distorted point (out_x, out_y) from undistort (in_x, in_y) point
	void distortPoint(cv::Mat &K, cv::Mat &distortionCoef, 
					double in_x, double in_y, 
					/*out*/double &out_x, /*out*/double &out_y)
	{
		// Camera parameters and distortion coefficients
		double fx = K.at<double>(0, 0);
		double fy = K.at<double>(1, 1);
		double cx = K.at<double>(0, 2);
		double cy = K.at<double>(1, 2);
		double skew = K.at<double>(0, 1);

		double k1 = distortionCoef.at<double>(0, 0);
		double k2 = distortionCoef.at<double>(0, 1);
		//double k3 = distortionCoef.at<double>(0, 2);
		//double k4 = distortionCoef.at<double>(0, 3);

		double p1 = distortionCoef.at<double>(0, 3);
		double p2 = distortionCoef.at<double>(0, 4);
		//double p31 = distortionCoef.at<double>(0, 6);
		//double p32 = distortionCoef.at<double>(0, 7);

		// Compute distorted points
		double x = (in_x - in_y * skew - cx)/fx;
		double y = (in_y - cy)/fy;
		double r2 = x*x + y*y;
		double kcoef = 1 + k1*r2 + k2*r2*r2 ;//+ k3*r2*r2*r2 ;// + k4*r2*r2*r2*r2;
		double x_ = x*kcoef + (p1*(r2 + 2*x*x) + 2*p2*x*y) ;//+ (p31*(r2 + 2*x*x) + 2*p32*x*y)*r2;
		double y_ = y*kcoef + (2*p1*x*y + p2*(r2 + 2*y*y)) ;//+ (2*p31*x*y + p32*(r2 + 2*y*y))*r2;
		out_x = x_*fx + cx;
		out_y = y_*fy + cy;
	}


cv::Mat calibrate(cv::Mat distorted_image)//subscriber
	{
		// Image to perform undistort
		//std::string image_path = "right-0000.png";
		// 1. Construct the K matrix and distortion matrix
		cv::Mat K = (cv::Mat_<double>(3,3) <<	fx, skew, cx,
												0.0, fy, cy,
												0.0, 0.0, 1.0);
		cv::Mat distortionCoef = (cv::Mat_<double>(1,4) << k1, k2, p1, p2);

		// 2. Read-in the input image
		//cv::Mat distorted_image = cv::imread(image_path, cv::IMREAD_COLOR);
		//if (distorted_image.data == NULL)
		//{
		//	std::cout << "Failed to read the image " << ", terminating..." << std::endl;
		//	return  c;
		//}
		cv::Size image_size = distorted_image.size();

		// 3. Compute the new image size of the undistorted image
		//		Compute the (min_x, min_y) and (max_x, max_y) of the undistorted image from the 4 boundary points.
		cv::Mat undistortedCorners;
		double dminx, dminy, dmaxx, dmaxy;
		computeUndistortedBoundary(image_size, K, distortionCoef,
					/*out*/undistortedCorners, dminx, dminy, dmaxx, dmaxy);
		int new_image_width = int(dmaxx+0.5) - int(dminx-0.5);
		int new_image_height = int(dmaxy+0.5) - int(dminy-0.5);
		cv::Size new_image_size = cv::Size(new_image_width, new_image_height); // for undistorted_image

		// 4. Use the OpenCV's functions to compute undistorted image
		// 4.1 Get the new K matrix with 0 distortion
		cv::Mat new_K = cv::getOptimalNewCameraMatrix(K, distortionCoef, image_size, /*alpha*/1, new_image_size);
		// 4.2 Compute the undistortion and rectification transformation map
		cv::Mat map_x, map_y;
		cv::initUndistortRectifyMap(K, distortionCoef, cv::Mat(), new_K, new_image_size, CV_32FC1, map_x, map_y);
		cv::Mat undistorted_image_out1; // output undistorted image
		cv::remap(/*in*/distorted_image, /*out*/undistorted_image_out1, map_x, map_y, cv::INTER_LINEAR, cv::BORDER_TRANSPARENT);
		cv::Rect myROI(250, 155, 1450, 750);
		cv::Mat croppedImage = undistorted_image_out1(myROI);
		cv::resize(croppedImage, croppedImage, cv::Size(1280,720));
		//std::cout<<croppedImage.size();
		return croppedImage;
	}
    rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc,argv);
    rclcpp::spin(std::make_shared<Node1>());
    rclcpp::shutdown();
    return 0;
}
