from PIL import Image, ImageOps
import numpy as np
import argparse
import math
from pathlib import Path
import csv
from copy import deepcopy
import cv2 as cv
import time
from statistics import mean


class ImageProcessing():

    def __init__(self, images_folder=None):
        self.images_folder = images_folder
        self.log_file_path = "image_processing.csv"
        f = open(self.log_file_path, 'w')
        writer = csv.writer(f)
        header = ["Image name", "Radius of black dot", "Distance from center", "Time to compute"]
        writer.writerow(header)

    def analyze_images(self, path_to_images):
        self.images_folder = Path(path_to_images).glob('*.jpeg')
        detect_circle_runtimes = []
        for image in self.images_folder:
            print(image)
            im = np.array(Image.open(image))
            time_1 = time.perf_counter()
            radius, dist_from_center, circle_center = self.detect_circle(im)
            time_2 = time.perf_counter()
            time_to_compute = time_2 - time_1
            # print(time_to_compute)
            detect_circle_runtimes.append(time_to_compute)
            self.log_to_file(image, radius, dist_from_center, time_to_compute)
        # print(detect_circle_runtimes)
        mean_runtime = mean(detect_circle_runtimes)
        max_runtime = max(detect_circle_runtimes)
        print("Mean runtime is: " + str(mean_runtime) + ", max runtime is: " + str(max_runtime))


    def log_to_file(self, image_name, radius, dist_from_center, time_to_compute=None):
        f = open(self.log_file_path, 'a')
        writer = csv.writer(f)
        data = [image_name, radius, dist_from_center, time_to_compute]
        writer.writerow(data)
        f.close()

    def detect_circle(self, image_array, draw_circle=False):
        """detects a circle and returns its radius and the distance from its center point
        to the center of the frame. Takes a numpy array (the grayscale version of an image)"""

        if image_array.shape[2] == 3: # get the red channel
            # print("colour image in... extracting red channel for circle detection")
            # gray, _, _ = cv.split(image_array) # assume RGB
            # ret,gray = cv.threshold(gray,2,255,cv.THRESH_BINARY)

            gray, green, blue = cv.split(image_array) # assume RGB
            ret,gray = cv.threshold(gray,10,255,cv.THRESH_BINARY)
            ret,green = cv.threshold(green,10,255,cv.THRESH_BINARY)
            ret,blue = cv.threshold(blue,10,255,cv.THRESH_BINARY)
            gray[np.where(green==0)] = 255
            gray[np.where(blue==0)] = 255

        else:
            gray = image_array
        radius = 0
        dist_from_center = -1
        circle_center = -1
        # # To use an image, get the following code instead:
        # src = cv.imread(image)
        # gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
        # output = src.copy()
        center_x = int((gray.shape[1]) / 2 - 1)
        center_y = int((gray.shape[0]) / 2 - 1)
        image_center = np.array((center_x, center_y))
        # print(center)

        # filter image so only red is left
        # gray = self.filter_not_red(gray)
        # detect circles in the image

        # Intuition for parameters of HoughCircles:
        # image: 8-bit, single channel image. If working with a color image, convert to grayscale first.
        # method: Defines the method to detect circles in images. Currently, the only implemented method is cv2.HOUGH_GRADIENT, which corresponds to the Yuen et al. paper.
        # dp: Resolution of the accumulator array. Votes cast are binned into squares set by dp size. Set too small and only perfect circles are found, set too high and noise collaborates to vote for non-circles.
        # minDist: Minimum distance between the center (x, y) coordinates of detected circles. If the minDist is too small, multiple circles in the same neighborhood as the original may be (falsely) detected. If the minDist is too large, then some circles may not be detected at all.
        # param1: Is a number forwarded to the Canny edge detector (applied to a grayscale image) that represents the threshold1 passed to Canny(...). Canny uses that number to guide edge detection: http://docs.opencv.org/2.4/modules/imgproc/doc/feature_detection.html?highlight=canny
        # param2: Accumulator threshold value for the cv2.HOUGH_GRADIENT method. The smaller the threshold is, the more circles will be detected (including false circles). The larger the threshold is, the more circles will potentially be returned.
        # minRadius: Minimum size of the radius in pixels. Don't set minRadius and maxRadius far apart unless you want all possible circles that might be found in that range.
        # maxRadius: Maximum size of the radius (in pixels). Don't set minRadius and maxRadius far apart unless you want all possible circles found in that range.
        # circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1.2, 100) # for image size 1600x1200
        # circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1.2, 30)  # for image size 640x480
        # Not perfect. Misses some circles and captures some noise
        # circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT,  1.58, 30,
        #                           param1=110, param2=30, minRadius=10, maxRadius=45)  # for image size 300x400
        # circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT,  1.38, 30,
        #                           param1=100, param2=28, minRadius=10, maxRadius=45) 
        circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT,  1.38, 30,
                                  param1=100, param2=26, minRadius=10, maxRadius=45) 
        # ensure at least some circles were found
        if circles is not None:
            # convert the (x, y) coordinates and radius of the circles to integers
            circles = np.round(circles[0, :]).astype("int")
            # if len(circles)>0:
            #     print("Number of circles found %i " %len(circles))
            # dist_from_center = 9999999999 # if we find multiple circles, use the closest to the centre
            best_mean = 30
            for (x, y, r) in circles:
                # print(str(x) + ", " + str(y))
                # print(gray.shape)
                if x >= gray.shape[1]:
                    x = gray.shape[1]-1
                if y >= gray.shape[0]:
                    y = gray.shape[0]-1
                x_np = x
                y_np = y
                yr_low = np.max([0,y_np-(r-5)])
                yr_up = np.min([gray.shape[0]-1,y_np+(r-5)])
                xr_low = np.max([0,x_np-(r-5)])
                xr_up = np.min([gray.shape[1]-1,x_np+(r-5)])
                # print("(%i, %i), (%i,%i)" %(yr_low,yr_up,xr_low,xr_up))
                #for thresholded images
                tmp_center = np.array((x, y))  
                tmp_dis = np.linalg.norm(image_center-tmp_center)
                tmp_mean = np.mean(np.append(gray[yr_low:yr_up,x_np],gray[y_np,xr_low:xr_up]))
                # print(tmp_mean)
                # print([gray[yr_low:yr_up,x_np],gray[y_np,xr_low:xr_up]])
                if tmp_mean < best_mean: # if the current circle is closer to the centre
                    dist_from_center = deepcopy(tmp_dis)
                    radius = deepcopy(r)
                    circle_center = deepcopy(tmp_center)
                    best_mean = deepcopy(tmp_mean)
                # circle_center = np.array((x, y))    
                # dist_from_center = np.linalg.norm(image_center-circle_center)
                #for grey scale images
                # if gray[y, x].mean() < 150:
                #     radius = deepcopy(r)
                #     circle_center = np.array((x, y))
                #     dist_from_center = np.linalg.norm(image_center-circle_center)
            # reward = 0 # or small negative penalty for time step?
        return radius, dist_from_center, circle_center

    def filter_not_red(self, img):
        # convert the BGR image to HSV colour space
        hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        # obtain the grayscale image of the original image
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # set the bounds for the red hue
        lower_red = np.array([160, 100, 50])
        upper_red = np.array([180, 255, 255])

        # create a mask using the bounds set
        mask = cv.inRange(hsv, lower_red, upper_red)
        # Filter only the red colour from the original image using the mask(foreground)
        res = cv.bitwise_and(img, img, mask=mask)
        return res

    def highlight_circles(self, image):
        src = image.copy()
        radius, dist_from_center, circle_center = self.detect_circle(src)
        if dist_from_center > -1:
            cv.circle(src, circle_center, radius, (255, 0, 255), 3)
            cv.circle(src, circle_center, radius, (0, 255, 0), 2)
        return src


if __name__ == '__main__':
    i = ImageProcessing()
    # i.analyze_images('/home/lamarchecl/Downloads/saved_images_black_on_white/')
    i.analyze_images('/home/colin/Documents/repositories/visual_bender/src/saved_images/default')
    # i = ImageProcessing(image='/home/lamarchecl/Downloads/red_dot_testing.jpg')
    im = np.array(i.im)
    # im_path = '/home/lamarchecl/Downloads/saved_images_black_on_white/' \
    #           'collect_visual_datavisualBenderFourJointsGoal_ep0_step89.jpeg'
    radius, dist_from_center = i.detect_circle(im, draw_circle=True)
    print("radius is: " + str(radius) + ", distance from center is: " + str(dist_from_center))
