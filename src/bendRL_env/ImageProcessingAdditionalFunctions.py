from PIL import Image, ImageOps
import numpy as np
import math
from pathlib import Path
import csv
from copy import deepcopy
import cv2 as cv
import os

class ImageProcessing():

    def __init__(self,
                 image='/home/lamarchecl/Downloads/target_images_wh/target_images/13440209-2022-08-11-152017_wh.png',
                 images_folder=None):
        self.im_path = image
        self.im = Image.open(image)
        self.rgb_im = self.im.convert('RGB')
        self.images_folder = images_folder
        self.screen_visible_color_threshold = 200
        self.log_file_path = "image_processing.csv"
        f = open(self.log_file_path, 'w')
        writer = csv.writer(f)
        header = ["Image name", "Radius of black dot", "Mean value (center)", "Mean value (whole image)",
                  "Distance from center"]
        writer.writerow(header)

    def analyze_images(self, path_to_images):
        self.images_folder = Path(path_to_images).glob('*.jpeg')
        for image in self.images_folder:
            # print(image)
            path_to_image = str(image)
            im = Image.open(image)
            # diameter = self.find_diameter(im)
            radius, dist_from_center = self.detect_circle(path_to_image)
            mean = self.find_mean_center(im)
            screen_visibility = self.is_screen_visible(im)
            # print("Diameter of white dot: " + str(diameter) +
            #       ", Mean for center 450x450 pixels: " + str(mean) +
            #       ", Mean colour value for whole image: " + str(screen_visibility))
            # reward = diameter * 0.50 + mean * 0.50
            self.log_to_file(image, radius, mean=mean, screen_visibility=screen_visibility,
                             dist_from_center=dist_from_center)

    def log_to_file(self, image_name, diameter, mean, screen_visibility, dist_from_center):
        f = open(self.log_file_path, 'a')
        writer = csv.writer(f)
        data = [image_name, diameter, mean, screen_visibility, dist_from_center]
        writer.writerow(data)
        f.close()

    def find_diameter(self, image=None):
        if image is None:
            rgb_image = self.rgb_im
        else:
            rgb_image = image.convert('RGB')

        x_init = rgb_image.size[0]/2 - 1
        # print("initial x: " + str(x_init))
        y_init = rgb_image.size[1]/2 - 1
        # print("initial y: " + str(y_init))
        r, g, b = rgb_image.getpixel((x_init, y_init))
        # print(r, g, b)
        diameter = 0
        if r > 250 and g > 250 and b > 250:
            # print("center pixel is white, finding diameter")
            x = deepcopy(x_init)
            y = deepcopy(y_init)
            r_right, g_right, b_right = rgb_image.getpixel((x + 1, y))
            r_left, g_left, b_left = rgb_image.getpixel((x - 1, y))
            r_up, g_up, b_up = rgb_image.getpixel((x, y + 1))
            r_down, g_down, b_down = rgb_image.getpixel((x, y - 1))

            # first handle the special cases

            # if nothing right or left
            if (r_right < 230 or g_right < 230 or b_right < 230) and (r_left < 230 or g_left < 230 or b_left < 230):
                # print("nothing right or left")
                pt_1 = (x, y)
                # if nothing up, search down
                if r_up < 230 or g_up < 230 or b_up < 230: # nothing up, go down
                    # print("nothing up, looking down")
                    temp_down = deepcopy(y)
                    while r > 230 and g > 230 and b > 230:
                        temp_down = deepcopy(y)
                        y -= 1
                        r, g, b = rgb_image.getpixel((x, y))
                    pt_2 = (x, temp_down)
                # if nothing down, search up
                elif r_down < 230 or g_down < 230 or b_down < 230:
                    # print("nothing down, looking up")
                    while r > 230 and g > 230 and b > 230:
                        temp_up = deepcopy(y)
                        y += 1
                        r, g, b = rgb_image.getpixel((x, y))
                    pt_2 = (x, temp_up)
                # if nothing to the right, left, up or down, we are not in a dot.
                else:
                    diameter = 0
                    return diameter
                # print("point 1 is " + str(pt_1) + ", point 2 is " + str(pt_2))
                diameter = math.sqrt((pt_2[0]-pt_1[0])**2+(pt_2[1]-pt_1[1])**2)
                return diameter

            # Then if we are somewhere inside the dot
            else:
                # step all the way right, find the point < 250 , get its coordinates (nothing right = edge found)
                temp_right = deepcopy(x)
                if r_right < 230 or g_right < 230 or b_right < 230:
                    # nothing right, edge found!
                    pt_2 = (x, y)
                else:
                    while r > 230 and g > 230 and b > 230:
                        temp_right = deepcopy(x)
                        x += 1
                        r, g, b = rgb_image.getpixel((x, y))
                    pt_2 = (temp_right, y) # last point where all pixels were above 200, right?
                # print("right edge found at:" + str(pt_2))

                # from initial pixel, step all the way left, find last white point, get coordinates
                x = deepcopy(x_init)
                temp_left = deepcopy(x_init)
                r, g, b = rgb_image.getpixel((x, y))
                # if we don't enter the loop, we have found the left edge!
                while r > 250 and g > 250 and b > 250:
                    temp_left = deepcopy(x)
                    x -= 1
                    r, g, b = rgb_image.getpixel((x, y))
                left_edge = (temp_left, y)
                # print("left edge found at: " + str(left_edge))

                # from the point to the left, go straight up until an edge <200
                # figure out am I the lower left or the upper left
                r, g, b = rgb_image.getpixel((x, y + 1))
                x = deepcopy(temp_left)
                temp_up = deepcopy(y)
                temp_down = deepcopy(y)
                # if there is nothing up, I am in upper part of the circle
                # print("pixel up is: ")
                # print (r, g, b)
                if r < 230 or g < 230 or b < 230:
                    # print("nothing up from left point, looking down")
                    r, g, b = rgb_image.getpixel((x, y - 1))
                    # print("pixel down is: ")
                    # print(r, g, b)
                    while r > 229 and g > 229 and b > 229:
                        temp_down = y - 1
                        y -= 1
                        r, g, b = rgb_image.getpixel((x, y))
                    pt_1 = (x, temp_down)
                    # print("third point found on the lower left quarter of circle at: " + str(pt_1))

                # otherwise, there is nothing down (I am in bottom part of the circle)
                else:
                    # print("nothing down from left point, looking up!")
                    while r > 230 and g > 230 and b > 230:
                        temp_up = deepcopy(y)
                        y += 1
                        r, g, b = rgb_image.getpixel((x, y))
                    pt_1 = (x, temp_up)
                    # print("third point found on the upper left quarter of circle at: " + str(pt_1))

                # diameter = distance from top point (or bottom point) to right-most point
                diameter = math.sqrt((pt_2[0]-pt_1[0])**2+(pt_2[1]-pt_1[1])**2)
                return diameter

        else:
            # need to evaluate if there is a big rectangle to provide some rewards
            return 0

    def find_diameter_red(self, image=None):
        """Finds the diameter of a red dot, if it is somewhat centered (center pixel is red). Otherwise, returns 0"""
        if image is None:
            rgb_image = self.rgb_im
        else:
            rgb_image = image.convert('RGB')
        r_threshold = 180
        g_b_threshold = 50

        x_init = rgb_image.size[0]/2 - 1
        print("initial x: " + str(x_init))
        y_init = rgb_image.size[1]/2 - 1
        print("initial y: " + str(y_init))
        r, g, b = rgb_image.getpixel((x_init, y_init))
        print(r, g, b)
        diameter = 0
        if r > r_threshold and g < 50 and b < 50:
            print("center pixel is red, finding diameter")
            x = deepcopy(x_init)
            y = deepcopy(y_init)
            r_right, g_right, b_right = rgb_image.getpixel((x + 1, y))
            r_left, g_left, b_left = rgb_image.getpixel((x - 1, y))
            r_up, g_up, b_up = rgb_image.getpixel((x, y + 1))
            r_down, g_down, b_down = rgb_image.getpixel((x, y - 1))

            # first handle the special cases

            # if nothing right or left is red
            if (r_right < r_threshold and g_right > 50 and b_right > 50) and \
                    (r_left < r_threshold and g_left > 50 and b_left > 50):
                print("nothing red right or left")
                pt_1 = (x, y)
                # if nothing up, search down
                if r_up < r_threshold and g_up > 50 and b_up > 50: # no red up, go down
                    print("nothing red up, looking down")
                    temp_down = deepcopy(y)
                    while r > r_threshold and g < 50 and b < 50:
                        temp_down = deepcopy(y)
                        y -= 1
                        r, g, b = rgb_image.getpixel((x, y))
                    pt_2 = (x, temp_down)
                # if nothing down, search up
                elif r_down < r_threshold and g_down > 50 and b_down > 50:
                    print("nothing red down, looking up")
                    while r > r_threshold and g < 50 and b < 50:
                        temp_up = deepcopy(y)
                        y += 1
                        r, g, b = rgb_image.getpixel((x, y))
                    pt_2 = (x, temp_up)
                # if nothing to the right, left, up or down, we are not in a dot.
                else:
                    diameter = 0
                    return diameter
                print("point 1 is " + str(pt_1) + ", point 2 is " + str(pt_2))
                diameter = math.sqrt((pt_2[0]-pt_1[0])**2+(pt_2[1]-pt_1[1])**2)
                return diameter

            # Then if we are somewhere inside the dot
            else:
                # step all the way right, find the point < 250 , get its coordinates (nothing right = edge found)
                temp_right = deepcopy(x)
                if r_right < r_threshold and g_right > 50 and b_right > 50:
                    # nothing right, edge found!
                    pt_2 = (x, y)
                else:
                    while r > r_threshold and g < 50 and b < 50:
                        temp_right = deepcopy(x)
                        x += 1
                        r, g, b = rgb_image.getpixel((x, y))
                    pt_2 = (temp_right, y) # last point where all pixels were above 200, right?
                print("red right edge found at:" + str(pt_2))

                # from initial pixel, step all the way left, find last white point, get coordinates
                x = deepcopy(x_init)
                temp_left = deepcopy(x_init)
                r, g, b = rgb_image.getpixel((x, y))
                # if we don't enter the loop, we have found the left edge!
                while r > r_threshold and g < 50 and b < 50:
                    temp_left = deepcopy(x)
                    x -= 1
                    r, g, b = rgb_image.getpixel((x, y))
                left_edge = (temp_left, y)
                print("red left edge found at: " + str(left_edge))

                # from the point to the left, go straight up until an edge <200
                # figure out am I the lower left or the upper left
                r, g, b = rgb_image.getpixel((x, y + 1))
                x = deepcopy(temp_left)
                temp_up = deepcopy(y)
                temp_down = deepcopy(y)
                # if there is nothing up, I am in upper part of the circle
                # print("pixel up is: ")
                # print (r, g, b)
                if r < r_threshold and g > 50 and b > 50:
                    print("nothing red up from left point, looking down")
                    r, g, b = rgb_image.getpixel((x, y - 1))
                    # print("pixel down is: ")
                    # print(r, g, b)
                    while r > r_threshold and g < 50 and b < 50:
                        temp_down = y - 1
                        y -= 1
                        r, g, b = rgb_image.getpixel((x, y))
                    pt_1 = (x, temp_down)
                    print("third point found on the lower left quarter of red circle at: " + str(pt_1))

                # otherwise, there is nothing down (I am in bottom part of the circle)
                else:
                    # print("nothing down from left point, looking up!")
                    while r > r_threshold and g < 50 and b < 50:
                        temp_up = deepcopy(y)
                        y += 1
                        r, g, b = rgb_image.getpixel((x, y))
                    pt_1 = (x, temp_up)
                    print("third point found on the upper left quarter of red circle at: " + str(pt_1))

                # diameter = distance from top point (or bottom point) to right-most point
                diameter = math.sqrt((pt_2[0]-pt_1[0])**2+(pt_2[1]-pt_1[1])**2)
                return diameter

        else:
            # need to evaluate if there is a big rectangle to provide some rewards
            return 0

    def detect_circle(self, image, draw_circle=False):
        """detects a circle and returns its radius and the distance from its center point to the center of the frame"""
        # src = cv.imread(image)
        # gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
        # output = src.copy()
        gray = image

        center_x = int((gray.shape[1]) / 2 - 1)
        center_y = int((gray.shape[0]) / 2 - 1)
        image_center = np.array((center_x, center_y))
        # print(center)

        # detect circles in the image
        circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1.2, 100)
        # ensure at least some circles were found
        if circles is not None:
            # convert the (x, y) coordinates and radius of the circles to integers
            circles = np.round(circles[0, :]).astype("int")
            # print(circles)
            radius = 0
            dist_from_center = None
            circle_center = None
            for (x, y, r) in circles:
                # print("entered loop, pixels evaluated are: ")
                # print(x, y)
                if gray[y, x].mean() < 150:
                    # print("mean was under 150, center pixels are: ")
                    # print(x, y)
                    radius = deepcopy(r)
                    circle_center = np.array((x, y))
                    dist_from_center = np.linalg.norm(image_center-circle_center)
            # # print(radius)
            # # loop over the (x, y) coordinates and radius of the circles
            # if draw_circle:
            #     for (x, y, r) in circles:
            #         # draw the circle in the output image, then draw a rectangle
            #         # corresponding to the center of the circle
            #         cv.circle(output, (x, y), r, (0, 255, 0), 4)
            #         cv.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
            #     # show the output image
            #     cv.namedWindow("output", cv.WINDOW_NORMAL)  # Create window with freedom of dimensions
            #     src_2 = cv.resize(src, (960, 540))  # Resize
            #     output_2 = cv.resize(output, (960, 540)) # Resize
            #     cv.imshow("output", np.hstack([src_2, output_2]))
            #     cv.waitKey(0)

        else:
            radius = 0
            dist_from_center = None
            centers = None
            # reward = 0 # or small negative penalty for time step?
        return radius, dist_from_center, circle_center

    def find_mean_center(self, image=None):
        if image is None:
            image = self.im.convert('L')
        else:
            image = image.convert('L')
        x = np.array(image)
        # 450 here is the max value determined for the white dot filling the screen
        max_size = 450
        x_init = int((image.size[0] - max_size) / 2 - 1)
        x_end = x_init + max_size
        y_init = int((image.size[1] - max_size) / 2 - 1)
        y_end = y_init + max_size
        center = x[y_init:y_end, x_init:x_end]

        # center = x[375:825, 575:1025]
        grayscale_mean_value = center.mean()
        return grayscale_mean_value

    def is_screen_visible(self, image=None):
        if image is None:
            gray_image = self.im.convert('L')
        else:
            gray_image = image.convert('L')
        x = np.array(image)
        grayscale_mean_value = x.mean()
        # # If the value is below a certain threshold, pixels are mostly black (lower value = darker)
        # if grayscale_mean_value < self.screen_visible_color_threshold:
        #     return True
        # else:
        #     return False
        return grayscale_mean_value





def test_circle_detection(path):
    print(path)
    count=1
    for filename in os.listdir(path):
        f = os.path.join(path, filename)
        # checking if it is a file
        if os.path.isfile(f):
            # print(f)
            gray = cv.imread(f, 0)
            output = gray.copy()
            #see https://pyimagesearch.com/2014/07/21/detecting-circles-images-using-opencv-hough-circles/
            circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1.58, 30, param1=100, param2=57, minRadius=0, maxRadius=35)
            # ensure at least some circles were found
            if circles is not None:
                # convert the (x, y) coordinates and radius of the circles to integers
                circles = np.round(circles[0, :]).astype("int")
                # print("found")
                # loop over the (x, y) coordinates and radius of the circles
                for (x, y, r) in circles:
                    # draw the circle in the output image, then draw a rectangle
                    # corresponding to the center of the circle
                    cv.circle(output, (x, y), r, (100, 0, 0), 4)
                    # cv.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
            # show the output image
            cv.imwrite("/home/laurence/git/visual_bender/src/saved_images/default/"+"out"+str(count)+".jpeg", np.hstack([gray, output]))
            # cv.imshow("output", np.hstack([gray, output]))
            # cv.waitKey(100)
        count += 1

if __name__ == '__main__':
    test_circle_detection('/home/laurence/git/visual_bender/src/saved_images/default/')
    # i = ImageProcessing(image='/home/lamarchecl/Downloads/saved_images_black_on_white/'
    #                           'collect_visual_datavisualBenderFourJointsGoal_ep0_step41.jpeg')
    # # i.analyze_images('/home/lamarchecl/Downloads/wetransfer_first-run-images_2022-08-18_1831/saved_images_2/'
    # #                         'saved_images/')
    # # i.analyze_images('/home/lamarchecl/Downloads/saved_images_black_on_white/')
    # # i = ImageProcessing(image='/home/lamarchecl/Downloads/red_dot_testing.jpg')
    # im = np.array(i.im)
    # # im_path = '/home/lamarchecl/Downloads/saved_images_black_on_white/' \
    # #           'collect_visual_datavisualBenderFourJointsGoal_ep0_step89.jpeg'
    # radius, dist_from_center = i.detect_circle(im, draw_circle=False)
    # print("radius is: " + str(radius) + ", distance from center is: " + str(dist_from_center))
    # red_im = i.im
    # diameter = i.find_diameter_red(red_im)
    #print("the diameter of the red centered dot is: " + str(diameter))
    # mean = i.find_mean_center(red_im)
    # print("the mean value of the center portion of the image is: " + str(mean))
    # reward_uofa = i.compute_reward_uofa(red_im)
    # print("the proposed reward is: " + str(reward_uofa))
    # diameter = i.find_diameter(im)
    # print("the diameter of the centered dot is: " + str(diameter))
    # screen_visibility = i.is_screen_visible(im)
    # print("the colour value for the whole image is: " + str(screen_visibility))
