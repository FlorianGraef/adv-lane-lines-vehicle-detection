import numpy as np
import cv2
#import matplotlib.pyplot as plt
from lane_line import LaneLine
#from cv2 import
THRESH_A = 0.0005


class Tracker():
    def __init__(self, Window_width, Window_height, Window_margin, Y_in_meters=1, X_in_meters=1, Smooth_factor=15):
        self.recent_centers = []

        self.window_width = Window_width

        self.window_height = Window_height
        self.window_no = None

        self.margin = Window_margin

        self.ym_per_pix = Y_in_meters

        self.xm_per_pix = X_in_meters

        self.dist_from_center = None

        self.left_lane = LaneLine()
        self.right_lane = LaneLine()

    def sanity_check(self, thresh_a):
        # check distance of lanes
        diff_x = self.left_lane.recent_xfitted[-1] - self.right_lane.recent_xfitted[-1]
        #print("Diff X: ",diff_x)
        diff_b = self.left_lane.last_fits[-1][1] - self.right_lane.last_fits[-1][1]
        #print("Diff b:", diff_b)
        # check lanes parellel through "a" factor comparison


        left_a = self.left_lane.last_fits[-1][0]
        right_a = self.right_lane.last_fits[-1][0]
        diff_a = left_a - right_a
        #print ("Difference left to right:", (left_a-right_a))
        if abs(diff_a) > 0.001:
            return False
        if abs(diff_x[0]) > 700:
            return False
        return True



    def process_independent_image(self, binary_birdseye_mask, left_lane = LaneLine(), right_lane = LaneLine()):
        print("processing individual image")

        self.left_lane = left_lane
        self.right_lane = right_lane
        #plt.imshow(binary_birdseye_mask, cmap='Greys')
        histogram = np.sum(binary_birdseye_mask[binary_birdseye_mask.shape[0] // 2:, :], axis=0)
        #plt.plot(histogram)
        # Create an output image to draw on and  visualize the result
        out_img = np.dstack((binary_birdseye_mask, binary_birdseye_mask, binary_birdseye_mask)) * 255
        # peaks left and right of the center
        centerpoint = np.int(histogram.shape[0] / 2)
        leftx_base = np.argmax(histogram[:centerpoint])
        rightx_base = np.argmax(histogram[centerpoint:]) + centerpoint

        # Choose the number of sliding windows
        self.window_no = np.int(binary_birdseye_mask.shape[0] / self.window_height)

        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_birdseye_mask.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        # Set the width of the windows +/- margin

        # Set minimum number of pixels found to recenter window
        minpix = 50
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(self.window_no):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_birdseye_mask.shape[0] - (window + 1) * self.window_height
            win_y_high = binary_birdseye_mask.shape[0] - window * self.window_height
            win_xleft_low = leftx_current - self.margin
            win_xleft_high = leftx_current + self.margin
            win_xright_low = rightx_current - self.margin
            win_xright_high = rightx_current + self.margin
            # Draw the windows on the visualization image
            cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
            cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
                nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
                nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

            #plt.imshow(out_img, cmap="Greys")

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        left_lane.allx = nonzerox[left_lane_inds]
        left_lane.ally = nonzeroy[left_lane_inds]
        right_lane.allx = nonzerox[right_lane_inds]
        right_lane.ally = nonzeroy[right_lane_inds]

        # Fit a second order polynomial to each
        left_lane.last_fits.append(np.polyfit(left_lane.ally, left_lane.allx, 2))
        right_lane.last_fits.append(np.polyfit(right_lane.ally, right_lane.allx, 2))

        #left_lane.best_fit = left_lane.last_fits[-1]
        #right_lane.best_fit = right_lane.last_fits[-1]

        # Generate x and y values for plotting
        left_lane.recent_yfitted = np.linspace(0, binary_birdseye_mask.shape[0] - 1, binary_birdseye_mask.shape[0])
        right_lane.recent_yfitted = np.copy(left_lane.recent_yfitted)
        left_fitx = left_lane.last_fits[-1][0] * left_lane.recent_yfitted ** 2 + left_lane.last_fits[-1][
                                                                                   1] * left_lane.recent_yfitted + \
                    left_lane.last_fits[-1][2]
        left_lane.recent_xfitted.append(left_fitx)

        left_lane.calc_best_fit()
        right_lane.calc_best_fit()


        left_lane.calc_curv_rad(self.ym_per_pix, self.xm_per_pix)
        right_fitx = right_lane.last_fits[-1][0] * right_lane.recent_yfitted ** 2 + right_lane.last_fits[-1][
                                                                                      1] * right_lane.recent_yfitted + \
                     right_lane.last_fits[-1][2]
        right_lane.recent_xfitted.append(right_fitx)

        right_lane.calc_curv_rad(self.ym_per_pix, self.xm_per_pix)

        left_lane.detected, right_lane.detected = True, True

        lane_center = (left_lane.recent_xfitted[-1][-1] + right_lane.recent_xfitted[-1][-1]) / 2
        self.dist_from_center = (lane_center - binary_birdseye_mask.shape[1]/2) * self.xm_per_pix
        self.sanity_check(THRESH_A)

        # assume you can't fail the full scan
        left_lane.detected = True
        right_lane.detected = True
        return left_lane, right_lane

    def process_next_image(self, binary_birdseye_mask, left_lane, right_lane):

        print("next frame")

        self.left_lane = left_lane
        self.right_lane = right_lane

        # Assume you now have a new warped binary image
        # from the next frame of video (also called "binary_warped")
        # It's now much easier to find line pixels!
        out_img = np.dstack((binary_birdseye_mask, binary_birdseye_mask, binary_birdseye_mask)) * 255
        #cv2.imshow('top_view', out_img)

        # left_lane.clear()
        # right_lane.clear()

        nonzero = binary_birdseye_mask.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        margin = 100
        left_lane_inds = (
            (nonzerox > (
            left_lane.last_fits[-1][0] * (nonzeroy ** 2) + left_lane.last_fits[-1][1] * nonzeroy + left_lane.last_fits[-1][
                2] - margin)) & (
                nonzerox < (left_lane.last_fits[-1][0] * (nonzeroy ** 2) + left_lane.last_fits[-1][1] * nonzeroy +
                            left_lane.last_fits[-1][2] + margin)))
        right_lane_inds = (
            (nonzerox > (
            right_lane.last_fits[-1][0] * (nonzeroy ** 2) + right_lane.last_fits[-1][1] * nonzeroy + right_lane.last_fits[-1][
                2] - margin)) & (
                nonzerox < (right_lane.last_fits[-1][0] * (nonzeroy ** 2) + right_lane.last_fits[-1][1] * nonzeroy +
                            right_lane.last_fits[-1][2] + margin)))

        # Again, extract left and right line pixel positions
        left_lane.allx = nonzerox[left_lane_inds]
        left_lane.ally = nonzeroy[left_lane_inds]
        right_lane.allx = nonzerox[right_lane_inds]
        right_lane.ally = nonzeroy[right_lane_inds]

        # Fit a second order polynomial to each
        #left_last_fit = left_lane.last_fits[-1]
        #right_last_fit = right_lane.last_fits[-1]
        left_lane.last_fits.append(np.polyfit(left_lane.ally, left_lane.allx, 2))
        right_lane.last_fits.append(np.polyfit(right_lane.ally, right_lane.allx, 2))

        #left_lane.best_fit = np.average( np.array([left_last_fit, left_lane.last_fits[-1]]), axis=0)
        #right_lane.best_fit = np.average(np.array([right_last_fit, right_lane.last_fits[-1]]), axis=0)

        # Generate x and y values for plotting
        # ploty = np.linspace(0, binary_birdseye_mask.shape[0] - 1, binary_birdseye_mask.shape[0])

        left_lane.recent_yfitted = np.linspace(0, binary_birdseye_mask.shape[0] - 1, binary_birdseye_mask.shape[0])
        right_lane.recent_yfitted = np.copy(left_lane.recent_yfitted)

        left_fitx = left_lane.last_fits[-1][0] * left_lane.recent_yfitted ** 2 + left_lane.last_fits[-1][
                                                                                   1] * left_lane.recent_yfitted + \
                    left_lane.last_fits[-1][2]
        left_lane.recent_xfitted.append(left_fitx)
        left_lane.calc_curv_rad(self.ym_per_pix, self.xm_per_pix)
        right_fitx = right_lane.last_fits[-1][0] * right_lane.recent_yfitted ** 2 + right_lane.last_fits[-1][
                                                                                      1] * right_lane.recent_yfitted + \
                     right_lane.last_fits[-1][2]
        right_lane.recent_xfitted.append(right_fitx)
        right_lane.calc_curv_rad(self.ym_per_pix, self.xm_per_pix)
        out_img[left_lane.ally, left_lane.allx] = [255, 0, 0]
        out_img[right_lane.ally, right_lane.allx] = [0, 0, 255]
        #cv2.imshow('top view new', out_img)

        lane_center = int((left_lane.recent_xfitted[-1][-1] + right_lane.recent_xfitted[-1][-1]) / 2)
        self.dist_from_center = (lane_center - binary_birdseye_mask.shape[1] / 2) * self.xm_per_pix

        if not self.sanity_check(THRESH_A):
            left_lane.detected = False
            right_lane.detected = False
            left_lane.last_fits.pop()
            right_lane.last_fits.pop()
        left_lane.calc_best_fit()
        right_lane.calc_best_fit()

        return left_lane, right_lane
