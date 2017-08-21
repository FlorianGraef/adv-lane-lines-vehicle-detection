from collections import deque

import cv2
#import matplotlib.pyplot as plt
from lane_line import LaneLine
from lane_line_tracker import Tracker
import numpy as np
from keras.models import load_model

from vehicle_inference import VehicleInference


def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def histogram_normalization(image):
    b, g, r = cv2.split(image)
    b = cv2.equalizeHist(b)
    g = cv2.equalizeHist(g)
    r = cv2.equalizeHist(r)

    return cv2.merge([b,g,r])

def abs_sobel_threshold(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))

    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return binary_output

def draw_lanes(Minv, binary_warped, img, left_lane, right_lane, dist_from_center):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_lane.bestx, left_lane.recent_yfitted]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_lane.bestx, right_lane.recent_yfitted])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    lane_colour = (0, 255, 0)
    if not left_lane.detected:
        lane_colour = (255, 0, 0)
    cv2.fillPoly(color_warp, np.int_([pts]), lane_colour)

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(img, 1, newwarp, 0.3, 0)
    #plt.axis("off")

    cv2.putText(result, ('Curve radius: ' + str(round(left_lane.radius_of_curvature, 1)) + 'm'), (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.putText(result, ('Distance from lane center: ' + str(round(dist_from_center, 3)) + 'm'), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    diff_a = left_lane.last_fits[-1][0] - right_lane.last_fits[-1][0]

    return result
#

def colour_threshold(image, sthreshold=(0, 255), vthreshold=(0, 255)):
    # filter by saturation thresholds
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    s_channel = hls[:, :, 2]
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= sthreshold[0]) & (s_channel <= sthreshold[1])] = 1

    # filter by value channel
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    v_channel = hsv[:, :, 2]
    v_binary = np.zeros_like(v_channel)
    v_binary[(v_channel >= vthreshold[0]) & (v_channel <= vthreshold[1])] = 1

    output = np.zeros_like(s_channel)
    output[(s_binary == 1) & (v_binary == 1)] = 1
    return output


class RoadImageStreamProcessor(object):
    def transform(self, org_img):
        pixel_mask = np.zeros_like(org_img[:, :, 0])
        #org_img = gaussian_blur( org_img, 3)
        gradx = abs_sobel_threshold(org_img, 'x', thresh=(12, 255))
        grady = abs_sobel_threshold(org_img, 'y', thresh=(25, 255))
        colour_mask = colour_threshold(org_img, (100, 255), (50, 255))
        #colour_mask = alt_colour_thresh(org_img)
        pixel_mask[((gradx == 1) & (grady == 1) | (colour_mask == 1))] = 255
        #pixel_mask[((colour_mask == 255))] = 255
        binary_warped = cv2.warpPerspective(pixel_mask, self.M, self.img_size, flags=cv2.INTER_LINEAR)
        result = binary_warped
        return result

    def __init__(self, M, Minv, img_size, model_path, nn_imput_dims):
        self.tracker = Tracker(50, 80, 100, 10 / 720, 4 / 384, 15)

        self.detect_gap = 16;
        self.allowed_detect_gap = 10
        self.left_lane = LaneLine()
        self.right_lane = LaneLine()
        self.M = M
        self.Minv = Minv
        self.img_size = img_size
        self.last_masks = deque(maxlen=20)

        self.vehicle_tracker = VehicleInference(model_path, nn_imput_dims)

    def process_image(self, img):
        #img_for_nn = histogram_normalization(img)


        mask = self.vehicle_tracker.get_mask(img)
        self.last_masks.append(mask)
        heat_map = np.sum(self.last_masks, axis=0)
        #cv2.imshow("heatmap", heat_map)


        #cv2.imshow("mask", mask)
        mask = self.prepare_mask(heat_map, self.last_masks)
        #cv2.imshow("mask", mask)
        top_img = self.transform(img)

        if self.detect_gap >= self.allowed_detect_gap:
            self.tracker.process_independent_image(top_img, self.left_lane, self.right_lane)
            self.detect_gap = 0
        else:
            self.tracker.process_next_image(top_img, self.left_lane, self.right_lane)
            if not (self.left_lane.detected & self.right_lane.detected):
                #self.tracker.process_independent_image(top_img, self.left_lane, self.right_lane)
                self.detect_gap += 1
        # reverse transform and draw on original

        rewarped = draw_lanes(self.Minv, top_img, img, self.left_lane, self.right_lane, self.tracker.dist_from_center)
        rewarped = cv2.addWeighted(mask,0.5,rewarped,1,0, dtype=cv2.CV_8UC3)
        rewarped = cv2.cvtColor(rewarped, cv2.COLOR_BGR2RGB)
        #cv2.imshow("combined", rewarped)
        return rewarped

    def prepare_mask(self, mask, last_masks, detec_tresh=19):
        if len(last_masks) >=3:
            mask[(last_masks[-1] < 0.5) & (last_masks[-2] < 0.5) & (last_masks[-3] < 0.5)] = 0
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        mask = cv2.convertScaleAbs(mask)
        mask[:, :, 0:1] = 0
        mask[mask <= detec_tresh] = 0
        mask = mask * 255
        mask = cv2.resize(mask, self.img_size)
        return mask
