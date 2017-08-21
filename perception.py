import numpy as np
import cv2
import pickle
import glob
#import matplotlib.pyplot as plt
import skvideo.io as skv



# from tracker import tracker

# helper functions
from moviepy.editor import VideoFileClip

from lane_line_tracker import Tracker
from road_processor import RoadImageStreamProcessor

M, Minv, img_size = None, None, None

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
#
# def contin_tracking(binary_output, left_fit, right_fit):
#     # Assume you now have a new warped binary image
#     # from the next frame of video (also called "binary_warped")
#     # It's now much easier to find line pixels!
#     nonzero = binary_warped.nonzero()
#     nonzeroy = np.array(nonzero[0])
#     nonzerox = np.array(nonzero[1])
#     margin = 100
#     left_lane_inds = ((nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] - margin)) & (
#     nonzerox < (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] + margin)))
#     right_lane_inds = (
#     (nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] - margin)) & (
#     nonzerox < (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] + margin)))
#
#     # Again, extract left and right line pixel positions
#     leftx = nonzerox[left_lane_inds]
#     lefty = nonzeroy[left_lane_inds]
#     rightx = nonzerox[right_lane_inds]
#     righty = nonzeroy[right_lane_inds]
#     # Fit a second order polynomial to each
#     left_fit = np.polyfit(lefty, leftx, 2)
#     right_fit = np.polyfit(righty, rightx, 2)
#     # Generate x and y values for plotting
#     ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
#     left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
#     right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
#
#     # Create an image to draw on and an image to show the selection window
#     out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
#     window_img = np.zeros_like(out_img)
#     # Color in left and right line pixels
#     out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
#     out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
#
#     # Generate a polygon to illustrate the search window area
#     # And recast the x and y points into usable format for cv2.fillPoly()
#     left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
#     left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin, ploty])))])
#     left_line_pts = np.hstack((left_line_window1, left_line_window2))
#     right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
#     right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin, ploty])))])
#     right_line_pts = np.hstack((right_line_window1, right_line_window2))
#
#     # Draw the lane onto the warped blank image
#     cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
#     cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
#     result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
#     plt.imshow(result)
#     plt.plot(left_fitx, ploty, color='yellow')
#     plt.plot(right_fitx, ploty, color='yellow')
#     plt.xlim(0, 1280)
#     plt.ylim(720, 0)


def draw_lanes(Minv, binary_warped, img, left_lane, right_lane, dist_from_center):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_lane.recent_xfitted[-1], left_lane.recent_yfitted]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_lane.recent_xfitted[-1], right_lane.recent_yfitted])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # draw laen onto binary warped
    bin_warped_rgb = cv2.cvtColor(binary_warped,cv2.COLOR_GRAY2RGB)
    cv2.fillPoly(bin_warped_rgb, np.int_([pts]), (0, 255, 255))
    #write binary warped
    write_file = "./test_images/binary_warped" + '.jpg'
    cv2.imwrite(write_file, bin_warped_rgb)

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(img, 1, newwarp, 0.3, 0)
    #plt.axis("off")

    cv2.putText(result, ('Curve radius: ' + str(round(left_lane.radius_of_curvature, 1)) + 'm'), (50, 100), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 2)
    cv2.putText(result, ('Distance from lane center: ' + str(round(dist_from_center, 3)) + 'm'), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    return result


def process_image(img, M, Minv, img_size):
    pixel_mask = np.zeros_like(img[:, :, 0])
    gradx = abs_sobel_threshold(img, 'x', thresh=(12, 255))
    grady = abs_sobel_threshold(img, 'y', thresh=(25, 255))
    colour_mask = colour_threshold(img, (100, 255), (50, 255))
    pixel_mask[((gradx == 1) & (grady == 1) | (colour_mask == 1))] = 255

    binary_warped = cv2.warpPerspective(pixel_mask, M, img_size, flags=cv2.INTER_LINEAR)
    result = binary_warped

    tracker = Tracker(50, 80, 100, 10 / 720, 4 / 384, 15)
    left_lane, right_lane = tracker.process_independent_image(result)
    rewarped = draw_lanes(Minv, binary_warped, img, left_lane, right_lane, tracker.dist_from_center)

    return rewarped



def main():
    dist_pickle = pickle.load(open("./camera_cal/calibration.p", "rb"))
    mtx = dist_pickle["mtx"]
    dist = dist_pickle["dist"]

    M, Minv, img_size = None, None, None

    # list test images
    test_images = glob.glob("./test_images/test*.jpg")
    straight = glob.glob("./test_images/straight*.jpg")

    tracker = Tracker(50, 80, 100, 10 / 720, 4 / 384, 15)

    for index, filename in enumerate(test_images):
        # read image
        img = cv2.imread(filename)
        img = cv2.undistort(img, mtx, dist, None, mtx)

        # write undsitorted file
        write_file = "./test_images/undistorted" + str(index) + '.jpg'
        cv2.imwrite(write_file, img)

        pixel_mask = np.zeros_like(img[:, :, 0])
        gradx = abs_sobel_threshold(img, 'x', thresh=(12, 255))
        grady = abs_sobel_threshold(img, 'y', thresh=(25, 255))
        colour_mask = colour_threshold(img, (100, 255), (50, 255))
        pixel_mask[((gradx == 1) & (grady == 1) | (colour_mask == 1))] = 255
        pixel_mask[( (colour_mask == 1))] = 255

        # write color thresholded mask
        write_file = "./test_images/lane_pixel_mask" + str(index) + '.jpg'
        cv2.imwrite(write_file, pixel_mask)

        M, Minv, img_size = get_trans_matrices(img)
        binary_warped = cv2.warpPerspective(pixel_mask, M, img_size, flags=cv2.INTER_LINEAR)
        result = binary_warped

        write_file = "./test_images/bird_" + str(index) + '.jpg'
        cv2.imwrite(write_file, result)

        left_lane, right_lane = tracker.process_independent_image(result)
        rewarped = draw_lanes(Minv, binary_warped, img, left_lane, right_lane, tracker.dist_from_center)
        write_file = "./test_images/rewarped" + str(index) + '.jpg'
        cv2.imwrite(write_file, rewarped)

    import sys
    sys.exit()
    M, Minv, img_size = get_trans_matrices(img)
    input_video = 'project_video.mp4'
    output_video = 'F:\SDCarND\CarND-Advanced-Lane-Lines\project_video_tracked_q.avi'
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    vid_writer = cv2.VideoWriter(output_video, fourcc, 30, (1280, 720))

    video_proc = RoadImageStreamProcessor(M, Minv, img_size, 'sem_seg_unet30e.h5', ( 608,320))

    #cap = cv2.VideoCapture('/home/florian/Coding/Projects/CarND-Advanced-Lane-Lines/project_video.mp4')
    vreader = skv.io.vreader(input_video)
    counter = 0
    start = 25 * 23 * 0
    end = 25* 30 * 9999
    for frame in  vreader:
        counter +=1
        if counter < start:
            continue
        if counter > end:
            break
        vid_writer.write(video_proc.process_image(frame))

    # while True:
    #     counter+=1
    #     ret, frame = cap.read()
    #     vid_writer.write(video_proc.process_image(frame))
    #     if counter >= 30*5:
    #         break
    # cap.release()


def get_trans_matrices(img):
    # perspective transform
    img_size = (img.shape[1], img.shape[0])
    bottom_width = 0.76
    mid_width = 0.08
    height_pct = 0.62
    bottom_trim = 0.935
    src = np.float32(
        [[img.shape[1] * (.5 - mid_width / 2), img.shape[0] * height_pct],
         [img.shape[1] * (.5 + mid_width / 2), img.shape[0] * height_pct],
         [img.shape[1] * (.5 + bottom_width / 2), img.shape[0] * bottom_trim],
         [img.shape[1] * (.5 - bottom_width / 2), img.shape[0] * bottom_trim]
         ])
    offset = img_size[0] * 0.25
    dst = np.float32([
        [offset, 0],
        [img_size[0] - offset, 0],
        [img_size[0] - offset, img_size[1]],
        [offset, img_size[1]]
    ]
    )
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    return M, Minv, img_size


if __name__ == '__main__':
    main()
