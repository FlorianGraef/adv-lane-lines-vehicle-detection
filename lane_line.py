import numpy as np
from collections import deque

class LaneLine():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False

        # x values of the last n fits of the line
        self.recent_xfitted = deque([])

        self.recent_yfitted = None

        # average x values of the fitted line over the last n iterations
        self.bestx = None

        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None

        # polynomial coefficients for the most recent fit
        self.last_fits = deque()

        # radius of curvature of the line in some units
        self.radius_of_curvature = None

        # distance in meters of vehicle center from the line
        self.line_base_pos = None

        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0, 0, 0], dtype='float')

        # x values for detected line pixels
        self.allx = None

        # y values for detected line pixels
        self.ally = None

        self.ym_perpix = None
        self.xm_perpix = None
        # curvature of the lane
        self.curv = None

    def clear(self):
        self.allx = None
        self.ally = None
        self.recent_yfitted = None

    def calc_curv_rad(self, ym_perpix, xm_perpix):
        curv_fit = np.polyfit(np.array(self.recent_yfitted)*ym_perpix, np.array(self.recent_xfitted[-1])*xm_perpix, 2)
        y_eval = np.min(self.recent_yfitted)
        curv_rad = ((1+ (2 * curv_fit[0]*y_eval*ym_perpix + curv_fit[1])**2)/np.absolute(2*curv_fit[0]))
        self.radius_of_curvature = curv_rad
        return curv_rad

    def calc_best_fit(self):
        # average last 5 fits


        last5poly = list(self.last_fits)[-5: ]
        self.best_fit = np.average(last5poly, axis=0)

        #self.best_fit = np.average( np.array([ i for i in  last5poly]), axis=0)
        self.bestx = self.best_fit[0] * self.recent_yfitted ** 2 + self.best_fit[1] * self.recent_yfitted + self.best_fit[2]