import cv2
import numpy as np
from config import SegmentationConfig
from rail_marking.cfg import BiSeNetV2Config
from rail_marking.rail_marking.segmentation.deploy import RailtrackSegmentationHandler


class Segmentator:
    def __init__(self, pattern_img, pattern_threshold, config: SegmentationConfig):
        self.pattern_img = pattern_img
        self.p_h, self.p_w = self.pattern_img.shape[:2]
        self.config = config
        self.pattern_threshold = pattern_threshold
        self.lower_white = np.array([200, 200, 200], dtype=np.uint8)
        self.upper_white = np.array([255, 255, 255], dtype=np.uint8)
        self.segmentation_handler = RailtrackSegmentationHandler(self.config.model_weights, BiSeNetV2Config())

    def find_pattern(self, frame):
        img = frame.copy()
        res = cv2.matchTemplate(img, self.pattern_img, cv2.TM_CCOEFF_NORMED)
        loc = np.where(res >= self.config.threshold)

        pt = list(zip(*loc[::-1]))[0]
        return img[pt[1] : pt[1] + self.p_h, pt[0] : pt[0] + self.p_w, :]

    def flipper(self, frame):
        img = frame.copy()
        res = cv2.matchTemplate(img, self.pattern_img, cv2.TM_CCOEFF_NORMED)
        loc = np.where(res >= self.pattern_threshold)
        pt = list(zip(*loc[::-1]))
        if len(pt) < 1:
            img = cv2.flip(img, 1)

        return img

    def flat_mask(self, frame):
        img = frame.copy()
        mask = cv2.inRange(img, self.lower_white, self.upper_white)
        inter = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        cnts, _ = cv2.findContours(inter, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnt = max(cnts, key=cv2.contourArea)
        out = np.zeros(mask.shape, np.uint8)
        mask = cv2.drawContours(out, [cnt], -1, 255, cv2.FILLED)
        img = cv2.bitwise_and(img, img, mask=mask)
        _, overlay = self.segmentation_handler.run(img, only_mask=False)
        return overlay

    def automatic_brightness_and_contrast(self, image, clip_hist_percent=25):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Calculate grayscale histogram
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist_size = len(hist)

        # Calculate cumulative distribution from the histogram
        accumulator = []
        accumulator.append(float(hist[0]))
        for index in range(1, hist_size):
            accumulator.append(accumulator[index - 1] + float(hist[index]))

        # Locate points to clip
        maximum = accumulator[-1]
        clip_hist_percent *= maximum / 100.0
        clip_hist_percent /= 2.0

        # Locate left cut
        minimum_gray = 0
        while accumulator[minimum_gray] < clip_hist_percent:
            minimum_gray += 1

        # Locate right cut
        maximum_gray = hist_size - 1
        while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
            maximum_gray -= 1

        # Calculate alpha and beta values
        alpha = 255 / (maximum_gray - minimum_gray)
        beta = -minimum_gray * alpha

        auto_result = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        return (auto_result, alpha, beta)
