import numpy as np
import cv2 as cv

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD


def createModel():
    # example of a 3-block vgg style architecture
    model = Sequential()
    model.add(
        Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    # model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    # model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    # model.add(MaxPooling2D((2, 2)))

    # example output part of the model
    model.add(Flatten())
    model.add(Dense(64, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(2, activation='softmax'))

    # compile model
    opt = SGD(lr=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# use hsv thresholding and connected components to find ball candidates


def thresholdHSVBallClusters(frame_bgr, h_min = 20, h_max = 45, s_min = 25, s_max = 180, v_min = 150, a_min = 10,  a_max = 1000):
    frame_rgb = cv.cvtColor(frame_bgr, cv.COLOR_BGR2RGB)
    frame_hsv = cv.cvtColor(frame_bgr, cv.COLOR_BGR2HSV)
    # frame_gray = cv.cvtColor(frame_bgr, cv.COLOR_BGR2GRAY)
    frame_thresh = np.zeros((frame_rgb.shape[0], frame_rgb.shape[1]))
    # frame_zoomedin = frame_rgb[475:525, 475:525, :]

    # print(frame_hsv[475+18:475+22,500:505,:])
    # h: 25-45, s:45-65, v: 250-255
    # https://stackoverflow.com/questions/51229126/how-to-find-the-red-color-regions-using-opencv

    # filter based on hsv
    idx_hue_bright = np.logical_and((frame_hsv[:, :, 0] >= h_min), (frame_hsv[:, :, 0] <= h_max))
    idx_sat_bright = np.logical_and((frame_hsv[:, :, 1] >= s_min), (frame_hsv[:, :, 1] <= s_max))
    idx_val_bright = (frame_hsv[:, :, 2] >= v_min)
    idx_hsv_bright = np.logical_and(np.logical_and(idx_hue_bright, idx_sat_bright), idx_val_bright)
    frame_thresh[idx_hsv_bright] = 255

    kernel = np.ones((2, 2), np.uint8)
    frame_open = cv.morphologyEx(frame_thresh, cv.MORPH_OPEN, kernel)
    num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(frame_open.astype(np.uint8), 8, cv.CV_32S)

    idx_good = np.logical_and((stats[:, cv.CC_STAT_AREA] > a_min), stats[:, cv.CC_STAT_AREA] < a_max)
    centroids_good = centroids[idx_good]
    stats_good = stats[idx_good]
    labels_good = np.arange(num_labels)[idx_good]

    return frame_open, labels_good, labels, stats_good, centroids_good



def thresholdFlowBallClusters(prev_frame, cur_frame, flow_threshold=127, a_min=30, a_max=10000):
    prev_frame = cv.cvtColor(((prev_frame)), cv.COLOR_BGR2GRAY)
    cur_frame = cv.cvtColor((((cur_frame))), cv.COLOR_BGR2GRAY)

    flow = cv.calcOpticalFlowFarneback(prev_frame, cur_frame, None, pyr_scale=0.5, levels=3, winsize=20, iterations=3, poly_n=7, poly_sigma=1.5, flags=0)
    mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
    flow_hsv = np.zeros((cur_frame.shape[0], cur_frame.shape[1],3))
    flow_hsv[..., 0] = ang * 180 / np.pi / 2
    flow_hsv[..., 1] = 255
    flow_hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)

    bgr = cv.cvtColor(flow_hsv.astype(np.uint8), cv.COLOR_HSV2BGR)
    gray = flow_hsv[...,2]
    ret, thresh = cv.threshold(gray, flow_threshold, 255, cv.THRESH_BINARY)

    num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(thresh.astype(np.uint8), 8, cv.CV_32S)
    idx_big_comps = np.logical_and((stats[:, cv.CC_STAT_AREA] > a_min), stats[:, cv.CC_STAT_AREA] < a_max)
    labels_good = np.arange(num_labels)[idx_big_comps]
    stats_good = stats[idx_big_comps]
    centroids_good = centroids[idx_big_comps]

    return bgr, gray, thresh, flow, (labels_good, labels, stats_good, centroids_good)