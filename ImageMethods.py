# METHODS FOR MODIFYING IMAGES

import cv2 as cv
import numpy as np
import random

def resize(frame, scale):
    return cv.resize(frame, (int(scale*frame.shape[1]), int(scale*frame.shape[0])))

def distortedCapture(full_img, x, y, img_size):
    #     rotate full image (ra), capture (d, s), flip (flip)
    ra = random.randint(0, 359)
    dx = (random.randint(0, 10) - 5)
    dy = (random.randint(0, 10) - 5)
    scale = (1.5) - 1
    sx = int(random.randint(0, img_size * scale) - img_size * scale / 2)
    sy = int(random.randint(0, img_size * scale) - img_size * scale / 2)
    flip = random.randint(0, 3) - 1

    M = cv.getRotationMatrix2D((x, y), ra, 1)
    full_img = cv.warpAffine(full_img, M, (full_img.shape[1], full_img.shape[0]))

    down_b = y - 16 + dy - sy
    up_b = y + 16 + dy + sy
    left_b = x - 16 + dx - sx
    right_b = x + 16 + dx + sx
    #     print(down_b,":",up_b ,",", left_b,":",right_b)
    if (down_b < 0 or up_b > full_img.shape[0] or left_b < 0 or right_b > full_img.shape[1]):
        return distortedCapture(full_img, x, y, img_size)

    ball_img = full_img[y - 16 + dy - sy: y + 16 + dy + sy, x - 16 + dx - sx: x + 16 + dx + sx]
    ball_img = cv.resize(ball_img, (img_size, img_size))

    if (flip != 2):
        ball_img = cv.flip(ball_img, flipCode=flip)

    return ball_img

