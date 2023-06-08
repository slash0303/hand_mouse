from eaxtension import LogE
import math
import numpy as np

# mediapipe 모듈에서 얻은 0~1 사이의 좌표값을 x: -50 ~ 50, y: -50 ~ 50으로 보정함.
def location_process(data):
    data = data * 100
    data = int(data) - 50
    data = -1 * data

    return data

# z축 연산 함수
def z_scale_transfer(z_scale):

    z_scale = z_scale - 20
    if z_scale < 0:
        z_scale = z_scale**2
        z_scale = -1 * z_scale
    elif z_scale >=0:
        z_scale = z_scale**2

    if z_scale >= 50:
        z_scale = 50
    elif z_scale < -50:
        z_scale = -50

    return z_scale

# 주먹 인식 함수
def fist_detect(ax, ay, bx, by, cx, cy, dx, dy, ex, ey, fx, fy, px, py):
    #a 5, b 9, c 17, d 8, e 16, f 20, g 0

    dist_ind_in = cal_dist(ax, ay, px, py)
    dist_ind_out = cal_dist(dx, dy, px, py)

    dist_mid_in = cal_dist(bx, by, px, py)
    dist_mid_out = cal_dist(cx, cy, px, py)

    dist_pin_in = cal_dist(ex, ey, px, py)
    dist_pin_out = cal_dist(fx, fy, px, py)

    ra = dist_comp(dist_ind_in, dist_ind_out)
    rb = dist_comp(dist_mid_in, dist_mid_out)
    rc = dist_comp(dist_pin_in, dist_pin_out)

    if ra and rb and rc:
        return 60
    else:
        return 120

# 주먹 인식에서 손가락이 말려들어갔는지를 인식함.
def dist_comp(dist_in, dist_out):
    if dist_in >= dist_out:
        return True
    elif dist_in < dist_out:
        return False
    else:
        pass

# 점 a, b 간의 거리를 계산함.
def cal_dist(ax, ay, bx, by):

    x = ax - bx
    x = x**2

    y = ay - by
    y = y**2

    z = x + y

    distance = int(math.sqrt(z))

    return distance