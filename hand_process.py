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

    z_const = 20
    z_scale = z_scale - z_const
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

# 점 a, b 간의 거리를 계산함.
def cal_dist(ax, ay, bx, by):

    x = ax - bx
    x = x**2

    y = ay - by
    y = y**2

    z = x + y

    distance = int(math.sqrt(z))

    return distance