import cv2
import numpy as np
import mediapipe as mp
import hand_process as hp
from eaxtension import LogE
from eaxtension import jsonE
import keyboard

# 캠 내부 텍스트 삽입 함수 (wrapping)
# 작동 안 되니까 수리할 것(굳이 필요 없긴 함)
def text_in(frame, text="default", location=(0,0), color=(0, 0, 0)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, text, location, font, 1, color, 2)
    # https://blog.naver.com/chandong83/220932297731

# 검지손가락 좌표 세팅
def point_set(frame, key:str, dict_index):
    if cv2.waitKey(2) == ord(key):
        LogE.d(key, "pressed")
        text_in(frame, "perspective settings")
        point_of_screen[dict_index] = middle_finger_tip
        LogE.d("pos", point_of_screen)

# 세팅된 좌표 저장
def save_point(points):
    if cv2.waitKey(2) == ord('s'):
        save_data = {0:{}, 1:{}, 2:{}, 3:{}}
        for i in range(0, 4):
            save_data[i]["x"] = points[i].x
            save_data[i]["y"] = points[i].y
        jsonE.dumps("points.json", save_data)


# 변환 행렬 계산
def persp_calc(frame, points:dict):
    type(frame)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    # 세팅된 좌표로 행렬 생성
    try:
        origin_corner = np.float32([[width*points[0].x, height*points[0].y],
                                    [width*points[1].x, height*points[1].y],
                                    [width*points[2].x, height*points[2].y],
                                    [width*points[3].x, height*points[3].y]])
    except:
        origin_corner = np.float32([[width*points[0]["x"], height*points[0]["y"]],
                                    [width*points[1]["x"], height*points[1]["y"]],
                                    [width*points[2]["x"], height*points[2]["y"]],
                                    [width*points[3]["x"], height*points[3]["y"]]])
    finally:
        # 생성된 창의 꼭짓점으로 행렬 생성
        frame_corner = np.float32([[0, 0],
                                   [width, 0],
                                   [width, height],
                                   [0, height]])
        modified_frame = cv2.getPerspectiveTransform(origin_corner, frame_corner)
        return cv2.warpPerspective(frame, modified_frame, (int(width), int(height)))

# json 불러오기
def load_point():
    loaded_data = jsonE.load("points.json")
    points = {0: {}, 1: {}, 2: {}, 3: {}}
    for i in range(0, 4):
        points[i]["x"] = loaded_data[str(i)]["x"]
        points[i]["y"] = loaded_data[str(i)]["y"]
    return points

# perspective 변환 이용을 위한 손가락 위치
middle_finger_tip = None
point_of_screen = {}

# hand landmark 모듈 init
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# VideoCapture 객체 생성 - 0번 카메라
cap = cv2.VideoCapture(0)

# Mediapipe Hands 모델 로드
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    max_num_hands = 2) as hands:

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
                middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

        # Flip the image horizontally for a selfie-view display.
        cv2.imshow('Hand mouse', cv2.flip(image, 1))
        if cv2.waitKey(5) & 0xFF == 27:
            break

        # 좌표계 설정 함수
        point_set(image, "q", 0)
        point_set(image, "p", 1)
        point_set(image, "m", 2)
        point_set(image, "z", 3)
        save_point(point_of_screen)

        if cv2.waitKey(4) == ord('r'):
            LogE.d("r", "pressed")
            image = persp_calc(image, point_of_screen)
            cv2.imshow('perspectived', image)

        if cv2.waitKey(4) == ord('l'):
            point_of_screen = load_point()

cap.release()