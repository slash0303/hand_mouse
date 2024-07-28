import cv2
import numpy as np
import mediapipe as mp
from eaxtension import LogE
from eaxtension import jsonE
import mouse
import pyautogui

pyautogui.FAILSAFE = False

# 캠 내부 텍스트 삽입 함수 (wrapping)
# 작동 안 되니까 수리할 것(굳이 필요 없긴 함)
def text_in(frame, text="default", location=(0,0), color=(0, 0, 0)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, text, location, font, 1, color, 2)
    # https://blog.naver.com/chandong83/220932297731

# 검지손가락 좌표 세팅
def point_set(frame, key:str, dict_index):
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
def persp_calc(frame, points:dict, middle_finger_tip):
    # frame의 높이, 너비
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    # LogE.t("width", width)
    # LogE.g("height", height)

    width_set = 2560.0
    height_set = 1440.0

    try:
        # 세팅된 좌표로 행렬 생성
        origin_corner = np.float32([[width*points[0].x, height*points[0].y],
                                    [width*points[1].x, height*points[1].y],
                                    [width*points[2].x, height*points[2].y],
                                    [width*points[3].x, height*points[3].y]])

    except:
        # 불러오기의 경우
        origin_corner = np.float32([[width*points[0]["x"], height*points[0]["y"]],
                                    [width*points[1]["x"], height*points[1]["y"]],
                                    [width*points[2]["x"], height*points[2]["y"]],
                                    [width*points[3]["x"], height*points[3]["y"]]])
    finally:
        # 생성된 창의 꼭짓점으로 행렬 생성
        frame_corner = np.float32([[0, 0],
                                   [width_set, 0],
                                   [width_set, height_set],
                                   [0, height_set]])


        # 점 위치 0,0으로 이동시키는 translate trasform 행렬
        transl_matrix = np.float32([[1, 0, 0],
                                    [0, 1, 0]])

        # perspective 변환 행렬 생성
        mod_matrix = cv2.getPerspectiveTransform(origin_corner, frame_corner)
        LogE.d("persp matrix", mod_matrix)

        frame = cv2.warpPerspective(frame, mod_matrix, (int(width_set), int(height_set)))
        # frame = cv2.warpAffine(frame, transl_matrix, (0, 0))

        return frame, mod_matrix


# 마우스 제어 함수
def mouse_control():
    try:
        # perspective 변환된 좌표계에서 손가락 위치 구하기
        middle_finger_x = middle_finger_tip.x
        middle_finger_y = middle_finger_tip.y
    except:
        # 카메라에 손이 잡히지 않는 경우
        middle_finger_x = 0
        middle_finger_y = 0

    finally:
        # 계산가능하도록 행렬로 변환
        persp_point = np.float32([[width*middle_finger_x],
                                  [height*middle_finger_y],
                                  [1]])

    persp_point = np.dot(mod_matrix, persp_point)
    # LogE.d("mid x", middle_finger_x)
    # LogE.d("mid y", middle_finger_y)
    # LogE.d("persp_point x not div", int(persp_point[0]))
    # LogE.d("persp_point y not div", int(persp_point[1]))
    # LogE.d("persp_point x", int(persp_point[0]/persp_point[2]))
    # LogE.d("persp_point y", int(persp_point[1]/persp_point[2]))

    # mouse.move(persp_point[0], persp_point[1])
    LogE.d("point", (int(persp_point[0] / persp_point[2]), int(persp_point[1] / persp_point[2])))
    pyautogui.moveTo(int(persp_point[0] / persp_point[2]), int(persp_point[1] / persp_point[2]))

# json 불러오기
def load_point():
    loaded_data = jsonE.load("points.json")
    points = {0: {}, 1: {}, 2: {}, 3: {}}
    for i in range(0, 4):
        points[i]["x"] = loaded_data[str(i)]["x"]
        points[i]["y"] = loaded_data[str(i)]["y"]
    return points



if __name__ == "__main__":

    # perspective 변환 이용을 위한 손가락 위치
    middle_finger_tip = None
    # perspective 변환 행렬
    mod_matrix = None
    # 시선 영역으로 설정 된 네 꼭짓점
    point_of_screen = {}

    # 마우스 제어 활성화 여부
    mouse_control_act = False

    # hand landmark 모듈 init
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands

    # VideoCapture 객체 생성 - 0번 카메라
    cap = cv2.VideoCapture(0)

    # 카메라 너비, 높이
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

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

            # 마우스 제어가 시작됐을 때
            if mouse_control_act:
                mouse_control()

            # 시선 영역 설정
            point_set(image, "q", 0)
            point_set(image, "p", 1)
            point_set(image, "m", 2)
            point_set(image, "z", 3)

            # 시선 영역 저장
            if cv2.waitKey(2) == ord('s'):
                save_point(point_of_screen)

            # 행렬 변환 실행
            if cv2.waitKey(4) == ord('t'):
                LogE.d("t", "pressed")
                persp_image, mod_matrix = persp_calc(image, point_of_screen, middle_finger_tip)

            # 시선 영역 표시
            if cv2.waitKey(4) == ord('r'):
                LogE.d("r", "pressed")
                cv2.imshow('perspectived', persp_image)

            # 시선 영역 점 위치 불러오기
            if cv2.waitKey(4) == ord('l'):
                point_of_screen = load_point()

            # 마우스 제어 시작
            if cv2.waitKey(2) == ord('c'):
                mouse_control_act = True
                LogE.g("control", "activated")

            # 마우스 제어 정지
            if cv2.waitKey(4) == ord('e'):
                mouse_control_act = False
                LogE.e("control", "deactivated")

    cap.release()