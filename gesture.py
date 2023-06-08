import cv2
import mediapipe as mp

# Mediapipe를 위한 초기화
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# 제스처 매핑
gesture_mapping = {
    0: '손 열기',
    1: '주먹 쥐기'
}

# VideoCapture 객체 생성
cap = cv2.VideoCapture(0)

# Mediapipe Hands 모델 로드
with mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        # 이미지를 BGR에서 RGB로 변환
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Mediapipe에 이미지 전달
        results = hands.process(image)

        # 손 인식 결과 확인
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # 감지된 손 랜드마크에 점 표시
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(150, 150, 150), thickness=2)
                )

                # 손 제스처 인식
                gesture_id = 0  # 기본 제스처 ID
                if hand_landmarks.landmark:
                    pass

                # 인식된 제스처 출력
                gesture_name = gesture_mapping.get(gesture_id, '알 수 없음')
                cv2.putText(image, gesture_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # 이미지를 RGB에서 BGR로 변환하여 출력
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imshow('Gesture Recognition', image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# 종료
cap.release()
cv2.destroyAllWindows()
