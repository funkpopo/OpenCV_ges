import cv2
import mediapipe as mp
import math
import numpy as np

def vector_2d_angle(v1, v2):
    '''计算二维向量的角度'''
    v1_x, v1_y = v1
    v2_x, v2_y = v2
    try:
        angle = math.degrees(math.acos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))))
    except:
        return 65535.
    return min(angle, 180.)

def hand_angle(hand_):
    '''获取对应手相关向量的二维角度'''
    angle_list = []
    for finger in range(5):
        v1 = np.array(hand_[0]) - np.array(hand_[finger * 4 + 2])
        v2 = np.array(hand_[finger * 4 + 3]) - np.array(hand_[finger * 4 + 4])
        angle_list.append(vector_2d_angle(v1, v2))
    return angle_list

def h_gesture(angle_list):
    '''定义手势'''
    thr_angle, thr_angle_thumb, thr_angle_s = 65., 53., 49.
    gesture_dict = {
        "fist": lambda al: all(a > (thr_angle_thumb if i == 0 else thr_angle) for i, a in enumerate(al)),
        "five": lambda al: all(a < thr_angle_s for a in al),
        "gun": lambda al: al[0] < thr_angle_s and al[1] < thr_angle_s and all(a > thr_angle for a in al[2:]),
        "love": lambda al: al[0] < thr_angle_s and al[1] < thr_angle_s and al[2] > thr_angle and al[3] > thr_angle and al[4] < thr_angle_s,
        "one": lambda al: al[0] > 5 and al[1] < thr_angle_s and all(a > thr_angle for a in al[2:]),
        "six": lambda al: al[0] < thr_angle_s and all(a > thr_angle for a in al[1:4]) and al[4] < thr_angle_s,
        "three": lambda al: al[0] > thr_angle_thumb and all(a < thr_angle_s for a in al[1:4]) and al[4] > thr_angle,
        "thumbUp": lambda al: al[0] < thr_angle_s and all(a > thr_angle for a in al[1:]),
        "two": lambda al: al[0] > thr_angle_thumb and al[1] < thr_angle_s and al[2] < thr_angle_s and al[3] > thr_angle and al[4] > thr_angle
    }
    
    for gesture, condition in gesture_dict.items():
        if condition(angle_list):
            return gesture
    return None

def detect():
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, 
                           min_detection_confidence=0.75, min_tracking_confidence=0.75)
    
    cap = cv2.VideoCapture(0)
    cv2.namedWindow('MediaPipe Hands', 0)
    cv2.resizeWindow('MediaPipe Hands', 500, 400)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        results = hands.process(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                hand_local = [(lm.x * frame.shape[1], lm.y * frame.shape[0]) for lm in hand_landmarks.landmark]
                
                angle_list = hand_angle(hand_local)
                gesture_str = h_gesture(angle_list)
                if gesture_str:
                    cv2.putText(frame, gesture_str, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('MediaPipe Hands', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    detect()