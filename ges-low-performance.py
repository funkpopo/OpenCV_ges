import cv2
from HandTrackingModule1 import HandDetector

class Main:
    def __init__(self):
        self.camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.camera.set(3, 1280)
        self.camera.set(4, 720)
        self.detector = HandDetector()  # 将detector移到__init__中初始化

    def get_gesture(self, fingers):
        gesture_map = {
            (0, 1, 1, 0, 0): "2_TWO",
            (0, 1, 1, 1, 0): "3_THREE",
            (0, 1, 1, 1, 1): "4_FOUR",
            (1, 1, 1, 1, 1): "5_FIVE",
            (0, 1, 0, 0, 0): "1_ONE",
            (1, 0, 0, 0, 0): "GOOD!"
        }
        return gesture_map.get(tuple(fingers), "")

    def Gesture_recognition(self):
        cv2.namedWindow('camera', 0)
        cv2.resizeWindow('camera', 500, 400)

        while True:
            _, img = self.camera.read()
            img = self.detector.findHands(img)
            lmList, bbox = self.detector.findPosition(img)

            if lmList:
                x_1, y_1 = bbox["bbox"][0], bbox["bbox"][1]  # type: ignore
                fingers = self.detector.fingersUp()
                gesture = self.get_gesture(fingers)
                if gesture:
                    cv2.putText(img, gesture, (x_1, y_1), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)

            cv2.imshow("camera", img)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        self.camera.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    Solution = Main()
    Solution.Gesture_recognition()