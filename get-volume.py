import cv2
import numpy as np
import time
import HandTrackingModule as htm
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

class VolumeControl:
    def __init__(self):
        self.wCam, self.hCam = 640, 480
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, self.wCam)
        self.cap.set(4, self.hCam)
        
        self.detector = htm.handDetector(detectionCon=0.7)
        
        # 音量控制初始化
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        self.volume = cast(interface, POINTER(IAudioEndpointVolume))  # type: ignore
        self.volRange = self.volume.GetVolumeRange()  # type: ignore
        self.minVol, self.maxVol = self.volRange[0], self.volRange[1]
        
        self.pTime = 0
        
    def run(self):
        while True:
            success, img = self.cap.read()
            if not success:
                break
            
            img = self.detector.findHands(img)
            lmList = self.detector.findPosition(img, draw=False)
            
            if lmList:
                x1, y1 = lmList[4][1], lmList[4][2]
                x2, y2 = lmList[8][1], lmList[8][2]
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                self.draw_hand_landmarks(img, x1, y1, x2, y2, cx, cy)

                length = np.hypot(x2 - x1, y2 - y1)
                vol = np.interp(length, [50, 300], [self.minVol, self.maxVol])
                volBar = np.interp(length, [50, 300], [400, 150])
                volPer = np.interp(length, [50, 300], [0, 100])
                
                self.volume.SetMasterVolumeLevel(vol, None)  # type: ignore
                
                if length < 50:
                    cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)

                self.draw_volume_bar(img, volBar, volPer)

            self.display_fps(img)
            cv2.imshow("Volume Control", img)
            
            if cv2.waitKey(1) & 0xFF == 27:  # ESC key
                break
        
        self.cap.release()
        cv2.destroyAllWindows()

    def draw_hand_landmarks(self, img, x1, y1, x2, y2, cx, cy):
        cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
        cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

    def draw_volume_bar(self, img, volBar, volPer):
        cv2.rectangle(img, (50, 150), (85, 400), (255, 0, 0), 3)
        cv2.rectangle(img, (50, int(volBar)), (85, 400), (255, 0, 0), cv2.FILLED)
        cv2.putText(img, f'{int(volPer)}%', (40, 450), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

    def display_fps(self, img):
        cTime = time.time()
        fps = 1 / (cTime - self.pTime)
        self.pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (10, 70), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

if __name__ == "__main__":
    vc = VolumeControl()
    vc.run()