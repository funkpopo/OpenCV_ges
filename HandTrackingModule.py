import cv2
import mediapipe as mp
import time

class handDetector():
    def __init__(self,mode = False,maxHands = 2,comp = 1,detectionCon = 0.5,trackCon = 0.5): # 这里由于函数库更新，所以多了一个复杂度参数，默认设为1
        self.mode = mode
        self.maxHands = maxHands
        self.comp = comp
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        # 导入mediapipe手部识别函数
        self.mpHands = mp.solutions.hands  # type: ignore
        self.hands = self.mpHands.Hands(self.mode,self.maxHands,self.comp,
                                        self.detectionCon,self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils  # type: ignore

        self.handLmStyle = self.mpDraw.DrawingSpec(color=(0, 0, 255), thickness=5)  # 点的样式，前一个参数是颜色，后一个是粗细
        self.handConStyle = self.mpDraw.DrawingSpec(color=(0, 255, 0), thickness=3)  # 线的样式BGR，前一个参数是颜色，后一个是粗细
        
    def findHands(self,img,draw = True):
        imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB) # 将图像转化为RGB图像
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    # self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS, self.handLmStyle, self.handConStyle)  # 画出点和线
        return img
    def findPosition(self,img,handNo = 0,draw = True):
        lmlist = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                # print(id,lm)
                h, w, c = img.shape  # 得到图像的长宽以及通道数
                cx, cy = int(lm.x * w), int(lm.y * h)  # 计算出中心点位置
                # print(id, cx, cy)
                lmlist.append([id,cx,cy])
                if id == 0:
                    cv2.circle(img, (cx, cy), 15, (0, 0, 255), cv2.FILLED)
        return lmlist

def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)  # 捕获摄像头
    detector = handDetector()
    while True:
        success, img = cap.read()  # 读入每一帧图像
        img = detector.findHands(img)
        limist = detector.findPosition(img)
        if len(limist) != 0:
            print(limist[4])
        cTime = time.time() # 用于计算FPS
        fps = 1/(cTime-pTime)
        pTime = cTime
        cv2.putText(img,f"FPS:{int(fps)}",(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),3) # 在图像上画出实时FPS
        # print(results.multi_hand_landmarks)
        cv2.namedWindow('Image', 0)
        cv2.resizeWindow('Image', 500, 400)
        cv2.imshow("Image",img) # 展示图像
        cv2.waitKey(1) # 延迟1ms
if __name__ == "__main__":
    main()