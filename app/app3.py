
# AIEN16 第二組 臉部偵測Sample code --> git clone from google mediapipe website.
# google MediaPipe - AI機器學習套件介紹, 它的套件包括: 
# 臉部偵測, 臉部特徵立體偵測, 手部關節運動追蹤, 人像與背景的分割, 髮型特徵分割, 人體立體運動姿勢追蹤, 物件辨識, 
# 臉/身體/支幹整體偵測, 3D物件偵測.
# https://google.github.io/mediapipe/
# google MediaPipe github 
# https://github.com/google/mediapipe
#

import cv2
import mediapipe as mp
import time
import math
import threading
import facialFeatures
 

# 創建一個lock對象: lock = threading.Lock()
# 獲取使用lock: lock.acquire()
# 解除釋放lock: lock.release()
# 釋放前其他Thread(執行緒)無法執行
lock = threading.Lock()

BLACK = (0, 0, 0)



class FaceMeshDetector():
    def __init__(self, staticMode=False, maxFaces=10, minDetectionCon=False, minTrackCon=0.5):
        
        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.minDetectionCon = minDetectionCon
        self.minTrackCon = minTrackCon
        self.mpDraw = mp.solutions.drawing_utils
        # 畫出臉上的線條
        self.mpDrawingStyles = mp.solutions.drawing_styles
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.staticMode, self.maxFaces, self.minDetectionCon, self.minTrackCon)
        # there are 468 points.
        self.drawSpec = self.mpDraw.DrawingSpec(color=(30,144,255), thickness=1, circle_radius=1) 

    def findFaceMesh(self, img, drawFaceLms=True, drawID=False, drawFortuneTelling=True):
        # 左右相反，for camera
        #self.imgRGB = cv2.cvtColor(cv2.flip(img, 1), cv2.COLOR_BGR2RGB) # opposite right/left for actual face sync-up detection when face turns left/right.
        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(self.imgRGB)
        # put faces[]/distance{} here to avoid error"local variable referenced before assignment" 
        faces = []
        distance = []
        READ_FACE=facialFeatures.READ_FACE        
        # multi_face_landmarks :臉上作完正規化的xyz座標
        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:                
                if drawFaceLms:
                    # image=img, landmark_list=faceLms, connections=self.mpFaceMesh.FACEMESH_TESSELATION, 
                    # landmark_drawing_spec=self.drawSpec,
                    # connection_drawing_spec=self.mpDrawingStyles.get_default_face_mesh_tesselation_style().
                    # 臉上的各點及其連線
                    self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACEMESH_TESSELATION,
                                                self.drawSpec, self.mpDrawingStyles.get_default_face_mesh_tesselation_style())
                    # image=img, landmark_list=faceLms, connections=self.mpFaceMesh.FACEMESH_CONTOURS, 
                    # landmark_drawing_spec=self.drawSpec,
                    # connection_drawing_spec=self.mpDrawingStyles.get_default_face_mesh_contours_style().
                    # goolge劃出的範例線條
                    # self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACEMESH_CONTOURS,
                    #                             self.drawSpec, self.mpDrawingStyles.get_default_face_mesh_contours_style())
                if drawFortuneTelling:
                    # draw specific IDs for fortune telling(畫出上面自行定義的各點)
                    for idx1,ff in enumerate(READ_FACE):    
                        summ=0
                        for idx2,value in enumerate(READ_FACE[idx1]):
                            # startID=READ_FACE[idx1][idx2][0]        
                            # endID=READ_FACE[idx1][idx2][1]          
                            startID, endID = value
                            # print(f'startID:{startID} endID:{endID} ')
                            # faceID to xyz                            
                            ih, iw, ic = img.shape
                            lm = faceLms.landmark[startID]
                            x, y, z = int(lm.x*iw), int(lm.y*ih), int(lm.z*ic)
                            startAddress = x, y
                            lm = faceLms.landmark[endID]
                            x, y, z = int(lm.x*iw), int(lm.y*ih), int(lm.z*ic)
                            endAddress = x, y    
                            # print(f'startAddress:{startAddress},endAddress:{endAddress}')
                            
                            # draw ID on image
                            # cv2.putText(img, str(startID), startAddress, cv2.FONT_HERSHEY_PLAIN, 0.7, (255, 0, 0), 1)
                            # cv2.putText(img, str(endID), endAddress, cv2.FONT_HERSHEY_PLAIN, 0.7, (255, 0, 0), 1)                     
                            lineDistance = self.drawSpecificLine(img, startAddress, endAddress)
                            lineDistance = int(lineDistance)
                            summ+=lineDistance
                            # print(lineDistance)
                            # print(summ)
                        # append into distance[]                           
                        distance.append(summ)                     
                    if distance:                        
                        print(f'RIGHT_EYEBROW:{distance[0]}, LEFT_EYEBROW:{distance[1]}, RIGHT_EYE:{distance[2]}, LEFT_EYE{distance[3]}, NOSE_LENGTH:{distance[4]}, NOSE_WIDTH:{distance[5]}, FOREHEAD:{distance[6]}, PHILTRUM:{distance[7]}, MOUTH:{distance[8]}')                  

                face = []
                for id, lm in enumerate(faceLms.landmark):   # use enumerate to get index and values 
                    ih, iw, ic = img.shape
                    x, y, z = int(lm.x*iw), int(lm.y*ih), int(lm.z*ic)
                    # 畫出臉上每個點的ID
                    if drawID:
                        cv2.putText(img, str(id), (x, y), cv2.FONT_HERSHEY_PLAIN, 0.7, (0, 255, 0), 1)  # print each ID on face
                    #print(lm)           # print x, y, z of each point
                    #print(id, x, y)
                    #face.append([x,y])  # save x, y without id. 
                    face.append([id,x,y,z])
                faces.append(face)
        return img, faces, distance
    
    # https://google.github.io/mediapipe/solutions/iris.html
    
        
    
    
    # Euclaidean Distance
    def euclaideanDistance(self, startPoint, endPoint):
        x1, y1 = startPoint
        x2, y2 = endPoint
        distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2 )
        return distance
    
    # draw specific line user defined
    def drawSpecificLine(self, img, startAddress, endAddress):
        # address for begin point and destination point
        self.startAddress = startAddress
        self.endAddress= endAddress        
        # using cv2.line to draw line 畫出指定點到點的線
        cv2.line(img, self.startAddress, self.endAddress, BLACK, 2)        
        # Euclaidean Distance 計算出距離
        LineDistance = self.euclaideanDistance(self.startAddress, self.endAddress)
        return LineDistance

def faceMeshDetection(videoMode=True, filePath="./videos/1-720p.mp4", drawFaceLms=True, drawID=False, drawFortuneTelling=True):
    #cap = cv2.VideoCapture(0)                         # From camera capture real time 
    #cap = cv2.VideoCapture("../videos/2-1080p.mp4")   # required AV1 hardware decoding
    #cap = cv2.VideoCapture("../videos/2-720pL.mp4")
    #cap = cv2.VideoCapture("../videos/1-720p.mp4")
    # grab global references to the lock variables
    global lock
    if videoMode:
        cap = cv2.VideoCapture(filePath)
    pTime = 0
    detector = FaceMeshDetector(maxFaces=10)
    #faceDetection = True
    while True:
        # success為True/False
        if videoMode:
            success, img = cap.read()
        else:
            #img = cv2.imread('images/03.png')
            img = cv2.imread(filePath)
        # In order to make all video_feed sources working at the same fps, we need to use threading lock. 
        with lock:                  # acquire threading lock, set the output frame, and release threading lock.  
            img, faces, distance = detector.findFaceMesh(img, drawFaceLms, drawID, drawFortuneTelling)      
                    
            # time.time():1970年之後經過的秒数
            cTime = time.time()
            fps = 1/(cTime-pTime)
            pTime = cTime
            cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
            #cv2.imshow("Image", img)
        
            # 傳送至前端
            frame = cv2.imencode('.jpg', img)[1].tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        if cv2.waitKey(1) == ord('q'): 
            break 
    if videoMode:    
        cap.release()   
    cv2.destroyAllWindows()
    return img, faces, distance

if __name__ == "__main__":
    faceMeshDetection()
    


