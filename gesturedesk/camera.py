"""Single-slot capture: slow inference cannot build an application frame queue."""
import threading
import time
import cv2

class Camera:
    def __init__(self,index=0,backend='auto'):
        api = {'auto':cv2.CAP_ANY,'dshow':cv2.CAP_DSHOW,'msmf':cv2.CAP_MSMF}[backend]
        self.cap = cv2.VideoCapture(index,api)
        if not self.cap.isOpened():
            self.cap.release()
            raise RuntimeError(f'Cannot open camera {index}. Close other camera apps; try --backend dshow or --camera 1.')
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
        self.cap.set(cv2.CAP_PROP_FPS,30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE,1)
        self.lock,self.stop = threading.Lock(),threading.Event()
        self.slot,self.error,self.sequence = None,None,0
        self.thread = threading.Thread(target=self._capture,daemon=True)
        self.thread.start()
    def _capture(self):
        failures = 0
        try:
            while not self.stop.is_set():
                ok,frame = self.cap.read()
                if not ok:
                    failures += 1
                    if failures >= 15:
                        self.error = 'Camera disconnected or stopped delivering frames'
                        break
                    self.stop.wait(.03)
                    continue
                failures = 0
                with self.lock:
                    self.sequence += 1
                    self.slot = (self.sequence,time.monotonic(),frame)
        except Exception as e:
            self.error = f'Camera capture failed: {e}'
        finally:
            self.cap.release()
    def latest(self):
        with self.lock:
            return self.slot
    def close(self):
        self.stop.set()
        self.thread.join(timeout=2)

class Tracker:
    def __init__(self,path):
        import os
        from pathlib import Path
        # MediaPipe imports matplotlib; keep its optional cache within the project.
        os.environ.setdefault('MPLCONFIGDIR',str(Path(path).parent.parent/'work/matplotlib'))
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import HandLandmarker,HandLandmarkerOptions,RunningMode
        self.mp,self.timestamp = mp,-1
        self.task = HandLandmarker.create_from_options(HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(path)),running_mode=RunningMode.VIDEO,
            num_hands=2,min_hand_detection_confidence=.65,min_hand_presence_confidence=.65,min_tracking_confidence=.65))
    def detect(self,frame,now):
        import numpy as np
        self.timestamp = max(self.timestamp+1,int(now*1000))
        rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        result = self.task.detect_for_video(self.mp.Image(image_format=self.mp.ImageFormat.SRGB,data=rgb),self.timestamp)
        if len(result.hand_landmarks) != 1:
            return None, 'NONE' if not result.hand_landmarks else 'MULTIPLE HANDS'
        h,w = frame.shape[:2]
        points = np.array([(p.x*w,p.y*h,p.z*w) for p in result.hand_landmarks[0]])
        # A clipped hand cannot safely drive actions.
        if np.any(points[:,:2] < 0) or np.any(points[:,0] >= w) or np.any(points[:,1] >= h):
            return None,'HAND CLIPPED'
        if np.linalg.norm(points[9]-points[0]) < w*.06:
            return None,'MOVE HAND CLOSER'
        return points,result.handedness[0][0].category_name
    def close(self):
        self.task.close()
