"""Offline real-model inference check; never opens a camera or stores images."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import numpy as np
from gesturedesk.camera import Tracker

path=Path(__file__).resolve().parent.parent/'models/hand_landmarker.task'
tracker=Tracker(path)
try:
    for timestamp in (0.,.033,.066):
        points,status=tracker.detect(np.zeros((480,640,3),np.uint8),timestamp)
        assert points is None and status=='NONE', 'Blank frame should not yield a usable hand'
    print('Official MediaPipe model: 3 timestamped blank-frame inference checks passed.')
finally:tracker.close()
