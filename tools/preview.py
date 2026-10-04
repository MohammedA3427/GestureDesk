"""Generate clearly marked camera-free UI previews, not claimed CV results."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import cv2
import numpy as np
from gesturedesk.core import State
from gesturedesk.ui import Dashboard,ACCENT
from tests.test_engine import hand

root=Path(__file__).resolve().parent.parent
folder=root/'reports';folder.mkdir(exist_ok=True)
ui=Dashboard(root/'portfolio.json')
frame=np.full((480,640,3),(34,29,23),np.uint8)
cv2.putText(frame,'SYNTHETIC LANDMARK PREVIEW',(30,50),cv2.FONT_HERSHEY_SIMPLEX,.65,ACCENT,1,cv2.LINE_AA)
cv2.putText(frame,'Layout only - not a webcam result',(30,85),cv2.FONT_HERSHEY_SIMPLEX,.55,(180,180,180),1,cv2.LINE_AA)
for name,drawing in [('landmarks',False),('drawing',True),('collection',False)]:
    ui.drawing=drawing
    if drawing:
        pts=np.array([[100,240],[140,140],[180,240],[220,140],[260,240],[300,140],[340,240],[380,140]],np.int32)
        cv2.polylines(ui.canvas,[pts],False,ACCENT,4,cv2.LINE_AA)
    status='1 palm 2 fist 3 point 4 peace 5 thumb 6 pinch 7 unknown | R: SETTLE | UNKNOWN: 0 | total 0' if name=='collection' else None
    image=ui.render(frame,hand(),State('OPEN_PALM','OPEN_PALM',progress=1),.85,'Synthetic',(0,0,0),'SYNTHETIC PREVIEW',status)
    cv2.imwrite(str(folder/f'{name}-preview.png'),image)
    print(folder/f'{name}-preview.png')
