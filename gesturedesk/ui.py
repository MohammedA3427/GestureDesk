"""OpenCV dashboard, intentionally independent of capture/inference."""
import json
from collections import deque
from pathlib import Path
import cv2
import numpy as np
from .core import CONNECTIONS

BG=(19,16,12); PANEL=(34,29,23); TEXT=(239,239,225); MUTED=(163,155,135); ACCENT=(187,226,79)

def text(image,value,x,y,size=.55,color=TEXT):
    cv2.putText(image,str(value),(int(x),int(y)),cv2.FONT_HERSHEY_SIMPLEX,size,color,1,cv2.LINE_AA)

class Dashboard:
    def __init__(self,portfolio):
        self.portfolio=json.loads(Path(portfolio).read_text(encoding='utf-8'))
        if not self.portfolio.get('cards'):
            raise ValueError('portfolio.json needs at least one card')
        if not all(isinstance(self.portfolio.get(k),str) for k in ('name','subtitle')):
            raise ValueError('Portfolio name and subtitle must be text')
        for card in self.portfolio['cards']:
            if not isinstance(card,dict) or not all(isinstance(card.get(k),str) for k in ('title','tag')) or not isinstance(card.get('lines'),list) or not all(isinstance(x,str) for x in card['lines']):
                raise ValueError('Each portfolio card needs a title, tag, and list of text lines')
        self.card,self.drawing,self.paused,self.debug=0,False,False,True
        self.canvas=np.zeros((360,510,3),np.uint8)
        self.last_pen=None
        self.history=deque(maxlen=5)
    def action(self,event):
        if not event:
            return
        self.history.appendleft(event)
        if self.paused:
            return
        if event=='SWIPE_RIGHT': self.card=(self.card+1)%len(self.portfolio['cards'])
        elif event=='SWIPE_LEFT': self.card=(self.card-1)%len(self.portfolio['cards'])
        elif event=='PEACE': self.drawing=not self.drawing; self.last_pen=None
    def pen(self,points,state,size):
        if self.paused or not self.drawing or points is None or state.raw!='PINCH' or state.stable!='PINCH':
            self.last_pen=None
            return
        # Map camera space to canvas; drawing stops on recognition uncertainty.
        pos=tuple(np.clip(points[8,:2]/np.asarray(size)*[510,360],[0,0],[509,359]).astype(int))
        if self.last_pen is not None and np.linalg.norm(np.array(pos)-self.last_pen)<100:
            cv2.line(self.canvas,self.last_pen,pos,ACCENT,4,cv2.LINE_AA)
        self.last_pen=pos
    def render(self,frame,points,state,score,hand,metrics,mode='GEOMETRY BASELINE',collection=None):
        out=np.full((800,1280,3),BG,np.uint8)
        text(out,'G E S T U R E D E S K',28,48,.9,ACCENT)
        text(out,self.portfolio['name']+' / '+self.portfolio['subtitle'],28,78,.5,MUTED)
        text(out,mode,825,45,.5,ACCENT)
        if self.paused: text(out,'PAUSED - press SPACE',825,78,.6,(100,180,255))
        elif self.drawing: text(out,'AIR DRAW / pinch to ink',825,78,.6)
        else: text(out,'PORTFOLIO / open-palm swipes',825,78,.6)
        image=cv2.resize(frame,(640,480)).copy()
        if points is not None and self.debug:
            h,w=frame.shape[:2]
            xy=(points[:,:2]*[640/w,480/h]).astype(int)
            for a,b in CONNECTIONS: cv2.line(image,tuple(xy[a]),tuple(xy[b]),ACCENT,2,cv2.LINE_AA)
            for i,p in enumerate(xy):
                cv2.circle(image,tuple(p),4,(240,240,240),-1,cv2.LINE_AA)
                text(image,i,p[0]+5,p[1]-5,.3,ACCENT)
        out[110:590,28:668]=image
        cv2.rectangle(out,(692,110),(1252,590),PANEL,-1)
        if self.drawing:
            out[155:515,717:1227]=self.canvas
            text(out,'Pinch to draw. Release to move. C clears.',715,555,.5,MUTED)
        else:
            card=self.portfolio['cards'][self.card]
            text(out,card['tag'],718,153,.48,ACCENT)
            # Fit long user-supplied headings and wrap lines to panel width.
            title_size=min(.9, 500/(max(1,cv2.getTextSize(card['title'],cv2.FONT_HERSHEY_SIMPLEX,1,1)[0][0])))
            text(out,card['title'],718,212,title_size)
            y=270
            for line in card['lines']:
                words=line.split(); current=''
                for word in words:
                    trial=(current+' '+word).strip()
                    if cv2.getTextSize(trial,cv2.FONT_HERSHEY_SIMPLEX,.54,1)[0][0]>500 and current:
                        text(out,current,718,y,.54);y+=28;current=word
                    else:current=trial
                text(out,current,718,y,.54);y+=42
                if y>500:break
            text(out,f'{self.card+1:02} / {len(self.portfolio["cards"]):02}     <  SWIPE TO EXPLORE  >',718,555,.5,MUTED)
        text(out,'HAND: '+hand,28,623,.55,ACCENT)
        text(out,'RAW: '+state.raw,28,655,.6)
        text(out,'STABLE: '+state.stable,320,655,.6)
        label='heuristic' if mode=='GEOMETRY BASELINE' else 'model score'
        text(out,f'{label}: {score:.2f}',28,685,.5,MUTED)
        cv2.rectangle(out,(320,672),(650,684),PANEL,-1)
        cv2.rectangle(out,(320,672),(320+int(330*state.progress),684),ACCENT,-1)
        text(out,f'{metrics[0]:.1f} FPS | process {metrics[1]:.0f} ms | frame age {metrics[2]:.0f} ms',692,623,.5,MUTED)
        text(out,'EVENTS: '+' / '.join(self.history),692,655,.45,ACCENT)
        text(out,'D draw | arrows browse | SPACE pause | C clear | H skeleton | Q quit',28,746,.55)
        text(out,collection or 'PEACE toggles canvas. PINCH draws. Open palm swipes browse.',28,778,.5,MUTED)
        return out
