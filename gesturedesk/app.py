import argparse
import json
from pathlib import Path
import time
import uuid
from collections import deque
import cv2
import numpy as np
from .core import Engine,LABELS,features,normalize
from .camera import Camera,Tracker
from .model import Classifier,train
from .ui import Dashboard

ROOT=Path(__file__).resolve().parent.parent

def parser():
    p=argparse.ArgumentParser(description='GestureDesk real-time hand interface')
    sub=p.add_subparsers(dest='command',required=True)
    for name in ('run','collect'):
        s=sub.add_parser(name)
        s.add_argument('--camera',type=int,default=0)
        s.add_argument('--backend',choices=['auto','dshow','msmf'],default='auto')
        s.add_argument('--hand-model',type=Path,default=ROOT/'models/hand_landmarker.task')
        s.add_argument('--classifier',type=Path)
        s.add_argument('--demo',action='store_true',help='No-camera keyboard presentation')
        s.add_argument('--headless',action='store_true')
        s.add_argument('--frames',type=int,default=0,help='Stop after N processed frames (smoke tests)')
        s.add_argument('--snapshot',type=Path)
        if name=='collect':
            s.add_argument('--clip-seconds',type=float,default=10.,help='Auto-stop each recording after this duration')
            s.add_argument('--sample-hz',type=float,default=5.,help='Maximum stored samples per second')
            s.add_argument('--participant',required=True,help='Pseudonymous participant ID')
            s.add_argument('--session',default=str(uuid.uuid4()),help='One independently recorded sitting')
            s.add_argument('--output',type=Path,default=ROOT/'data/landmarks.jsonl')
    t=sub.add_parser('train')
    t.add_argument('--data',type=Path,default=ROOT/'data/landmarks.jsonl')
    t.add_argument('--output',type=Path,default=ROOT/'models/custom.npz')
    t.add_argument('--report',type=Path,default=ROOT/'reports/evaluation.json')
    t.add_argument('--test-sessions',nargs='+',default=[])
    t.add_argument('--test-participants',nargs='+',default=[])
    t.add_argument('--validation-participants',nargs='+',default=[])
    t.add_argument('--validation-sessions',nargs='+',default=[])
    t.add_argument('--threshold',type=float,default=.75)
    t.add_argument('--margin',type=float,default=.15)
    return p

def live(args):
    if args.headless and not args.frames:
        raise ValueError('--headless requires a finite --frames limit')
    if args.command=='collect' and (not 0 < args.sample_hz <= 10 or not 0 < args.clip_seconds <= 60):
        raise ValueError('Use sample-hz in (0,10] and clip-seconds in (0,60]')
    if args.command=='collect' and (args.demo or args.headless):
        raise ValueError('Collection requires a real camera and interactive label controls')
    if args.frames<0:
        raise ValueError('--frames cannot be negative')
    model=Classifier(args.classifier) if args.classifier else None
    engine,ui=Engine(model),Dashboard(ROOT/'portfolio.json')
    camera=tracker=dataset=None
    recording,label,count,last_saved=False,'UNKNOWN',0,0.
    label_counts={label:0 for label in LABELS}
    timings=deque(maxlen=600)
    record_after=0.
    seq,processed,last_loop,start=-1,0,time.monotonic(),time.monotonic()
    view=None
    mode='KEYBOARD DEMO' if args.demo else ('CUSTOM CLASSIFIER' if model else 'GEOMETRY BASELINE')
    try:
        if not args.demo:
            if not args.hand_model.exists():
                raise RuntimeError('Hand model missing. Run python tools/download_model.py first.')
            tracker=Tracker(args.hand_model)
            camera=Camera(args.camera,args.backend)
        if args.command=='collect':
            if not args.session.strip() or not args.participant.strip():raise ValueError('Participant and session cannot be empty')
            # Validate existing data before appending; one session belongs to one person.
            from .model import read_rows
            if args.output.exists() and args.output.stat().st_size:
                for row in read_rows(args.output):
                    if row['session']==args.session and row.get('participant')!=args.participant:
                        raise ValueError('Session already belongs to another or legacy participant; use a new ID')
            args.output.parent.mkdir(parents=True,exist_ok=True)
            dataset=args.output.open('a',encoding='utf-8')
            print('Collection session:',args.session)
        while True:
            now=time.monotonic()
            if args.demo:
                frame=np.full((480,640,3),(34,29,23),np.uint8)
                cv2.putText(frame,'KEYBOARD PRESENTATION',(95,220),cv2.FONT_HERSHEY_SIMPLEX,.85,(187,226,79),2)
                cv2.putText(frame,'Connect a camera for live tracking',(75,260),cv2.FONT_HERSHEY_SIMPLEX,.6,(200,200,200),1)
                points,hand=None,'DEMO - no inference'
                captured=now
            else:
                if camera.error:raise RuntimeError(camera.error)
                slot=camera.latest()
                if slot is None or slot[0]==seq:
                    if now-start>5 and (slot is None or now-slot[1]>2):
                        raise RuntimeError('Camera timed out. Try another camera/backend.')
                    if not args.headless and cv2.waitKey(1)&255 in (27,ord('q')):break
                    time.sleep(.002)
                    continue
                seq,captured,raw=slot
                frame=cv2.flip(raw,1)
                points,hand=tracker.detect(frame,captured)
                if time.monotonic()-captured>.3:
                    points,hand=None,'STALE FRAME - actions stopped'
            size=(frame.shape[1],frame.shape[0])
            filtered,state,score=engine.update(points,captured,size,hand)
            if args.command!='collect':
                ui.action(state.event)
                ui.pen(filtered,state,size)
            if dataset and recording and now >= record_after + args.clip_seconds:
                recording=False
            if dataset and recording and now>=record_after and points is not None and now-last_saved>=1/args.sample_hz:
                try:
                    features(points)
                    dataset.write(json.dumps({'version':2,'participant':args.participant,'session':args.session,'label':label,
                        'normalized_landmarks':normalize(points).tolist(),
                        'timestamp':captured,'hand':hand,'frame_size':size,'landmarks':points.tolist()})+'\n')
                    dataset.flush();count+=1;label_counts[label]+=1;last_saved=now
                except ValueError:pass
            elapsed=time.monotonic()-now
            timings.append((elapsed*1000,(time.monotonic()-captured)*1000))
            fps=1/max(time.monotonic()-last_loop,1e-6);last_loop=time.monotonic()
            status=None
            if dataset:
                recording_text='SETTLE' if recording and now<record_after else ('ON' if recording else 'OFF')
                status=f'1 palm 2 fist 3 point 4 peace 5 thumb 6 pinch 7 unknown | R: {recording_text} | {label}: {label_counts[label]} | total {count}'
            view=ui.render(frame,filtered,state,score,hand,(fps,elapsed*1000,(time.monotonic()-captured)*1000),mode,status)
            processed+=1
            if args.frames and processed>=args.frames:break
            if not args.headless:
                cv2.imshow('GestureDesk',view)
                key=cv2.waitKeyEx(1)
                if key in (27,ord('q')):break
                if key==ord('d'):ui.drawing=not ui.drawing;ui.last_pen=None
                elif key==ord('h'):ui.debug=not ui.debug
                elif key==ord('c'):ui.canvas.fill(0);ui.last_pen=None
                elif key==32:ui.paused=not ui.paused;ui.last_pen=None;engine.debounce.reset()
                elif key in (2555904,83):ui.action('SWIPE_RIGHT')
                elif key in (2424832,81):ui.action('SWIPE_LEFT')
                elif dataset and ord('1')<=key<=ord('7'):
                    label=LABELS[key-ord('1')];recording=False
                elif dataset and key==ord('r'):
                    recording=not recording;record_after=time.monotonic()+1.
                if cv2.getWindowProperty('GestureDesk',cv2.WND_PROP_VISIBLE)<1:break
            if args.demo:time.sleep(.025)
    finally:
        if camera:camera.close()
        if tracker:tracker.close()
        if dataset:dataset.close()
        cv2.destroyAllWindows()
        if args.snapshot and view is not None:
            args.snapshot.parent.mkdir(parents=True,exist_ok=True)
            if not cv2.imwrite(str(args.snapshot),view):raise RuntimeError('Could not write snapshot')
    if timings:
        t=np.asarray(timings)
        print(f'Processing median {np.median(t[:,0]):.1f} ms; frame age p95 {np.percentile(t[:,1],95):.1f} ms')
    return processed

def main():
    args=parser().parse_args()
    try:
        if args.command=='train':
            if not .5<=args.threshold<=1 or not 0<=args.margin<=1:raise ValueError('Invalid rejection thresholds')
            results=train(args.data,args.output,args.report,args.test_sessions,args.validation_sessions,args.threshold,args.margin,
                          test_participants=args.test_participants,validation_participants=args.validation_participants)
            print('Saved classifier:',args.output,'\nSaved session evaluation:',args.report)
            print('Held-out test accuracy:',results['splits']['test']['metrics']['accuracy'])
        else:print('Processed frames:',live(args))
    except (RuntimeError,ValueError,OSError) as e:
        print('GestureDesk:',e)
        return 1
    return 0
