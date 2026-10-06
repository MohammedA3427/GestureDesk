"""Camera-independent geometry and event state machines."""
from collections import deque
from dataclasses import dataclass
import numpy as np

LABELS = ('OPEN_PALM', 'FIST', 'POINT', 'PEACE', 'THUMBS_UP', 'PINCH', 'UNKNOWN')
CONNECTIONS = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),
               (9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),
               (13,17),(0,17),(17,18),(18,19),(19,20)]

def valid_hand(points):
    p = np.asarray(points, dtype=float)
    return p.shape == (21,3) and np.isfinite(p).all() and np.linalg.norm(p[9]-p[0]) > 1e-5

def normalize(points):
    """Wrist origin, palm scale, in-plane rotation, left/right canonicalization.

    Input x/y/z must share units (pixel-equivalent coordinates in live capture).
    Keep depth as measured; never infer a physical camera distance.
    """
    if not valid_hand(points):
        raise ValueError('Expected 21 finite non-degenerate xyz landmarks')
    p = np.asarray(points, float) - points[0]
    scale = np.linalg.norm(p[9])
    p /= scale
    direction = p[9,:2]
    length = np.linalg.norm(direction)
    if length < .1:
        raise ValueError('Palm is too edge-on for reliable normalization')
    ux, uy = direction / length
    xy = p[:,:2].copy()
    p[:,0] = xy[:,0] * (-uy) + xy[:,1] * ux
    p[:,1] = -(xy[:,0] * ux + xy[:,1] * uy)
    if p[5,0] < p[17,0]:
        p[:,0] *= -1
    return p

def features(points):
    p = normalize(points)
    distances = [np.linalg.norm(p[a]-p[b]) for a,b in [(4,8),(8,12),(12,16),(16,20),(4,5)]]
    return np.concatenate((p.ravel(), distances)).astype(np.float32)

def angle(a,b,c):
    u,v = a-b,c-b
    return float(np.degrees(np.arccos(np.clip(np.dot(u,v)/(np.linalg.norm(u)*np.linalg.norm(v)+1e-9),-1,1))))

def rule_gesture(points):
    """Conservative geometry baseline. Scores are heuristic, not probabilities."""
    try:
        p = normalize(points)
    except ValueError:
        return 'UNKNOWN', 0.
    extended, folded = [], []
    for base,pip,dip,tip in [(5,6,7,8),(9,10,11,12),(13,14,15,16),(17,18,19,20)]:
        straight = angle(p[base],p[pip],p[tip])
        reach = np.linalg.norm(p[tip]-p[0]) - np.linalg.norm(p[pip]-p[0])
        extended.append(straight > 155 and reach > .25)
        folded.append(straight < 125 and reach < .15)
    thumb = angle(p[2],p[3],p[4]) > 150 and np.linalg.norm(p[4]-p[5]) > .55
    pinch = np.linalg.norm(p[4]-p[8])
    if pinch < .25 and sum(extended[1:]) >= 2:
        return 'PINCH', .85
    if all(extended) and thumb:
        return 'OPEN_PALM', .85
    # Image-space up requirement prevents a sideways thumb being a thumbs-up.
    if all(folded) and thumb and points[4,1] < points[3,1] < points[2,1]:
        return 'THUMBS_UP', .8
    if all(folded) and np.linalg.norm(p[4]-p[9]) < 1.1:
        return 'FIST', .8
    if extended[0] and all(folded[1:]) and not thumb:
        return 'POINT', .85
    if all(extended[:2]) and all(folded[2:]) and not thumb:
        return 'PEACE', .85
    return 'UNKNOWN', 0.

class LandmarkFilter:
    def __init__(self, tau=.065):
        self.tau, self.last, self.time = tau, None, None
    def reset(self):
        self.last = self.time = None
    def update(self, p, now):
        if self.last is None or now-self.time > .25:
            self.last = p.copy()
        else:
            alpha = 1-np.exp(-max(0,now-self.time)/self.tau)
            self.last += alpha*(p-self.last)
        self.time = now
        return self.last.copy()

@dataclass
class State:
    raw: str = 'NONE'
    stable: str = 'NONE'
    event: str | None = None
    progress: float = 0.

class Debouncer:
    """Dwell before activation, one event per hold, neutral dwell to re-arm."""
    def __init__(self, dwell=.22, release=.16):
        self.dwell, self.release = dwell, release
        self.reset()
    def reset(self):
        self.candidate, self.since, self.stable, self.armed = 'NONE', None, 'NONE', True
        self.fired_label = None
    def update(self, label, now):
        if label == 'NONE':
            self.reset()
            return State()
        if label != self.candidate:
            self.candidate, self.since = label, now
        age = now-self.since
        if label == 'UNKNOWN':
            self.stable = 'UNKNOWN'
            if age >= self.release:
                self.armed = True
            return State(label,self.stable)
        event = None
        if age >= self.dwell:
            if self.fired_label != label:
                # A different deliberate gesture also re-arms once it has dwelled.
                self.armed = True
            self.stable = label
            if self.armed:
                event, self.armed = label, False
                self.fired_label = label
        return State(label,self.stable,event,min(1,age/self.dwell))

class SwipeDetector:
    """Open-palm horizontal travel in mirrored display coordinates."""
    def __init__(self):
        self.history = deque()
        self.cooldown = 0.
    def reset(self):
        self.history.clear()
    def update(self, center, label, now):
        if label != 'OPEN_PALM':
            self.reset()
            return None
        if now < self.cooldown:
            return None
        self.history.append((now,*center))
        while self.history and now-self.history[0][0] > .65:
            self.history.popleft()
        if len(self.history) < 4:
            return None
        first = self.history[0]
        dt,dx,dy = now-first[0],center[0]-first[1],center[1]-first[2]
        steps = np.diff([p[1] for p in self.history])
        consistency = abs(steps.sum())/(np.abs(steps).sum()+1e-9)
        if .15 <= dt <= .65 and abs(dx) > .23 and abs(dy) < .12 and consistency > .8:
            self.cooldown = now+.8
            self.reset()
            return 'SWIPE_RIGHT' if dx > 0 else 'SWIPE_LEFT'
        return None

class Engine:
    def __init__(self, classifier=None):
        self.filter, self.debounce, self.swipe = LandmarkFilter(), Debouncer(), SwipeDetector()
        self.classifier = classifier
        self.identity, self.last_time, self.last_wrist = None, None, None
    def update(self, points, now, frame_size=(640,480), identity='Right'):
        if points is None or not valid_hand(points):
            self.filter.reset(); self.debounce.reset(); self.swipe.reset()
            self.identity = self.last_time = self.last_wrist = None
            return None, State(), 0.
        if (identity != self.identity or self.last_time is None or now-self.last_time > .25
                or np.linalg.norm((points[0]-self.last_wrist)[:2]) > frame_size[0]*.35):
            self.filter.reset(); self.debounce.reset(); self.swipe.reset()
        self.identity, self.last_time, self.last_wrist = identity, now, points[0].copy()
        # Classify raw features so collection/training/live inference have identical inputs.
        label, score = self.classifier.predict(points) if self.classifier else rule_gesture(points)
        filtered = self.filter.update(points,now)
        state = self.debounce.update(label,now)
        center = np.mean(points[[0,5,9,13,17],:2],axis=0)/np.asarray(frame_size)
        swipe = self.swipe.update(center,label,now)
        if swipe:
            state.event = swipe
        return filtered,state,score


class DrawingPinch:
    """Pen-only gate: tip distance in wrist-to-middle-MCP palm lengths."""
    def __init__(self, close=.20, release=.32, dwell=.06):
        self.close, self.release, self.dwell = close, release, dwell
        self.reset()
    def reset(self):
        self.active, self.since, self.last_time, self.identity = False, None, None, None
    def update(self, points, now, identity):
        if points is None or not valid_hand(points):
            self.reset()
            return False
        if (identity != self.identity or self.last_time is None
                or not 0 <= now-self.last_time <= .25):
            self.reset()
        self.last_time, self.identity = now, identity
        try:
            p = normalize(points)
        except ValueError:
            self.reset()
            return False
        distance = np.linalg.norm(p[4]-p[8])
        if distance >= self.release:
            self.active, self.since = False, None
        elif not self.active:
            if distance <= self.close:
                if self.since is None:
                    self.since = now
                self.active = now-self.since >= self.dwell
            else:
                self.since = None
        return self.active
