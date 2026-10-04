"""Portable numeric classifier artifact; no pickle execution on model loading."""
from pathlib import Path
import json
import numpy as np
from .core import features, LABELS

FEATURE_VERSION = 1

class Classifier:
    def __init__(self, path):
        with np.load(path,allow_pickle=False) as a:
            if int(a['version']) != FEATURE_VERSION:
                raise ValueError('Model feature version does not match application')
            self.mean,self.scale,self.weights,self.bias = [a[k].copy() for k in ('mean','scale','weights','bias')]
            self.classes = a['classes'].astype(str)
            self.threshold,self.margin = float(a['threshold']),float(a['margin'])
        if (self.mean.shape != (68,) or self.scale.shape != (68,) or
            self.weights.shape != (len(self.classes),68) or self.bias.shape != (len(self.classes),)
            or len(self.classes) < 3 or len(set(self.classes)) != len(self.classes)
            or not set(self.classes) <= set(LABELS) or 'UNKNOWN' not in self.classes
            or not all(np.isfinite(x).all() for x in (self.mean,self.scale,self.weights,self.bias))
            or np.any(self.scale <= 0) or not .5 <= self.threshold <= 1 or not 0 <= self.margin <= 1):
            raise ValueError('Invalid classifier artifact')
    def probabilities(self, x):
        logits = ((x-self.mean)/self.scale) @ self.weights.T + self.bias
        logits -= np.max(logits,axis=-1,keepdims=True)
        p = np.exp(logits)
        return p / p.sum(axis=-1,keepdims=True)
    def predict(self, points):
        try:
            p = self.probabilities(features(points))
        except ValueError:
            return 'UNKNOWN',0.
        order = np.argsort(p)
        score = float(p[order[-1]])
        label = str(self.classes[order[-1]])
        if score < self.threshold or score-float(p[order[-2]]) < self.margin:
            label = 'UNKNOWN'
        return label,score

def load_dataset(path):
    rows = []
    for number,line in enumerate(Path(path).read_text(encoding='utf-8').splitlines(),1):
        try:
            r = json.loads(line)
            p = np.asarray(r['landmarks'],float)
            if r['label'] not in LABELS or not isinstance(r['session'],str) or not r['session'].strip():
                raise ValueError('invalid label or session')
            rows.append((features(p),r['label'],r['session']))
        except (KeyError,ValueError,TypeError) as e:
            raise ValueError(f'Invalid dataset row {number}: {e}') from e
    if not rows:
        raise ValueError('Dataset is empty')
    return np.stack([r[0] for r in rows]),np.array([r[1] for r in rows]),np.array([r[2] for r in rows])

def train(data, output, report, test_sessions, validation_sessions, threshold=.75, margin=.15):
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report,confusion_matrix
    if Path(output).suffix!='.npz':
        raise ValueError('Classifier output must end in .npz')
    if not .5<=threshold<=1 or not 0<=margin<=1:
        raise ValueError('Invalid rejection thresholds')
    x,y,groups = load_dataset(data)
    test_sessions, validation_sessions = set(test_sessions),set(validation_sessions)
    if not test_sessions or not validation_sessions or test_sessions & validation_sessions:
        raise ValueError('Specify disjoint validation and test sessions')
    if not (test_sessions | validation_sessions) <= set(groups):
        raise ValueError('Requested held-out sessions not present in dataset')
    masks = {'test':np.isin(groups,list(test_sessions)), 'validation':np.isin(groups,list(validation_sessions))}
    masks['train'] = ~(masks['test'] | masks['validation'])
    classes = set(y)
    if len(classes) < 3 or 'UNKNOWN' not in classes:
        raise ValueError('Collect UNKNOWN and at least two intentional gestures')
    for name,mask in masks.items():
        if set(y[mask]) != classes:
            raise ValueError(f'{name} must contain every collected class; collect more independent sessions')
        for label in classes:
            if np.sum(y[mask] == label) < 10:
                raise ValueError(f'{name}/{label}: need at least 10 samples')
    scaler = StandardScaler().fit(x[masks['train']])
    clf = LogisticRegression(max_iter=2000,class_weight='balanced',C=1.).fit(scaler.transform(x[masks['train']]),y[masks['train']])
    Path(output).parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output,version=FEATURE_VERSION,mean=scaler.mean_,scale=scaler.scale_,
                        weights=clf.coef_,bias=clf.intercept_,classes=clf.classes_,threshold=threshold,margin=margin)
    model = Classifier(output)
    results = {'feature_version':FEATURE_VERSION,'threshold':threshold,'margin':margin,
               'note':'Fixed thresholds; validation is for development. Do not tune against test results.',
               'classes':clf.classes_.tolist(),'splits':{}}
    for name,mask in masks.items():
        p = model.probabilities(x[mask])
        order = np.sort(p,axis=1)
        pred = model.classes[np.argmax(p,axis=1)].copy()
        pred[(order[:,-1]<threshold) | ((order[:,-1]-order[:,-2])<margin)] = 'UNKNOWN'
        results['splits'][name] = {'sessions':sorted(set(groups[mask])), 'samples':int(mask.sum()),
            'metrics':classification_report(y[mask],pred,labels=clf.classes_,output_dict=True,zero_division=0),
            'confusion_matrix':confusion_matrix(y[mask],pred,labels=clf.classes_).tolist()}
    Path(report).parent.mkdir(parents=True,exist_ok=True)
    Path(report).write_text(json.dumps(results,indent=2),encoding='utf-8')
    return results
