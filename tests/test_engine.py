import unittest
import numpy as np
from gesturedesk.core import normalize,features,rule_gesture,Debouncer,SwipeDetector,Engine,LandmarkFilter

def hand(extended=(True,True,True,True),thumb=True):
    p=np.zeros((21,3),float)
    p[0]=[320,420,0]
    for base,x,length,up in zip((5,9,13,17),(275,310,345,375),(180,210,180,140),extended):
        p[base]=[x,310,0]
        if up:
            p[base+1]=[x,310-length*.45,0]
            p[base+2]=[x,310-length*.75,0]
            p[base+3]=[x,310-length,0]
        else:
            p[base+1]=[x,270,0]
            p[base+2]=[x+10,310,-15]
            p[base+3]=[x+10,340,-20]
    p[1]=[280,380,0];p[2]=[250,350,0];p[3]=[225,320,0];p[4]=[200,290,0]
    if not thumb:
        p[3]=[270,335,0];p[4]=[300,320,-15]
    return p

class GeometryTests(unittest.TestCase):
    def test_translation_scale_rotation_and_mirror(self):
        p=hand();expected=features(p)
        theta=.6;rot=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
        q=p.copy();q[:,:2]=q[:,:2]@rot.T
        np.testing.assert_allclose(features(q*2.5+[30,60,10]),expected,atol=1e-6)
        p[:,0]*=-1
        np.testing.assert_allclose(features(p),expected,atol=1e-6)
    def test_bad_input(self):
        for p in (np.zeros((21,3)),np.full((21,3),np.nan),np.zeros((20,3))):
            with self.assertRaises(ValueError):features(p)
        self.assertEqual(rule_gesture(np.zeros((21,3)))[0],'UNKNOWN')
    def test_baseline_fixtures(self):
        for label,p in [('OPEN_PALM',hand()),('FIST',hand((False,)*4,False)),
                        ('POINT',hand((True,False,False,False),False)),
                        ('PEACE',hand((True,True,False,False),False))]:
            self.assertEqual(rule_gesture(p)[0],label)
    def test_thumb_pinch_and_unsupported_pose(self):
        p=hand((False,)*4);p[2]=[245,350,0];p[3]=[245,290,0];p[4]=[245,230,0]
        self.assertEqual(rule_gesture(p)[0],'THUMBS_UP')
        p=hand();p[4]=p[8]+[12,0,0]
        self.assertEqual(rule_gesture(p)[0],'PINCH')
        self.assertEqual(rule_gesture(hand((True,False,True,False)))[0],'UNKNOWN')

class TemporalTests(unittest.TestCase):
    def test_dwell_jitter_and_one_event_per_hold(self):
        d=Debouncer()
        self.assertIsNone(d.update('PEACE',0).event)
        self.assertIsNone(d.update('UNKNOWN',.1).event)
        self.assertIsNone(d.update('PEACE',.15).event)
        self.assertEqual(d.update('PEACE',.4).event,'PEACE')
        self.assertIsNone(d.update('PEACE',1).event)
        d.update('UNKNOWN',1.1);d.update('PEACE',1.15)
        self.assertIsNone(d.update('PEACE',1.5).event)
        d.update('UNKNOWN',2);d.update('UNKNOWN',2.2);d.update('PEACE',2.3)
        self.assertEqual(d.update('PEACE',2.6).event,'PEACE')
    def test_none_immediate_reset(self):
        d=Debouncer();d.update('PINCH',0);d.update('PINCH',.3)
        self.assertEqual(d.update('NONE',.31).stable,'NONE')
        self.assertIsNone(d.update('PINCH',.32).event)
    def test_swipe_direction_cooldown_and_vertical_rejection(self):
        for sign,label in [(1,'SWIPE_RIGHT'),(-1,'SWIPE_LEFT')]:
            s=SwipeDetector();events=[]
            for i in range(10):events.append(s.update((.5+sign*i*.04,.4),'OPEN_PALM',i*.04))
            self.assertEqual([e for e in events if e],[label])
        s=SwipeDetector()
        self.assertTrue(all(s.update((.2+i*.04,.2+i*.04),'OPEN_PALM',i*.04) is None for i in range(10)))
    def test_swipe_unknown_clears_history(self):
        s=SwipeDetector();s.update((.2,.4),'OPEN_PALM',0)
        s.update((.3,.4),'UNKNOWN',.1)
        self.assertEqual(len(s.history),0)
    def test_tracking_loss_and_identity_reset(self):
        e=Engine();p=hand()
        e.update(p,0);e.update(p,.24)
        self.assertEqual(e.update(None,.25)[1].stable,'NONE')
        self.assertIsNone(e.update(p,.26)[1].event)
        self.assertIsNone(e.update(p,.52,identity='Left')[1].event)
    def test_filter_reset_and_step(self):
        f=LandmarkFilter();p=hand();f.update(p,0)
        q=f.update(p+10,.03)
        self.assertTrue(np.all(q>p));self.assertTrue(np.all(q<p+10))
        np.testing.assert_allclose(f.update(p+20,1),p+20)

if __name__=='__main__':unittest.main()
