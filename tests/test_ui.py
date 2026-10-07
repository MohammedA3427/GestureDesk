import unittest
import numpy as np
from pathlib import Path
from gesturedesk.ui import Dashboard
from gesturedesk.core import State,Engine
from tests.test_engine import hand

ROOT=Path(__file__).resolve().parent.parent

class DashboardTests(unittest.TestCase):
    def test_portfolio_wrap_pause_and_toggle(self):
        ui=Dashboard(ROOT/'portfolio.json')
        ui.action('SWIPE_LEFT');self.assertEqual(ui.card,len(ui.portfolio['cards'])-1)
        ui.action('SWIPE_RIGHT');self.assertEqual(ui.card,0)
        ui.action('PEACE');self.assertTrue(ui.drawing)
        ui.paused=True;ui.action('SWIPE_RIGHT');ui.action('PEACE')
        self.assertEqual(ui.card,0);self.assertTrue(ui.drawing)
    def test_drawing_stops_on_uncertainty_and_hand_loss(self):
        ui=Dashboard(ROOT/'portfolio.json');ui.drawing=True
        p=hand();state=State('PINCH','PINCH')
        ui.pen(p,state,(640,480));ui.pen(p+10,state,(640,480))
        self.assertGreater(np.count_nonzero(ui.canvas),0)
        ui.pen(p,State('UNKNOWN','PINCH'),(640,480));self.assertIsNone(ui.last_pen)
        ui.pen(p,state,(640,480));ui.pen(None,state,(640,480));self.assertIsNone(ui.last_pen)
    def test_dashboard_render_has_expected_dimensions(self):
        ui=Dashboard(ROOT/'portfolio.json')
        for drawing in (False,True):
            ui.drawing=drawing
            result=ui.render(np.zeros((480,640,3),np.uint8),hand(),State('OPEN_PALM','OPEN_PALM'),.85,'Right',(30,15,20))
            self.assertEqual(result.shape,(800,1280,3))
    def test_engine_to_dashboard_one_toggle_per_hold(self):
        ui=Dashboard(ROOT/'portfolio.json');engine=Engine()
        peace=hand((True,True,False,False),False)
        for i in range(30):
            _,state,_=engine.update(peace,i/30)
            ui.action(state.event)
        self.assertTrue(ui.drawing)
        engine.update(None,1)
        for i in range(30):
            _,state,_=engine.update(peace,1.1+i/30)
            ui.action(state.event)
        self.assertFalse(ui.drawing)




class DrawingPinchIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.ui=Dashboard(ROOT/'portfolio.json')
        class Custom:
            label='UNKNOWN'
            def predict(self, points): return self.label,.9
        self.model=Custom()
        self.engine=Engine(self.model)
    def step(self,p,t,identity='Right'):
        filtered,state,score=self.engine.update(p,t,identity=identity)
        self.ui.action(state.event)
        self.ui.pen(filtered,state,(640,480),pinch_input=(p,t,identity))
        return state
    def pinch(self,distance=.1):
        p=hand((True,False,False,False),False)
        p[4]=p[8]+[distance*np.linalg.norm(p[9]-p[0]),0,0]
        return p
    def test_relaxed_fingers_hysteresis_and_release(self):
        self.ui.drawing=True
        p=self.pinch()
        self.assertEqual(self.step(p,0).raw,'UNKNOWN')
        self.assertIsNone(self.ui.last_pen)
        self.step(p,.07);self.assertIsNotNone(self.ui.last_pen)
        self.step(self.pinch(.27)+[10,0,0],.1)
        self.assertGreater(np.count_nonzero(self.ui.canvas),0)
        self.step(self.pinch(.33),.13);self.assertIsNone(self.ui.last_pen)
        count=np.count_nonzero(self.ui.canvas)
        self.step(self.pinch(.27)+[20,0,0],.16)
        self.assertIsNone(self.ui.last_pen)
        self.assertEqual(np.count_nonzero(self.ui.canvas),count)
    def test_outside_drawing_pause_loss_and_identity_reset(self):
        p=self.pinch()
        for t in (0,.07,.1):self.step(p,t)
        self.assertFalse(self.ui.drawing_pinch.active)
        self.assertEqual(np.count_nonzero(self.ui.canvas),0)
        self.ui.drawing=True
        self.step(p,.13);self.step(p,.2)
        self.assertIsNotNone(self.ui.last_pen)
        self.step(p,.23,'Left');self.assertIsNone(self.ui.last_pen)
        self.step(p,.3,'Left');self.assertIsNotNone(self.ui.last_pen)
        self.step(None,.33);self.assertIsNone(self.ui.last_pen)
        self.step(p,.36);self.assertIsNone(self.ui.last_pen)
        self.step(p,.43);self.ui.paused=True
        self.step(p,.46);self.assertIsNone(self.ui.last_pen)
        self.ui.paused=False;self.step(p,.49);self.assertIsNone(self.ui.last_pen)
        self.step(p,.8);self.assertIsNone(self.ui.last_pen)
    def test_peace_toggle_once_and_swipes_unchanged(self):
        self.model.label='PEACE'
        p=hand((True,True,False,False),False)
        events=[self.step(p,i/30).event for i in range(30)]
        self.assertEqual([e for e in events if e],['PEACE'])
        self.assertTrue(self.ui.drawing)
        self.assertEqual(np.count_nonzero(self.ui.canvas),0)
        self.step(None,1)
        for i in range(30):self.step(p,1.1+i/30)
        self.assertFalse(self.ui.drawing)
        for start,drawing in ((3.,False),(5.,True)):
            self.ui.drawing=drawing
            self.step(None,start)
            self.model.label='OPEN_PALM'
            card=self.ui.card
            events=[self.step(hand()+[i*.04*640,0,0],start+.1+i*.04).event for i in range(10)]
            self.assertEqual([e for e in events if e and e.startswith('SWIPE')],['SWIPE_RIGHT'])
            self.assertEqual(self.ui.card,(card+1)%len(self.ui.portfolio['cards']))
            self.assertIsNone(self.ui.last_pen)
    def test_original_classifier_gate_remains_available(self):
        self.ui.drawing=True
        p=self.pinch()
        self.ui.pen(p,State('UNKNOWN','UNKNOWN'),(640,480))
        self.assertIsNone(self.ui.last_pen)
        self.ui.pen(p,State('PINCH','PINCH'),(640,480))
        self.assertIsNotNone(self.ui.last_pen)

if __name__=='__main__':unittest.main()
