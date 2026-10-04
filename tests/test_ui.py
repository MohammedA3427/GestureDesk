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
        ui.action('SWIPE_LEFT');self.assertEqual(ui.card,3)
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

if __name__=='__main__':unittest.main()
