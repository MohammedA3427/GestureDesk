import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from gesturedesk.model import train,Classifier,load_dataset
from tests.test_engine import hand

class ModelTests(unittest.TestCase):
    def dataset(self,path):
        rng=np.random.default_rng(7)
        rows=[]
        for session in ('train-a','train-b','validation','test'):
            for label,p in [('OPEN_PALM',hand()),('FIST',hand((False,)*4,False)),
                            ('UNKNOWN',hand((False,True,False,True),True))]:
                for _ in range(15):
                    rows.append(json.dumps({'session':session,'label':label,'landmarks':(p+rng.normal(0,1,p.shape)).tolist()}))
        path.write_text('\n'.join(rows),encoding='utf-8')
    def test_training_export_reload_and_disjoint_sessions(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);data=d/'data.jsonl';self.dataset(data)
            result=train(data,d/'model.npz',d/'report.json',['test'],['validation'])
            self.assertEqual(set(result['splits']['train']['sessions']),{'train-a','train-b'})
            self.assertEqual(result['splits']['test']['sessions'],['test'])
            model=Classifier(d/'model.npz')
            self.assertEqual(model.predict(hand())[0],'OPEN_PALM')
            self.assertEqual(model.predict(hand((False,)*4,False))[0],'FIST')
            self.assertEqual(model.predict(hand((False,True,False,True)))[0],'UNKNOWN')
            self.assertEqual(model.predict(np.zeros((21,3)))[0],'UNKNOWN')
    def test_leakage_and_missing_session_fail(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);data=d/'data.jsonl';self.dataset(data)
            for test,val in [(['test'],['test']),(['missing'],['validation'])]:
                with self.assertRaises(ValueError):train(data,d/'m.npz',d/'r.json',test,val)
    def test_missing_class_in_holdout_fails(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);data=d/'data.jsonl';self.dataset(data)
            rows=[r for r in data.read_text().splitlines() if not (json.loads(r)['session']=='test' and json.loads(r)['label']=='UNKNOWN')]
            data.write_text('\n'.join(rows))
            with self.assertRaises(ValueError):train(data,d/'m.npz',d/'r.json',['test'],['validation'])
    def test_malformed_row_fails(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'data';path.write_text('{}')
            with self.assertRaisesRegex(ValueError,'row 1'):load_dataset(path)

if __name__=='__main__':unittest.main()
