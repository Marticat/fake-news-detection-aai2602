import tempfile
import unittest
from pathlib import Path


class WorkflowTests(unittest.TestCase):
    def test_demo_run_and_prediction(self):
        from newscheck.workflow import run_project, predict_text
        with tempfile.TemporaryDirectory() as d:
            run=run_project(Path(d)/'raw',Path(d)/'runs',demo=True,isolated=False,documents=False)
            import json,sqlite3
            meta=json.loads((run/'run.json').read_text())
            self.assertEqual(meta['status'],'complete')
            self.assertEqual(meta['dataset_kind'],'synthetic_demo')
            self.assertEqual(len(meta['metrics']),4)
            self.assertEqual(meta['metrics'][0]['dataset_kind'],'synthetic_demo')
            self.assertEqual(meta['metrics'][0]['run_id'],run.name)
            self.assertEqual(meta['data']['clean_rows'],240)
            con=sqlite3.connect(run/'news.db')
            self.assertEqual(con.execute('SELECT COUNT(*) FROM predictions').fetchone()[0],144)
            con.close()
            result=predict_text(run,'The agency published official figures and records for the public hearing.')
            self.assertIn(result['label'],['real','fake'])
            self.assertGreaterEqual(result['probability_fake'],0)
            self.assertLessEqual(result['probability_fake'],1)
            with self.assertRaises(ValueError):
                predict_text(run,'!!!')

    def test_menu_survives_failed_worker(self):
        from unittest.mock import patch
        import contextlib,io,subprocess
        from main import menu
        with patch('newscheck.workflow.run_project',side_effect=subprocess.CalledProcessError(1,['worker'])), patch('builtins.input',side_effect=['1','data/raw','y','0']), contextlib.redirect_stdout(io.StringIO()) as output:
            menu()
        self.assertIn('Operation failed',output.getvalue())


if __name__=='__main__':
    unittest.main()
