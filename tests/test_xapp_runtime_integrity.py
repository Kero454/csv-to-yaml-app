"""
Regression test: the generated O-RAN xApp runtime (build_xapp.py template)
must verify the .pkl against the catalogued sha256 carried in the descriptor's
controls.model_identity, and restore embedded trained weights.
"""
import hashlib
import importlib.util
import json
import os
import pickle
import subprocess
import sys
import tempfile
import threading
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BUILDER = os.path.normpath(os.path.join(ROOT, '..', 'oran-ric', 'nonrtric', 'xapp-builder', 'build_xapp.py'))


def _make_pkl(path, weights=b'\x00\x01trained'):
    art = {
        'artifact_schema': 'csv-to-yaml-pkl/v2',
        'model_uid': 'uid-123',
        'model_name': 'smoke-lstm',
        'model_type': 'lstm',
        'version': '1.0.0',
        'hyperparameters': {'hidden': 8},
        'weights': {'file': 'ckpt/best.ckpt', 'size': len(weights),
                    'sha256': hashlib.sha256(weights).hexdigest()},
        'weights_blob': weights,
    }
    with open(path, 'wb') as f:
        pickle.dump(art, f)
    return hashlib.sha256(open(path, 'rb').read()).hexdigest()


def _load_runtime(build_dir, env):
    for k, v in env.items():
        os.environ[k] = v
    spec = importlib.util.spec_from_file_location('xapp_main_%d' % id(env),
                                                  os.path.join(build_dir, 'xapp_main.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@unittest.skipUnless(os.path.exists(BUILDER), 'oran-ric repo not checked out next to this repo')
class XappRuntimeIntegrityTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        subprocess.run([sys.executable, BUILDER, '--name', 'smoke-lstm', '--version', '1.0.0',
                        '--type', 'lstm', '--output-dir', self.tmp, '--no-build', '--demo'],
                       check=True, capture_output=True)
        self.pkl = os.path.join(self.tmp, 'test_model.pkl')
        self.sha = _make_pkl(self.pkl)
        self.cfg_dir = os.path.join(self.tmp, 'cfg')
        os.makedirs(self.cfg_dir)

    def _write_descriptor(self, sha):
        with open(os.path.join(self.cfg_dir, 'config-file.json'), 'w') as f:
            json.dump({'controls': {'model_type': 'lstm',
                                    'model_identity': {'model_id': 'cat-1', 'model_uid': 'uid-123',
                                                       'pkl_sha256': sha}}}, f)

    def _runner(self, port):
        mod = _load_runtime(self.tmp, {
            'MODEL_PATH': self.pkl,
            'WEIGHTS_DIR': os.path.join(self.tmp, 'w'),
            'XAPP_DESCRIPTOR_PATH': self.cfg_dir,
            'HEALTH_PORT': str(port),
            'RMR_PORT': '0',
        })
        r = mod.MLModelRunner()
        return r

    def test_hash_matches_catalog(self):
        self._write_descriptor(self.sha)
        r = self._runner(18081)
        self.assertTrue(r.verify_model_integrity())
        self.assertTrue(r.integrity['checked'])
        self.assertTrue(r.integrity['match'])
        self.assertTrue(r.load_model())
        self.assertTrue(os.path.exists(r.weights_path))
        self.assertEqual(open(r.weights_path, 'rb').read(), b'\x00\x01trained')
        h = r.get_health_status()
        self.assertEqual(h['model_id'], 'cat-1')
        self.assertTrue(h['weights_loaded'])
        self.assertTrue(h['integrity']['match'])
        r.running = False

    def test_hash_mismatch_is_reported(self):
        self._write_descriptor('0' * 64)
        r = self._runner(18082)
        self.assertTrue(r.verify_model_integrity())  # non-strict: warn only
        self.assertFalse(r.integrity['match'])
        os.environ['STRICT_MODEL_INTEGRITY'] = 'true'
        try:
            r2 = self._runner(18083)
            self.assertFalse(r2.verify_model_integrity())
            r2.running = False
        finally:
            os.environ.pop('STRICT_MODEL_INTEGRITY', None)
        r.running = False


if __name__ == '__main__':
    unittest.main()
