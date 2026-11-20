"""Exercise the real Bash argv builder without touching Docker or GPUs."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASH = os.environ.get('K8V4_BASH') or shutil.which('bash')
if os.name == 'nt' and Path('C:/Program Files/Git/bin/bash.exe').exists():
    BASH = 'C:/Program Files/Git/bin/bash.exe'


@unittest.skipUnless(BASH, 'Bash is needed to exercise launch.sh')
class LaunchProfileTest(unittest.TestCase):
    def command(self, **overrides):
        with tempfile.TemporaryDirectory() as temp:
            env = {k: v for k, v in os.environ.items() if not k.startswith(('K8V4_', 'XE2_KV_', 'MTP_PATCH_'))
                   and k not in ('IMAGE', 'NAME', 'PORT', 'BIND_HOST', 'MODEL_DIR', 'CHAT_TEMPLATE', 'VLLM_CACHE_ROOT')}
            cache = Path(temp) / 'cache'
            env.update(K8V4_DRY_RUN='1', K8V4_CACHE=str(cache), MODEL_DIR=temp, CHAT_TEMPLATE=str(ROOT / 'templates/chat_template.jinja'))
            env.update(overrides)
            result = subprocess.run([BASH, str(ROOT / 'k8v4_v030/launch.sh')], env=env, capture_output=True)
            if result.returncode:
                return result, []
            self.assertFalse(cache.exists())
            return result, result.stdout.decode().rstrip('\0').split('\0')

    def test_tp2_retains_the_measured_settings(self):
        result, args = self.command()
        self.assertEqual(result.returncode, 0, result.stderr)
        for arg in ('--tensor-parallel-size=2', '--max-num-seqs=4', '--max-num-batched-tokens=4224',
                    '--gpu-memory-utilization=0.95', '--enable-prefix-caching', 'ZE_AFFINITY_MASK=0,1',
                    'CCL_SYCL_ALLREDUCE_LL=twoshots', '127.0.0.1:8200:8000', 'vllm-openai-xpu:v0.34.0-k8v4-unified'):
            self.assertIn(arg, args)
        graph = next(arg for arg in args if arg.startswith('--compilation-config='))
        self.assertEqual(json.loads(graph.split('=', 1)[1])['cudagraph_capture_sizes'], [7, 14, 21, 28])
        self.assertNotIn('--mamba-cache-mode=none', args)

    def test_tp1_selects_one_gpu_with_separate_cache_and_no_collectives(self):
        result, args = self.command(K8V4_TP='1', K8V4_DEVICES='1')
        self.assertEqual(result.returncode, 0, result.stderr)
        for arg in ('--tensor-parallel-size=1', '--max-num-seqs=1', '--gpu-memory-utilization=0.92',
                    '--no-enable-prefix-caching', '--mamba-cache-mode=none', 'ZE_AFFINITY_MASK=1',
                    'NEOReadDebugKeys=1', 'ForceZeDeviceCanAccessPerReturnValue=0',
                    '127.0.0.1:8201:8000', 'vllm-openai-xpu:v0.34.0-k8v4-unified'):
            self.assertIn(arg, args)
        self.assertFalse(any(arg.startswith('CCL_') for arg in args))
        self.assertIn('VLLM_CACHE_ROOT=/root/.cache/vllm/w4a8-tp1-v1', args)
        graph = next(arg for arg in args if arg.startswith('--compilation-config='))
        self.assertEqual(json.loads(graph.split('=', 1)[1])['cudagraph_capture_sizes'], [7])

    def test_explicit_tp1_overrides_apply_to_command(self):
        result, args = self.command(K8V4_TP='1', K8V4_PREFIX_CACHING='1', K8V4_MAX_SEQS='2', K8V4_MAX_MODEL_LEN='133056')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--enable-prefix-caching', args)
        self.assertNotIn('--mamba-cache-mode=none', args)
        self.assertIn('--max-num-seqs=2', args)
        self.assertIn('--max-model-len=133056', args)

    def test_gdn_recoverssm_is_opt_in(self):
        _, default = self.command(K8V4_TP='1')
        self.assertNotIn('--use-replayssm', default)
        self.assertFalse([arg for arg in default if 'RSSM' in arg or 'RECOVERSSM' in arg])
        result, args = self.command(K8V4_TP='1', K8V4_GDN_RECOVERSSM='0')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(args), len(default))
        self.assertNotIn("--use-replayssm", args)
        for bad in ('2', 'yes'):
            result, args = self.command(K8V4_TP='1', K8V4_GDN_RECOVERSSM=bad)
            self.assertEqual(result.returncode, 2)
        result, args = self.command(K8V4_TP='1', K8V4_GDN_RECOVERSSM='1', K8V4_RSSM_LIB='/nonexistent/lib.so')
        self.assertEqual(result.returncode, 2)
        self.assertIn(b'compile_rssm.sh', result.stderr)
        with tempfile.NamedTemporaryFile(suffix='.so') as lib:
            result, args = self.command(K8V4_TP='1', K8V4_GDN_RECOVERSSM='1', K8V4_RSSM_LIB=lib.name)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('--use-replayssm', args)
            self.assertIn('%s:/opt/k8v4/libk8v4_rssm.so:ro' % os.path.realpath(lib.name), args)
            self.assertIn('K8V4_GDN_RECOVERSSM=1', args)
            self.assertIn('K8V4_RSSM_LIB=/opt/k8v4/libk8v4_rssm.so', args)
            self.assertEqual(len(args), len(default) + 7)

    def test_rssm_verify_and_deferred_commit_are_opt_in(self):
        _, default = self.command(K8V4_TP='1')
        self.assertFalse([arg for arg in default if 'K8V4_RSSM_VERIFY' in arg or 'K8V4_RSSM_DEFER' in arg])
        result, args = self.command(K8V4_TP='1', K8V4_RSSM_DEFER_COMMIT='0')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(args), len(default))
        for bad in ({'K8V4_RSSM_VERIFY': 'fast'}, {'K8V4_RSSM_DEFER_COMMIT': 'yes'}, {'K8V4_RSSM_VERIFY': 'slm'},
                    {'K8V4_RSSM_DEFER_COMMIT': '1'}):
            result, _ = self.command(K8V4_TP='1', **bad)
            self.assertEqual(result.returncode, 2, bad)
        with tempfile.NamedTemporaryFile(suffix='.so') as lib:
            _, base = self.command(K8V4_TP='1', K8V4_GDN_RECOVERSSM='1', K8V4_RSSM_LIB=lib.name)
            result, args = self.command(K8V4_TP='1', K8V4_GDN_RECOVERSSM='1', K8V4_RSSM_LIB=lib.name,
                                        K8V4_RSSM_VERIFY='slm', K8V4_RSSM_DEFER_COMMIT='1')
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('K8V4_RSSM_VERIFY=slm', args)
            self.assertIn('K8V4_RSSM_DEFER_COMMIT=1', args)
            self.assertEqual(len(args), len(base) + 4)

    def test_invalid_gpu_selections_are_rejected_before_launch(self):
        for options in ({'K8V4_TP': '4'}, {'K8V4_TP': '1', 'K8V4_DEVICES': '0,1'},
                        {'K8V4_TP': '2', 'K8V4_DEVICES': '0'}, {'K8V4_DEVICES': '1,1'},
                        {'K8V4_DEVICES': '0,01'}, {'K8V4_TP': '1', 'K8V4_PREFIX_CACHING': 'yes'}):
            with self.subTest(options=options):
                result, args = self.command(**options)
                self.assertEqual(result.returncode, 2)
                self.assertEqual(args, [])
