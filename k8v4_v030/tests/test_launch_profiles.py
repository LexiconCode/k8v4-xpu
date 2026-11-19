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

    def spec_and_graph(self, args):
        spec = json.loads(next(arg for arg in args if arg.startswith('--speculative-config=')).split('=', 1)[1])
        graph = json.loads(next(arg for arg in args if arg.startswith('--compilation-config=')).split('=', 1)[1])
        return spec, graph['cudagraph_capture_sizes']

    def test_shared_mtp_depth_is_off_by_default(self):
        for options in ({}, {'K8V4_TP': '1'}, {'K8V4_TP': '1', 'K8V4_MAX_SEQS': '1', 'K8V4_MTP_SHARED_DEPTH': '3'}):
            with self.subTest(options=options):
                result, args = self.command(**options)
                self.assertEqual(result.returncode, 0, result.stderr)
                spec, _ = self.spec_and_graph(args)
                self.assertEqual(spec, {'method': 'mtp', 'num_speculative_tokens': 6})

    def test_shared_mtp_depth_sets_batch_schedule_and_captures(self):
        result, args = self.command(K8V4_TP='1', K8V4_MAX_SEQS='4', K8V4_MTP_SHARED_DEPTH='3')
        self.assertEqual(result.returncode, 0, result.stderr)
        spec, sizes = self.spec_and_graph(args)
        self.assertEqual(spec['num_speculative_tokens'], 6)
        self.assertEqual(spec['num_speculative_tokens_per_batch_size'], [[1, 1, 6], [2, 4, 3]])
        self.assertEqual(sizes, [7, 8, 12, 14, 16, 21, 28])
        for bad in ('0', '7', 'x'):
            with self.subTest(bad=bad):
                result, args = self.command(K8V4_TP='1', K8V4_MAX_SEQS='4', K8V4_MTP_SHARED_DEPTH=bad)
                self.assertEqual(result.returncode, 2)
    def test_w4a8_layers_is_passed_only_when_set(self):
        result, default = self.command(K8V4_TP='1')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(any('K8V4_W4A8_LAYERS' in arg for arg in default))
        result, args = self.command(K8V4_TP='1', K8V4_W4A8_LAYERS='mlp,attn,gdn')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('K8V4_W4A8_LAYERS=mlp,attn,gdn', args)
        self.assertIn('VLLM_CACHE_ROOT=/root/.cache/vllm/w4a8-tp1-v1-layers-mlp-attn-gdn', args)
        def stable(argv):
            return [a for a in argv if a not in ('-e', 'K8V4_W4A8_LAYERS=mlp,attn,gdn')
                    and not a.startswith(('VLLM_CACHE_ROOT=', '/tmp', '/var/folders')) and ':/model' not in a]
        self.assertEqual(stable(args), stable(default))
        for bad in ('qkv', 'mlp,', 'all,mlp'):
            result, args = self.command(K8V4_TP='1', K8V4_W4A8_LAYERS=bad)
            self.assertEqual(result.returncode, 2, bad)

    def test_invalid_gpu_selections_are_rejected_before_launch(self):
        for options in ({'K8V4_TP': '4'}, {'K8V4_TP': '1', 'K8V4_DEVICES': '0,1'},
                        {'K8V4_TP': '2', 'K8V4_DEVICES': '0'}, {'K8V4_DEVICES': '1,1'},
                        {'K8V4_DEVICES': '0,01'}, {'K8V4_TP': '1', 'K8V4_PREFIX_CACHING': 'yes'}):
            with self.subTest(options=options):
                result, args = self.command(**options)
                self.assertEqual(result.returncode, 2)
                self.assertEqual(args, [])
