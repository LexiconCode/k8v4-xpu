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


def _no_temp(args):
    """argv without the per-call temporary model and cache paths."""
    return [x for x in args if '/model:ro' not in x and ':/root/.cache/vllm' not in x]


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

    def test_k8v4_draft_lastrow_is_passed_only_when_set(self):
        _, default = self.command(K8V4_TP='1')
        self.assertFalse([arg for arg in default if arg.startswith('K8V4_DRAFT_LASTROW=')])
        result, args = self.command(K8V4_TP='1', K8V4_DRAFT_LASTROW='1')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('K8V4_DRAFT_LASTROW=1', args)
        self.assertEqual(len(args), len(default) + 2)

    def test_draft_vocab_is_mounted_only_when_set(self):
        _, default = self.command(K8V4_TP='1')
        self.assertFalse([arg for arg in default if 'draft_vocab' in arg])
        with tempfile.NamedTemporaryFile('w', suffix='.json') as ids:
            ids.write('[0, 1, 2]')
            ids.flush()
            result, args = self.command(K8V4_TP='1', K8V4_DRAFT_VOCAB=ids.name)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('%s:/opt/k8v4/draft_vocab.json:ro' % os.path.realpath(ids.name), args)
            self.assertIn('K8V4_DRAFT_VOCAB=/opt/k8v4/draft_vocab.json', args)
            self.assertEqual(len(args), len(default) + 4)
        result, _ = self.command(K8V4_TP='1', K8V4_DRAFT_VOCAB='/nonexistent/ids.json')
        self.assertEqual(result.returncode, 2)

    def test_k8v4_verify_chunk_is_passed_only_when_set(self):
        _, default = self.command(K8V4_TP='1')
        self.assertFalse([arg for arg in default if arg.startswith('K8V4_VERIFY_CHUNK=')])
        result, args = self.command(K8V4_TP='1', K8V4_VERIFY_CHUNK='8')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('K8V4_VERIFY_CHUNK=8', args)
        self.assertEqual(len(args), len(default) + 2)

    def test_deep_draft_profile_only_when_set(self):
        def spec_and_sizes(args):
            spec = json.loads(next(x for x in args if x.startswith('--speculative-config=')).split('=', 1)[1])
            graph = json.loads(next(x for x in args if x.startswith('--compilation-config=')).split('=', 1)[1])
            return spec, graph['cudagraph_capture_sizes']

        _, default = self.command(K8V4_TP='1', K8V4_MAX_SEQS='4')
        self.assertEqual(spec_and_sizes(default), ({'method': 'mtp', 'num_speculative_tokens': 6}, [7, 14, 21, 28]))
        self.assertFalse([x for x in default if x.startswith(('K8V4_DEEP', 'K8V4_VERIFY'))])
        result, args = self.command(K8V4_TP='1', K8V4_MAX_SEQS='4', K8V4_DEEP_DRAFT='12,16')
        self.assertEqual(result.returncode, 0, result.stderr)
        spec, sizes = spec_and_sizes(args)
        self.assertEqual(spec, {'method': 'mtp', 'num_speculative_tokens': 15,
                                'num_speculative_tokens_per_batch_size': [[1, 1, 15], [2, 4, 3]]})
        self.assertEqual(sizes, [7, 8, 12, 16])
        self.assertIn('K8V4_DEEP_DRAFT=12,16', args)
        self.assertIn('K8V4_VERIFY_CHUNK=8', args)
        _, one = self.command(K8V4_TP='1', K8V4_DEEP_DRAFT='12,16')
        self.assertEqual(spec_and_sizes(one)[1], [7, 16])
        for bad in ({'K8V4_DEEP_DRAFT': 'x'}, {'K8V4_DEEP_DRAFT': '12,16', 'K8V4_DEEP_DEPTH': '40'}):
            with self.subTest(bad=bad):
                result, _ = self.command(K8V4_TP='1', **bad)
                self.assertEqual(result.returncode, 2)
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

    def test_deep_all_widens_the_capture_list_only_when_set(self):
        def sizes(args):
            graph = json.loads(next(x for x in args if x.startswith('--compilation-config=')).split('=', 1)[1])
            return graph['cudagraph_capture_sizes']

        _, deep = self.command(K8V4_TP='1', K8V4_MAX_SEQS='4', K8V4_DEEP_DRAFT='12,16')
        self.assertFalse([x for x in deep if x.startswith('K8V4_DEEP_ALL')])
        result, args = self.command(K8V4_TP='1', K8V4_MAX_SEQS='4', K8V4_DEEP_DRAFT='12,16', K8V4_DEEP_ALL='1')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(sizes(args), [7, 8, 12, 16, 32, 48, 64])
        self.assertIn('K8V4_DEEP_ALL=1', args)
        self.assertEqual(len(args), len(deep) + 2)
        _, zero = self.command(K8V4_TP='1', K8V4_MAX_SEQS='4', K8V4_DEEP_DRAFT='12,16', K8V4_DEEP_ALL='0')
        self.assertEqual(_no_temp(zero), _no_temp(deep))
        _, anyone = self.command(K8V4_TP='1', K8V4_MAX_SEQS='4', K8V4_DEEP_DRAFT='12,16', K8V4_DEEP_ALL='any')
        self.assertEqual(sizes(anyone), [7, 8, 12, 16, 32, 48, 64])
        self.assertIn('K8V4_DEEP_ALL=any', anyone)
        for bad in ({'K8V4_DEEP_ALL': '1'}, {'K8V4_DEEP_DRAFT': '12,16', 'K8V4_DEEP_ALL': 'yes'}):
            with self.subTest(bad=bad):
                result, _ = self.command(K8V4_TP='1', K8V4_MAX_SEQS='4', **bad)
                self.assertEqual(result.returncode, 2)

    def test_invalid_gpu_selections_are_rejected_before_launch(self):
        for options in ({'K8V4_TP': '4'}, {'K8V4_TP': '1', 'K8V4_DEVICES': '0,1'},
                        {'K8V4_TP': '2', 'K8V4_DEVICES': '0'}, {'K8V4_DEVICES': '1,1'},
                        {'K8V4_DEVICES': '0,01'}, {'K8V4_TP': '1', 'K8V4_PREFIX_CACHING': 'yes'}):
            with self.subTest(options=options):
                result, args = self.command(**options)
                self.assertEqual(result.returncode, 2)
                self.assertEqual(args, [])
