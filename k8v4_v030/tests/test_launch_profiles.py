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

    def test_w4a8_kernel_and_group_need_the_library(self):
        result, args = self.command(K8V4_TP='1', K8V4_W4A8_KERNEL='onednn')
        self.assertEqual(result.returncode, 2)
        result, args = self.command(K8V4_TP='1', K8V4_W4A8_AGROUP='64')
        self.assertEqual(result.returncode, 2)
        with tempfile.NamedTemporaryFile(suffix='.so') as lib:
            result, args = self.command(K8V4_TP='1', K8V4_W4A8_KERNEL='onednn', K8V4_W4A8_LIB=lib.name,
                                        K8V4_W4A8_AGROUP='64')
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('%s:/opt/k8v4/libk8v4_w4a8.so:ro' % lib.name, args)
            self.assertIn('K8V4_W4A8_KERNEL=onednn', args)
            self.assertIn('K8V4_W4A8_AGROUP=64', args)
            self.assertIn('VLLM_CACHE_ROOT=/root/.cache/vllm/w4a8-tp1-v1-kernel-onednn-agroup-64', args)
            result, args = self.command(K8V4_TP='1', K8V4_W4A8_KERNEL='b70', K8V4_W4A8_LIB=lib.name)
            self.assertEqual(result.returncode, 2)

    def test_w4a8_fused_needs_onednn_and_group(self):
        with tempfile.NamedTemporaryFile(suffix='.so') as lib:
            base = dict(K8V4_TP='1', K8V4_W4A8_KERNEL='onednn', K8V4_W4A8_LIB=lib.name)
            result, args = self.command(K8V4_W4A8_FUSED='silu', **base)
            self.assertEqual(result.returncode, 2)
            result, args = self.command(K8V4_W4A8_FUSED='gelu', K8V4_W4A8_AGROUP='64', **base)
            self.assertEqual(result.returncode, 2)
            result, args = self.command(K8V4_W4A8_FUSED='silu', K8V4_W4A8_AGROUP='64', **base)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('K8V4_W4A8_FUSED=silu', args)
            self.assertIn('VLLM_CACHE_ROOT=/root/.cache/vllm/w4a8-tp1-v1-kernel-onednn-agroup-64-fused-silu', args)
    def test_ncontig_is_passed_only_when_set(self):
        result, default = self.command(K8V4_TP='1')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(any('K8V4_NCONTIG_RE' in arg for arg in default))
        result, args = self.command(K8V4_TP='1', K8V4_NCONTIG_RE='in_proj_ba')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('K8V4_NCONTIG_RE=in_proj_ba', args)
        self.assertIn('VLLM_CACHE_ROOT=/root/.cache/vllm/w4a8-tp1-v1-ncontig', args)
    def test_embed_host_is_passed_only_when_set(self):
        for options in ({}, {'K8V4_TP': '1'}):
            with self.subTest(options=options):
                result, args = self.command(**options)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertFalse(any(arg.startswith('K8V4_EMBED_HOST') for arg in args))
        for value in ('0', '1'):
            result, args = self.command(K8V4_TP='1', K8V4_EMBED_HOST=value)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('K8V4_EMBED_HOST=%s' % value, args)
        for bad in ('2', 'yes'):
            with self.subTest(bad=bad):
                result, args = self.command(K8V4_TP='1', K8V4_EMBED_HOST=bad)
                self.assertEqual(result.returncode, 2)

    def test_draft_skip_shared_is_passed_only_when_set(self):
        for options in ({}, {'K8V4_TP': '1'}):
            with self.subTest(options=options):
                result, args = self.command(**options)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertFalse(any(arg.startswith('K8V4_DRAFT_SKIP_SHARED') for arg in args))
        for value in ('0', '1'):
            result, args = self.command(K8V4_TP='1', K8V4_DRAFT_SKIP_SHARED=value)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('K8V4_DRAFT_SKIP_SHARED=%s' % value, args)
        for bad in ('2', 'yes'):
            with self.subTest(bad=bad):
                result, args = self.command(K8V4_TP='1', K8V4_DRAFT_SKIP_SHARED=bad)
                self.assertEqual(result.returncode, 2)
    def test_gdn_heads_mount_is_opt_in(self):
        lib_target = '/opt/venv/lib/python3.12/site-packages/vllm_xpu_kernels/libgdn_attn_kernels_xe_2.so'
        def norm(args):  # each call gets its own temporary model and cache directories
            return [a for a in args if not a.endswith((':/model:ro', ':/root/.cache/vllm'))]
        _, default = self.command(K8V4_TP='1')
        _, off = self.command(K8V4_TP='1', K8V4_GDN_HEADS='0')
        default, off = norm(default), norm(off)
        self.assertEqual(default, off)
        self.assertFalse(any(lib_target in arg for arg in default))
        with tempfile.TemporaryDirectory() as temp:
            lib = Path(temp) / 'libgdn_attn_kernels_xe_2.so'
            lib.write_bytes(b'')
            result, args = self.command(K8V4_TP='1', K8V4_GDN_HEADS='1', K8V4_GDN_HEADS_LIB=str(lib))
            self.assertEqual(result.returncode, 0, result.stderr)
            mount = args.index('%s:%s:ro' % (lib, lib_target))
            self.assertEqual(args[mount - 1], '-v')
            self.assertEqual(norm([a for i, a in enumerate(args) if i not in (mount - 1, mount)]), default)
            for options in ({'K8V4_GDN_HEADS': 'yes'},
                            {'K8V4_GDN_HEADS': '1', 'K8V4_GDN_HEADS_LIB': str(Path(temp) / 'missing.so')}):
                with self.subTest(options=options):
                    result, args = self.command(K8V4_TP='1', **options)
                    self.assertEqual(result.returncode, 2)
                    self.assertEqual(args, [])
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
    def test_mamba_ssm_dtype_is_passed_only_when_set(self):
        _, default = self.command(K8V4_TP='1')
        self.assertFalse([arg for arg in default if arg.startswith('--mamba-ssm-cache-dtype')])
        for dtype in ('float16', 'bfloat16', 'float32'):
            with self.subTest(dtype=dtype):
                result, args = self.command(K8V4_TP='1', K8V4_MAMBA_SSM_DTYPE=dtype)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn('--mamba-ssm-cache-dtype=%s' % dtype, args)
                self.assertEqual(len(args), len(default) + 1)
        for bad in ('fp16', 'half', 'auto', 'float16 '):
            with self.subTest(bad=bad):
                result, args = self.command(K8V4_TP='1', K8V4_MAMBA_SSM_DTYPE=bad)
                self.assertEqual(result.returncode, 2)
                self.assertIn(b'K8V4_MAMBA_SSM_DTYPE', result.stderr)
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

    def test_onednn_deterministic_is_passed_only_when_set(self):
        result, default = self.command(K8V4_TP='1')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(any('K8V4_ONEDNN_DETERMINISTIC' in arg for arg in default))
        result, args = self.command(K8V4_TP='1', K8V4_ONEDNN_DETERMINISTIC='1')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('K8V4_ONEDNN_DETERMINISTIC=1', args)
        self.assertIn('VLLM_CACHE_ROOT=/root/.cache/vllm/w4a8-tp1-v1', args)
        i = args.index('K8V4_ONEDNN_DETERMINISTIC=1')
        self.assertEqual(args[i - 1], '-e')
        model = lambda xs: [x.split('/', 3)[-1] if x.startswith('/tmp/') else x for x in xs]  # temp dirs differ per call
        self.assertEqual(model(args[:i - 1] + args[i + 1:]), model(default))
        result, _ = self.command(K8V4_TP='1', K8V4_ONEDNN_DETERMINISTIC='yes')
        self.assertEqual(result.returncode, 2)
        self.assertIn(b'K8V4_ONEDNN_DETERMINISTIC must be 0 or 1', result.stderr)

    def test_s1_fast_is_passed_only_when_set(self):
        result, args = self.command(K8V4_TP='1')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse([a for a in args if 'XE2_KV_S1_FAST' in a])
        for value in ('0', '1'):
            with self.subTest(value=value):
                result, args = self.command(K8V4_TP='1', XE2_KV_S1_SPLITS='32', XE2_KV_S1_FAST=value)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn('XE2_KV_S1_FAST=' + value, args)
        for bad in ('2', 'yes', ' 1'):
            with self.subTest(bad=bad):
                result, args = self.command(K8V4_TP='1', XE2_KV_S1_FAST=bad)
                self.assertEqual(result.returncode, 2)
                self.assertEqual(args, [])

    def test_batch_reqs_is_passed_only_when_set(self):
        result, args = self.command(K8V4_TP='1')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse([a for a in args if 'XE2_KV_BATCH_REQS' in a])
        for value in ('0', '1'):
            with self.subTest(value=value):
                result, args = self.command(K8V4_TP='1', XE2_KV_S1_SPLITS='32', XE2_KV_BATCH_REQS=value)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn('XE2_KV_BATCH_REQS=' + value, args)
        for bad in ('2', 'on'):
            with self.subTest(bad=bad):
                result, args = self.command(K8V4_TP='1', XE2_KV_BATCH_REQS=bad)
                self.assertEqual(result.returncode, 2)
                self.assertEqual(args, [])

    def test_draft_fused_decode_is_passed_only_when_set(self):
        _, default = self.command(K8V4_TP='1')
        self.assertFalse([arg for arg in default if 'K8V4_DRAFT_FUSED_DECODE' in arg])
        for value in ('0', '1'):
            with self.subTest(value=value):
                result, args = self.command(K8V4_TP='1', K8V4_DRAFT_FUSED_DECODE=value)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn('K8V4_DRAFT_FUSED_DECODE=%s' % value, args)
                self.assertEqual(len(args), len(default) + 2)
        for bad in ('2', 'yes', '1 '):
            with self.subTest(bad=bad):
                result, args = self.command(K8V4_TP='1', K8V4_DRAFT_FUSED_DECODE=bad)
                self.assertEqual(result.returncode, 2)
                self.assertIn(b'K8V4_DRAFT_FUSED_DECODE', result.stderr)

    def test_dtype_is_bfloat16_unless_set(self):
        _, default = self.command(K8V4_TP='1')
        self.assertIn('--dtype=bfloat16', default)
        for dtype in ('bfloat16', 'float16'):
            with self.subTest(dtype=dtype):
                result, args = self.command(K8V4_TP='1', K8V4_DTYPE=dtype)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn('--dtype=%s' % dtype, args)
                self.assertEqual(len(args), len(default))
        for bad in ('fp16', 'half', 'float32', 'auto'):
            with self.subTest(bad=bad):
                result, args = self.command(K8V4_TP='1', K8V4_DTYPE=bad)
                self.assertEqual(result.returncode, 2)
                self.assertIn(b'K8V4_DTYPE', result.stderr)

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
