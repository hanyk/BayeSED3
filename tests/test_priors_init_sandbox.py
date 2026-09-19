#!/usr/bin/env python3
"""Seams for the two priors_init fixes (no binary).

A — classify_sandbox_round (Z4 literals from live dummy output)
B — union_copy_missing never overwrites
C — priors_init copies missing factory names even when some already exist
D — SEDInference.priors_init raises when dummy generate is not Z4
"""

import os
import shutil
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

from bayesed import SEDInference
from bayesed.core import BayeSEDExecutionError
from bayesed.priors import Priors, classify_sandbox_round, union_copy_missing

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SAMPLE = os.path.join(_REPO_ROOT, 'observation', 'test', '2dal8.iprior')


def test_classify_sandbox_round_continue_and_done():
    template = (
        "Can't open /tmp/foo.iprior and /tmp/foo.iprior.0! "
        "A template of /tmp/foo.iprior.0 has been generated."
    )
    assert classify_sandbox_round(1, template, ['/tmp/foo.iprior.0']) == 'continue'
    done = "0 valid objects loaded from empty.txt!\nNo valid objects in empty.txt!"
    assert classify_sandbox_round(0, done, []) == 'done'
    assert classify_sandbox_round(1, 'mpirun detected non-zero status', []) == 'fail'
    assert classify_sandbox_round(1, template, []) == 'fail'


def test_union_copy_missing_does_not_overwrite():
    tmp = tempfile.mkdtemp(prefix='bayesed_union_')
    try:
        src = os.path.join(tmp, 'src')
        dst = os.path.join(tmp, 'dst')
        os.makedirs(src)
        os.makedirs(dst)
        with open(os.path.join(src, 'a.iprior'), 'w') as handle:
            handle.write('factory\n')
        with open(os.path.join(src, 'b.iprior'), 'w') as handle:
            handle.write('factory-b\n')
        with open(os.path.join(dst, 'a.iprior'), 'w') as handle:
            handle.write('custom\n')
        copied = union_copy_missing(src, dst, ['a.iprior', 'b.iprior'])
        assert copied == ['b.iprior']
        with open(os.path.join(dst, 'a.iprior')) as handle:
            assert handle.read() == 'custom\n'
        with open(os.path.join(dst, 'b.iprior')) as handle:
            assert handle.read() == 'factory-b\n'
    finally:
        shutil.rmtree(tmp)


def test_priors_init_copies_missing_and_keeps_working():
    tmp = tempfile.mkdtemp(prefix='bayesed_priors_init_')
    try:
        input_dir = os.path.join(tmp, 'input')
        outdir = os.path.join(tmp, 'out')
        sandbox = os.path.join(outdir, 'priors_defaults')
        os.makedirs(input_dir)
        os.makedirs(sandbox)
        shutil.copy(_SAMPLE, os.path.join(sandbox, '2dal8.iprior'))
        shutil.copy(_SAMPLE, os.path.join(sandbox, 'BLR.iprior.3'))
        custom = os.path.join(input_dir, '2dal8.iprior')
        shutil.copy(_SAMPLE, custom)
        with open(custom, 'a') as handle:
            handle.write('# custom\n')
        catalog = os.path.join(input_dir, 'gal.txt')
        with open(catalog, 'w') as handle:
            handle.write('# gal 0 0 0\n')
        Priors().priors_init(SimpleNamespace(input_file=catalog, outdir=outdir))
        names = {
            name for name in os.listdir(input_dir)
            if name.endswith('.iprior') or '.iprior.' in name
        }
        assert names == {'2dal8.iprior', 'BLR.iprior.3'}
        with open(custom) as handle:
            assert '# custom' in handle.read()
    finally:
        shutil.rmtree(tmp)


def test_priors_init_raises_when_dummy_generate_is_not_z4():
    tmp = tempfile.mkdtemp(prefix='bayesed_priors_init_raise_')
    try:
        input_dir = os.path.join(tmp, 'input')
        outdir = os.path.join(tmp, 'out')
        os.makedirs(input_dir)
        shutil.copy(_SAMPLE, os.path.join(input_dir, '2dal8.iprior'))
        catalog = os.path.join(input_dir, 'gal.txt')
        with open(catalog, 'w') as handle:
            handle.write('# gal 0 0 0\n')
        params = SimpleNamespace(input_file=catalog, outdir=outdir)
        fail = BayeSEDExecutionError(
            'binary failed',
            returncode=1,
            output='mpirun detected non-zero status',
        )
        with patch('bayesed.priors.BayeSEDInterface') as iface_cls:
            iface_cls.return_value.run.side_effect = fail
            try:
                SEDInference().priors_init(params)
            except (RuntimeError, ValueError) as exc:
                if 'mpirun detected non-zero status' in str(exc):
                    return
                raise AssertionError(
                    'priors_init raised, but not with the dummy binary output'
                ) from exc
            raise AssertionError(
                'priors_init must raise when dummy generate returns unexpected binary output'
            )
    finally:
        shutil.rmtree(tmp)


if __name__ == '__main__':
    test_classify_sandbox_round_continue_and_done()
    test_union_copy_missing_does_not_overwrite()
    test_priors_init_copies_missing_and_keeps_working()
    test_priors_init_raises_when_dummy_generate_is_not_z4()
    print('ok')
