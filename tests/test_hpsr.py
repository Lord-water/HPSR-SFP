"""Synthetic engineering regressions; no empirical study data or model downloads."""
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from hpsr_sfp import LABELS
from hpsr_sfp.hierarchical import hfit, hpredict, objective, rules
from hpsr_sfp.io import features, read_json, write_json
from hpsr_sfp.readouts import prepare, scores
from hpsr_sfp.selection import fit_selected, predict_bundle, restore_bundle
from hpsr_sfp.statistics import holm, paired


class ScientificMath(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(29)
        self.x = self.rng.normal(size=(8, 12, 3))
        self.u = self.rng.normal(size=(8, 12))
        self.ub = self.u.mean(1)
        self.y = self.rng.integers(0, 2, size=(8, 12)).astype(float)
        self.y[0, 0] = -1

    def test_gradient_all_parameter_groups(self):
        for rho, cues in [(5., True), (float('inf'), False)]:
            dim = 3+12+(0 if np.isinf(rho) else 36)+(2 if cues else 0)
            v = self.rng.normal(size=dim)*.1
            _, gradient = objective(v, self.x, self.u, self.ub, self.y, .3, rho, cues)
            numeric = []
            for index in range(dim):
                plus, minus = v.copy(), v.copy()
                plus[index] += 1e-6
                minus[index] -= 1e-6
                numeric.append((objective(plus, self.x, self.u, self.ub, self.y, .3, rho, cues)[0]
                               - objective(minus, self.x, self.u, self.ub, self.y, .3, rho, cues)[0])/2e-6)
            np.testing.assert_allclose(gradient, numeric, atol=1e-7, rtol=1e-6)

    def test_unknown_labels_do_not_enter_loss(self):
        y = np.full((2, 12), -1.)
        y[0] = 1.
        v = np.zeros(3+12)
        x = self.x[:2].copy()
        loss, grad = objective(v, x, self.u[:2], self.ub[:2], y, .3, float('inf'), False)
        x[1] *= 100000
        other, other_grad = objective(v, x, self.u[:2], self.ub[:2], y, .3, float('inf'), False)
        self.assertEqual(loss, other)
        np.testing.assert_array_equal(grad, other_grad)

    def test_one_sided_labels_have_finite_stabilized_fit(self):
        model = hfit(self.x, self.u, self.ub, np.ones((8, 12)), .3, 5., True)
        self.assertTrue(model['success'])
        self.assertLess(model['grad_max'], 1e-4)
        self.assertTrue(np.isfinite(hpredict(model, self.x, self.u, self.ub)).all())

    def test_raw_and_centering_have_distinct_shift_behavior(self):
        shifted = self.x + self.rng.normal(size=(8, 1, 3))
        raw, _ = prepare(self.x, self.u, 'raw')
        raw_shift, _ = prepare(shifted, self.u, 'raw')
        centered, _ = prepare(self.x, self.u, 'center_linear')
        centered_shift, _ = prepare(shifted, self.u, 'center_linear')
        np.testing.assert_allclose(centered, centered_shift, atol=1e-14)
        self.assertGreater(np.max(np.abs(raw-raw_shift)), .1)

    def test_fixed_decoders_and_empty_gate(self):
        p = np.full((2, 12), .01)
        p[0, 0], p[0, 1], p[1, 2] = .2, .8, .4
        logq = np.full((2, 12), -10.)
        logq[0, :2], logq[1, :3] = [-.1, -.3], [0., -.2, -.5]
        result = rules(p, logq, np.tile([0, 1]+[-1]*10, (2, 1)))
        self.assertEqual(result['Usage-Only']['indices'], [0, 0])
        self.assertEqual(result['Rule-PoE']['indices'], [1, 2])
        self.assertEqual(result['Rule-BayesGate']['indices'], [1, 2])
        self.assertEqual(result['Rule-BayesGate']['fallbacks'], 1)

    def test_metrics_and_paired_statistics(self):
        y = np.tile([0., 1.], (12, 1)).T
        p = np.full_like(y, .5)
        report = scores(p, y)
        self.assertAlmostEqual(report['macro_brier'], .25)
        self.assertAlmostEqual(report['macro_auroc'], .5)
        self.assertAlmostEqual(report['pooled_ece_10_equal_width_bins'], 0.)
        comparison = paired(p, p, y, ['context0', 'context1'], reps=40)
        self.assertEqual(comparison['macro_brier_difference'], 0.)
        self.assertEqual(comparison['context_cluster_bootstrap95'], [0., 0.])
        self.assertEqual(comparison['two_sided_randomization_p'], 1.)
        comparisons = {'a': {'two_sided_randomization_p': .01}, 'b': {'two_sided_randomization_p': .04}, 'c': {'two_sided_randomization_p': .03}}
        holm(comparisons)
        self.assertEqual([comparisons[k]['holm_p'] for k in ['a', 'b', 'c']], [.03, .06, .06])

    def test_portable_bundle_and_train_dev_selection(self):
        train = dict(h=self.x, u=self.u, record_ids=np.array(['t'+str(i) for i in range(8)]), encoder_id=np.array('fixture'))
        dev = dict(h=self.rng.normal(size=(10, 12, 3)), u=self.rng.normal(size=(10, 12)),
                   record_ids=np.array(['d'+str(i) for i in range(10)]), encoder_id=np.array('fixture'))
        dy = np.tile(np.array([0, 1]*5)[:, None], (1, 12))
        # Exercise residual/shared and cue-enabled/cue-free branches without
        # running the full scientific hyperparameter search in a unit test.
        with patch('hpsr_sfp.selection.itertools') as grid:
            grid.product.return_value = iter([(.3, 5., True), (.3, float('inf'), False)])
            model, trace = fit_selected(train, dev, self.y, dy, np.repeat(np.arange(5), 2), 'hpsr')
        self.assertEqual(len(trace), 4)
        before = predict_bundle(model, dev)
        with tempfile.TemporaryDirectory() as folder:
            file = Path(folder)/'readout.json'
            write_json(file, model)
            restored = restore_bundle(read_json(file))
        np.testing.assert_array_equal(before, predict_bundle(restored, dev))
        wrong = dict(dev, encoder_id=np.array('different'))
        with self.assertRaises(ValueError):
            predict_bundle(restored, wrong)


class CommandLine(unittest.TestCase):
    def test_fit_predict_subset_score(self):
        rng = np.random.default_rng(8)
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            for role, count in [('train', 8), ('dev', 10), ('test', 5)]:
                ids = [role+str(i) for i in range(count)]
                np.savez(folder/(role+'.npz'), h=rng.normal(size=(count, 12, 3)), u=rng.normal(size=(count, 12)),
                         record_ids=np.array(ids), label_order=np.array(LABELS), encoder_id=np.array('fixture'))
                mapping = {rid: {label: ('A' if i % 2 else 'N') for label in LABELS} for i, rid in enumerate(ids)}
                write_json(folder/(role+'.json'), mapping)
            write_json(folder/'folds.json', {'dev'+str(i): i//2 for i in range(10)})
            def run(*args):
                from hpsr_sfp.cli import main
                with patch.object(sys, 'argv', ['hpsr_sfp', *map(str, args)]), contextlib.redirect_stdout(io.StringIO()):
                    main()
            run('fit', '--train-features', folder/'train.npz', '--dev-features', folder/'dev.npz',
                '--train-labels', folder/'train.json', '--dev-labels', folder/'dev.json', '--folds', folder/'folds.json',
                '--method', 'raw', '--output', folder/'head.json')
            run('predict', '--features', folder/'test.npz', '--readout', folder/'head.json', '--output', folder/'predictions.json')
            subset = {key: value for key, value in read_json(folder/'test.json').items() if key != 'test4'}
            write_json(folder/'subset.json', subset)
            run('score', '--predictions', folder/'predictions.json', '--labels', folder/'subset.json', '--output', folder/'score.json')
            self.assertEqual(sum(read_json(folder/'score.json')['selection']['Usage-Only'].values()), 4)
            self.assertIn('N_minus_Usage-Only', read_json(folder/'score.json')['selection']['Rule-PoE'])
            self.assertIn('fallbacks', read_json(folder/'score.json')['selection']['Rule-BayesGate'])
            run('subset', '--features', folder/'test.npz', '--labels', folder/'subset.json', '--output', folder/'subset.npz')
            self.assertEqual(features(folder/'subset.npz')['h'].shape, (4, 12, 3))


class EncoderInterface(unittest.TestCase):
    def test_tiny_local_encoder_and_candidate_positions(self):
        import torch
        from transformers import BertConfig, BertModel
        from hpsr_sfp.encoder import UsageEncoder
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as folder:
            config = BertConfig(vocab_size=32, hidden_size=16, num_hidden_layers=1,
                                num_attention_heads=2, intermediate_size=24)
            BertModel(config).save_pretrained(folder)
            model = UsageEncoder(folder, device='cpu', adapter=True)
            self.assertTrue(model.adapted)
            ids = torch.randint(0, 32, (12, 5))
            utilities, h = model(ids, torch.ones_like(ids), torch.full((12,), 3))
            self.assertEqual(tuple(h.shape), (12, 16))
            self.assertEqual(tuple(utilities.shape), (12,))
            self.assertTrue(torch.isfinite(h).all())
            saved = model.state()
            self.assertTrue(any('.a' in name for name in saved))
            self.assertFalse(any('.base.' in name for name in saved))
            model.restore(saved)


if __name__ == '__main__':
    unittest.main()
