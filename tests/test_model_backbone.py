import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import picklikeme.model as model_module
from picklikeme.model import DINOV3_BACKBONE, ModelConfig, PreferenceHead


class CnnBackboneTests(unittest.TestCase):
    def test_forward_shape_and_checkpoint_keys(self):
        model = PreferenceHead(ModelConfig(backbone="cnn"))
        out = model(torch.zeros(2, 3, 32, 32))
        self.assertEqual(out.shape, (2, 1))
        self.assertIn("classifier.weight", model.state_dict())

    def test_all_params_trainable(self):
        model = PreferenceHead(ModelConfig(backbone="cnn"))
        self.assertTrue(all(p.requires_grad for p in model.parameters()))


class PretrainedBackboneTests(unittest.TestCase):
    def test_default_backbone_is_dinov3(self):
        self.assertEqual(ModelConfig().backbone, DINOV3_BACKBONE)

    def test_frozen_backbone_forward_and_grad(self):
        model = PreferenceHead(ModelConfig(backbone=DINOV3_BACKBONE, pretrained=False, freeze_backbone=True))
        out = model(torch.zeros(1, 3, 64, 64))
        self.assertEqual(out.shape, (1, 1))
        self.assertTrue(all(not p.requires_grad for p in model.backbone.parameters()))
        self.assertTrue(all(p.requires_grad for p in model.classifier.parameters()))

    def test_frozen_backbone_stays_in_eval_after_model_train(self):
        model = PreferenceHead(ModelConfig(backbone=DINOV3_BACKBONE, pretrained=False, freeze_backbone=True))
        model.train()
        self.assertFalse(model.backbone.training)
        self.assertTrue(model.classifier.training)

    def test_unfrozen_backbone_is_trainable(self):
        model = PreferenceHead(ModelConfig(backbone=DINOV3_BACKBONE, pretrained=False, freeze_backbone=False))
        self.assertTrue(all(p.requires_grad for p in model.backbone.parameters()))

    def test_checkpoint_keys_stable_across_backbones(self):
        cnn_model = PreferenceHead(ModelConfig(backbone="cnn"))
        pretrained_model = PreferenceHead(ModelConfig(backbone=DINOV3_BACKBONE, pretrained=False))
        self.assertIn("classifier.weight", cnn_model.state_dict())
        self.assertIn("classifier.weight", pretrained_model.state_dict())

    def test_missing_timm_raises_helpful_error(self):
        original = model_module.timm
        model_module.timm = None
        try:
            with self.assertRaises(RuntimeError) as ctx:
                PreferenceHead(ModelConfig(backbone=DINOV3_BACKBONE))
            self.assertIn("timm", str(ctx.exception))
        finally:
            model_module.timm = original


if __name__ == "__main__":
    unittest.main()
