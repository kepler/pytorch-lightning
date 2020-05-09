from unittest.mock import MagicMock

import pytest
import torch

import tests.base.utils as tutils
from pytorch_lightning import Trainer
from tests.base import EvalModelTemplate


@pytest.mark.parametrize('max_steps', [1, 2, 3])
def test_on_before_zero_grad_called(max_steps):

    class CurrentTestModel(EvalModelTemplate):
        on_before_zero_grad_called = 0

        def on_before_zero_grad(self, optimizer):
            self.on_before_zero_grad_called += 1

    model = CurrentTestModel(tutils.get_default_hparams())

    trainer = Trainer(
        max_steps=max_steps,
        num_sanity_val_steps=5,
    )
    assert 0 == model.on_before_zero_grad_called
    trainer.fit(model)
    assert max_steps == model.on_before_zero_grad_called

    model.on_before_zero_grad_called = 0
    trainer.test(model)
    assert 0 == model.on_before_zero_grad_called


@pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires GPU machine")
def test_transfer_batch_hook():

    class CustomBatch:

        def __init__(self, data):
            self.samples = data[0]
            self.targets = data[1]

    class CurrentTestModel(EvalModelTemplate):

        def transfer_batch_to_device(self, batch, device):
            if isinstance(batch, CustomBatch):
                batch.samples = batch.samples.to(device)
                batch.targets = batch.targets.to(device)
            return batch

    model = CurrentTestModel(tutils.get_default_hparams())
    batch = CustomBatch((torch.zeros(5, 28), torch.ones(5, 1, dtype=torch.long)))

    trainer = Trainer()
    # running .fit() would require us to implement custom data loaders, we mock the model reference instead
    trainer.get_model = MagicMock(return_value=model)

    batch_gpu = trainer.transfer_batch_to_gpu(batch, 0)
    device = torch.device('cuda', 0)
    assert batch_gpu.samples.device == batch_gpu.targets.device == device
