import pytest
import torch
from crackvision.model import (
    CrackVisionNet,
    SwinBlock,
    WindowAttention,
    partition,
    reverse,
    SpatialAttention,
    UNet,
)
from crackvision.losses import SegmentationLoss

torch.set_num_threads(2)


def tiny(**kw):
    return CrackVisionNet(
        embed_dim=12, depths=(2, 2, 2, 2), heads=(3, 3, 6, 12), window=4, **kw
    )


@pytest.mark.parametrize("shape", [(64, 64), (65, 91), (32, 33)])
@pytest.mark.parametrize("attention", [True, False])
def test_shape_gradient(shape, attention):
    model = tiny(spatial_attention=attention)
    x = torch.randn(1, 3, *shape)
    y = model(x)
    assert y.shape == (1, 1, *shape)
    loss = SegmentationLoss()(y, torch.zeros_like(y))
    loss.backward()
    assert torch.isfinite(loss)
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()
    )


def test_partition_inverse():
    x = torch.randn(2, 8, 12, 9)
    torch.testing.assert_close(reverse(partition(x, 4), 4, 8, 12, 2), x)


def test_shift_mask_blocks_wraparound_and_padding():
    block = SwinBlock(12, 3, 4, True)
    mask = block.attention_mask(7, 9, 8, 12, 2, "cpu")
    assert mask.shape == (6, 16, 16)
    assert (mask < 0).any() and (mask == 0).any()
    # On an unpadded map only windows at cyclic wrap boundaries are split.
    mask = block.attention_mask(8, 8, 8, 8, 2, "cpu")
    assert (mask[0] == 0).all()
    assert (mask[-1] < 0).any()


def test_attention_matches_direct_formula():
    torch.manual_seed(2)
    m = WindowAttention(12, 3, 2)
    x = torch.randn(2, 4, 12)
    mask = torch.zeros(1, 4, 4)
    q, k, v = m.qkv(x).reshape(2, 4, 3, 3, 4).permute(2, 0, 3, 1, 4).unbind(0)
    bias = m.bias[m.index.flatten()].reshape(4, 4, 3).permute(2, 0, 1)
    expected = m.proj(
        (((q * 0.5) @ k.transpose(-2, -1) + bias).softmax(-1) @ v)
        .transpose(1, 2)
        .reshape(2, 4, 12)
    )
    torch.testing.assert_close(m(x, mask), expected)


@pytest.mark.parametrize("name", ["hybrid", "bce", "dice", "focal"])
def test_loss_extreme_logits(name):
    z = torch.tensor([[[[-100.0, 100.0]]]], requires_grad=True)
    t = torch.tensor([[[[0.0, 1.0]]]])
    l = SegmentationLoss(name)(z, t)
    l.backward()
    assert torch.isfinite(l) and torch.isfinite(z.grad).all()


def test_unet_arbitrary_shape():
    assert UNet(base=4)(torch.zeros(1, 3, 37, 55)).shape == (1, 1, 37, 55)


def test_spatial_gate():
    x = torch.ones(2, 11, 9, 12)
    y = SpatialAttention()(x)
    assert y.shape == x.shape and (y >= 0).all() and (y <= 1).all()
