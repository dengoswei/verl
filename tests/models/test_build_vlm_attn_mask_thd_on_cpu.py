# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""One-row VLM + sequence-parallel must pack as B>1 before TP scatter."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import verl.models.mcore.util as mcore_util
from verl.models.mcore.model_forward_fused import fused_forward_model_engine
from verl.models.mcore.util import build_vlm_attn_mask_thd


class _FakeMpu:
    def __init__(self, tp: int = 1, cp: int = 1, cp_rank: int = 0):
        self._tp, self._cp, self._cp_rank = tp, cp, cp_rank

    def get_tensor_model_parallel_world_size(self):
        return self._tp

    def get_context_parallel_world_size(self):
        return self._cp

    def get_context_parallel_rank(self):
        return self._cp_rank


@pytest.fixture
def parallel_state(monkeypatch):
    def _set(tp=1, cp=1):
        monkeypatch.setattr(mcore_util, "mpu", _FakeMpu(tp=tp, cp=cp))

    return _set


def _nested(rows, dtype=torch.long):
    return torch.nested.nested_tensor([torch.tensor(r, dtype=dtype) for r in rows], layout=torch.jagged)


def test_a_one_row_vlm_sp_batch_forces_pack_then_scatter(parallel_state):
    parallel_state(tp=4, cp=1)
    input_ids = _nested([[1, 2, 3, 4, 5]])

    bshd, attention_mask = build_vlm_attn_mask_thd(input_ids, pad_token_id=0, sequence_parallel=True)

    assert bshd.shape == (2, 8)
    assert attention_mask.shape == bshd.shape
    assert attention_mask[0].tolist() == [True, True, True, True, True, False, False, False]
    assert not attention_mask[1].any()
    assert not bshd[1].any()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_a_one_row_vlm_sp_batch_stays_on_cuda(parallel_state):
    parallel_state(tp=4, cp=1)
    row = torch.tensor([1, 2, 3, 4, 5], dtype=torch.long, device="cuda")
    input_ids = torch.nested.nested_tensor([row], layout=torch.jagged)
    bshd, attention_mask = build_vlm_attn_mask_thd(input_ids, pad_token_id=0, sequence_parallel=True)
    assert bshd.device.type == "cuda"
    assert attention_mask.device.type == "cuda"
    assert bshd.shape == (2, 8)
    assert not attention_mask[1].any()


def test_a_one_row_vlm_batch_without_sp_stays_batch_one(parallel_state):
    parallel_state(tp=4, cp=1)
    input_ids = _nested([[1, 2, 3, 4, 5]])

    bshd, attention_mask = build_vlm_attn_mask_thd(input_ids, pad_token_id=0, sequence_parallel=False)

    assert bshd.shape == (1, 8)
    assert attention_mask.shape == (1, 8)
    assert attention_mask[0].sum().item() == 5


@pytest.mark.parametrize(("sequence_parallel", "expected_shape"), [(True, (2, 8)), (False, (1, 8))])
def test_fused_vlm_forward_uses_the_aligned_bshd_contract(parallel_state, sequence_parallel, expected_shape):
    parallel_state(tp=4, cp=1)

    class Model:
        pre_process = True
        post_process = False
        config = SimpleNamespace(fp8=None, sequence_parallel=sequence_parallel)

        def __call__(self, **kwargs):
            input_ids = kwargs["input_ids"]
            attention_mask = kwargs["attention_mask"]
            assert input_ids.shape == expected_shape
            assert attention_mask.shape == input_ids.shape
            assert attention_mask[0].sum().item() == 5
            if sequence_parallel:
                assert not attention_mask[1].any()
            return "model-output"

    forward = fused_forward_model_engine(vision_model=True)
    result = forward(
        Model(),
        input_ids=_nested([[1, 2, 3, 4, 5]]),
        labels=_nested([[2, 3, 4, 5, 6]]),
        multi_modal_inputs={},
        temperature=1.0,
        calculate_entropy=False,
        pad_token_id=0,
    )
    assert result == "model-output"
