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
"""SGLang prompt-logprob metadata may echo an internal visual pad id (or 0)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from verl.workers.rollout.sglang_rollout.async_sglang_server import (
    _extract_prompt_logprobs_sglang,
    _record_prompt_logprob_id_aliases,
)

IMAGE_PAD = 99
HASH_PAD = 1_222_222


class _FakeImageItem:
    def __init__(self, offsets, pad_value=HASH_PAD):
        self.offsets = offsets
        self.pad_value = pad_value

    def set_pad_value(self):
        if self.pad_value is None:
            self.pad_value = HASH_PAD


def test_aliases_are_limited_to_concrete_multimodal_offsets():
    request = SimpleNamespace()
    mm_inputs = {
        "input_ids": [10, IMAGE_PAD, 20],
        "padded_input_ids": [1_111_111, HASH_PAD, 20],
        "mm_items": [_FakeImageItem(offsets=[(1, 1)], pad_value=HASH_PAD)],
    }

    _record_prompt_logprob_id_aliases(mm_inputs, [10, IMAGE_PAD, 20], request, vocab_size=500)

    assert request._verl_prompt_logprob_id_aliases == {1: {0, HASH_PAD}}


def test_matching_padded_and_exact_ids_are_not_aliased():
    request = SimpleNamespace()
    mm_inputs = {
        "input_ids": [10, IMAGE_PAD, 20],
        "padded_input_ids": [10, IMAGE_PAD, 20],
        "mm_items": [_FakeImageItem(offsets=[(1, 1)], pad_value=IMAGE_PAD)],
    }

    _record_prompt_logprob_id_aliases(mm_inputs, [10, IMAGE_PAD, 20], request, vocab_size=500)

    assert request._verl_prompt_logprob_id_aliases == {}


def test_text_tokens_next_to_a_span_are_not_aliased():
    request = SimpleNamespace()
    mm_inputs = {
        "input_ids": [10, IMAGE_PAD, 20],
        "padded_input_ids": [0, HASH_PAD, 0],
        "mm_items": [_FakeImageItem(offsets=[(1, 1)])],
    }

    _record_prompt_logprob_id_aliases(mm_inputs, [10, IMAGE_PAD, 20], request, vocab_size=500)

    assert request._verl_prompt_logprob_id_aliases == {1: {0, HASH_PAD}}


def test_extract_restores_clipped_pad_id_and_keeps_logprobs():
    request = SimpleNamespace()
    sequence = [10, IMAGE_PAD, 20]
    mm_inputs = {
        "padded_input_ids": [10, HASH_PAD, 20],
        "mm_items": [_FakeImageItem(offsets=[(1, 1)])],
    }
    _record_prompt_logprob_id_aliases(mm_inputs, sequence, request, vocab_size=500)

    extra = {}
    restored = _extract_prompt_logprobs_sglang(
        meta_info={
            "input_token_logprobs": [
                (None, 10, None),
                (-0.2, 0, None),
                (-0.3, 20, None),
            ]
        },
        num_prompt_logprobs=0,
        sequence_ids=sequence,
        result_dict=extra,
        prompt_logprob_id_aliases=request._verl_prompt_logprob_id_aliases,
    )

    assert restored == 1
    assert extra["prompt_ids"] == [[IMAGE_PAD], [20], [0]]
    assert extra["prompt_logprobs"][0] == [pytest.approx(-0.2)]
    assert extra["prompt_logprobs"][1] == [pytest.approx(-0.3)]


def test_extract_fail_closes_on_an_unaliased_mismatch():
    extra = {}
    with pytest.raises(AssertionError, match="first_mismatch_index=1"):
        _extract_prompt_logprobs_sglang(
            meta_info={
                "input_token_logprobs": [
                    (None, 10, None),
                    (-0.2, 0, None),
                    (-0.3, 20, None),
                ]
            },
            num_prompt_logprobs=0,
            sequence_ids=[10, IMAGE_PAD, 20],
            result_dict=extra,
        )


def test_extract_accepts_omitted_first_row():
    extra = {}
    restored = _extract_prompt_logprobs_sglang(
        meta_info={
            "input_token_logprobs": [
                (-0.2, 11, None),
                (-0.3, 12, None),
            ]
        },
        num_prompt_logprobs=0,
        sequence_ids=[10, 11, 12],
        result_dict=extra,
    )
    assert restored == 0
    assert extra["prompt_ids"] == [[11], [12], [0]]
