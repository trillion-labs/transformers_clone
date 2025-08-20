# coding=utf-8
# Copyright 2025 the HuggingFace Team. All rights reserved.
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
"""Testing suite for the PyTorch Tri-70B model."""

import unittest

from transformers import AutoTokenizer, is_torch_available
from transformers.testing_utils import (
    Expectations,
    cleanup,
    require_read_token,
    require_torch,
    require_torch_accelerator,
    slow,
    torch_device,
)

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


if is_torch_available():
    import torch

    from transformers import (
        TriConfig,
        TriForCausalLM,
        TriModel,
    )
    from transformers.models.llama.modeling_llama import TriRotaryEmbedding


class TriModelTester(CausalLMModelTester):
    if is_torch_available():
        config_class = TriConfig
        base_model_class = TriModel
        causal_lm_class = TriForCausalLM


@require_torch
class TriModelTest(CausalLMModelTest, unittest.TestCase):
    all_model_classes = (
        (
            TriModel,
            TriForCausalLM,
        )
        if is_torch_available()
        else ()
    )
    pipeline_model_mapping = (
        {
            "feature-extraction": TriModel,
            "text-generation": TriForCausalLM,
        }
        if is_torch_available()
        else {}
    )
    test_headmasking = False
    test_pruning = False
    fx_compatible = False  # Broken by attention refactor cc @Cyrilvallez
    model_tester_class = TriModelTester
    rotary_embedding_layer = TriRotaryEmbedding  # Enables RoPE tests if set

    # Need to use `0.8` instead of `0.9` for `test_cpu_offload`
    # This is because we are hitting edge cases with the causal_mask buffer
    model_split_percents = [0.5, 0.7, 0.8]

    # used in `test_torch_compile_for_training`
    _torch_compile_train_cls = TriForCausalLM if is_torch_available() else None


@require_torch_accelerator
@require_read_token
class TriIntegrationTest(unittest.TestCase):
    def setup(self):
        cleanup(torch_device, gc_collect=True)

    def tearDown(self):
        # TODO (joao): automatic compilation, i.e. compilation when `cache_implementation="static"` is used, leaves
        # some memory allocated in the cache, which means some object is not being released properly. This causes some
        # unoptimal memory usage, e.g. after certain tests a 7B model in FP16 no longer fits in a 24GB GPU.
        # Investigate the root cause.
        cleanup(torch_device, gc_collect=True)

    @slow
    def test_model_generation(self):
        expected_texts = Expectations(
            {
                ("cuda", None): "Tell me about the Chosen kingdom",
            }
        )
        EXPECTED_TEXT = expected_texts.get_expectation()

        tokenizer = AutoTokenizer.from_pretrained("trillionlabs/Tri-70B-preview-SFT")
        model = TriForCausalLM.from_pretrained(
            "trillionlabs/Tri-70B-preview-SFT", device_map="auto", torch_dtype=torch.bfloat16
        )
        input_text = ["Tell me about the Chosen kingdom."]
        model_inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

        generated_ids = model.generate(**model_inputs, max_new_tokens=128, do_sample=False)
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        self.assertEqual(generated_text, EXPECTED_TEXT)

    @slow
    def test_model_7b_logits_bf16(self):
        input_ids = [1, 306, 4658, 278, 6593, 310, 2834, 338]

        model = TriForCausalLM.from_pretrained(
            "trillionlabs/Tri-70B-preview-SFT", device_map="auto", torch_dtype=torch.bfloat16, attn_implementation="eager"
        )

        with torch.no_grad():
            out = model(torch.tensor([input_ids]).to(torch_device))
        # Expected mean on dim = -1

        # fmt: off
        expected_means = Expectations(
            {
            ("xpu", 3): torch.tensor([[-6.5208, -4.1218, -4.9377, -3.2536,  0.8127, -2.9811,  1.2918, -3.3848]]),
            ("cuda", 7): torch.tensor([[-6.5061, -4.1147, -4.9669, -3.2038, 0.8069, -2.9694, 1.2864, -3.3786]]),
            ("cuda", 8): torch.tensor([[-6.5208, -4.1218, -4.9377, -3.2536,  0.8127, -2.9811,  1.2918, -3.3848]]),
            ("rocm", (9, 4)): torch.tensor([[-6.5094, -4.1329, -4.9754, -3.5042,  0.8082, -2.9443,  1.2830, -3.3539]]),
        })

        expected_mean = expected_means.get_expectation().to(torch_device)
        actual_mean = out.logits.float().mean(-1)
        self.assertTrue(
            torch.allclose(
                expected_mean,
                actual_mean,
                atol=1e-2,
                rtol=1e-2
            )
        )

        # slicing logits[0, 0, 0:15]
        expected_slices = Expectations(
            {
            ("cuda", None): torch.tensor([[-12.5000, -7.0625, -0.6289, -7.8750, -6.9688, -7.8125, -6.4688, -7.4375, -7.6875, -6.9375, -6.0312, -7.0000, -1.8594, 1.8438, -8.5000]]),
        })
        # fmt: on
        expected_slice = expected_slices.get_expectation().to(torch_device)
        actual_slice = out.logits[0, 0, :15].float()
        self.assertTrue(torch.allclose(expected_slice, actual_slice, atol=1e-2, rtol=1e-2))

    @slow
    def test_model_7b_logits(self):
        input_ids = [1, 306, 4658, 278, 6593, 310, 2834, 338]

        model = TriForCausalLM.from_pretrained("trillionlabs/Tri-70B-preview-SFT", device_map="auto", torch_dtype=torch.float16)

        with torch.no_grad():
            out = model(torch.tensor([input_ids]).to(torch_device))

        # fmt: off
        # Expected mean on dim = -1
        expected_means = Expectations(
          {
            ("xpu", 3): torch.tensor([[-6.6544, -4.1259, -4.9840, -3.2456,  0.8261, -3.0124,  1.2971, -3.3641]]),
            ("cuda", 7): torch.tensor([[-6.6420, -4.1227, -4.9809, -3.2041, 0.8261, -3.0052, 1.2957, -3.3648]]),
            ("cuda", 8): torch.tensor([[-6.6544, -4.1259, -4.9840, -3.2456,  0.8261, -3.0124,  1.2971, -3.3641]]),
        })

        expected_mean = expected_means.get_expectation()
        self.assertTrue(
            torch.allclose(
                expected_mean.to(torch_device),
                out.logits.float().mean(-1),
                atol=1e-2,
                rtol=1e-2
            )
        )

        # slicing logits[0, 0, 0:15]
        expected_slices = Expectations(
            {
              ("xpu", 3): torch.tensor([-12.8281,  -7.4609,  -0.4668,  -8.0703,  -7.2539,  -8.0078,  -6.4961, -7.7734,  -7.8516,  -7.0352,  -6.2188,  -7.1367,  -1.8564,   1.9922, -8.6328]),
              ("cuda", 7): torch.tensor([-12.8125, -7.3359, -0.4846, -8.0234, -7.2383, -7.9922, -6.4805, -7.7344, -7.8125, -7.0078, -6.1797, -7.1094, -1.8633, 1.9736, -8.6016]),
              ("cuda", 8): torch.tensor([-12.8281,  -7.4609,  -0.4668,  -8.0703,  -7.2539,  -8.0078,  -6.4961, -7.7734,  -7.8516,  -7.0352,  -6.2188,  -7.1367,  -1.8564,   1.9922, -8.6328])
        })
        # fmt: on

        expected_slice = expected_slices.get_expectation()
        self.assertTrue(
            torch.allclose(
                expected_slice.to(torch_device),
                out.logits[0, 0, :15].float(),
                atol=1e-2,
                rtol=1e-2,
            )
        )

