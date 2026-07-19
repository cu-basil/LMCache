# SPDX-License-Identifier: Apache-2.0
import logging
from types import SimpleNamespace

import torch

from lmcache.v1.memory_management import BytesBufferMemoryObj
from lmcache.v1.storage_backend.naive_serde import cachegen_decoder


def test_deserialized_cachegen_tensor_has_one_owned_reference(
    monkeypatch,
    caplog,
) -> None:
    """Normal retrieval cleanup must not look like a double free."""
    num_layers = 2
    num_tokens = 4
    num_heads = 2
    head_size = 4
    channels = num_heads * head_size

    encoded = SimpleNamespace(
        cdf=torch.zeros((num_layers * 2, channels)),
        data_chunks=[],
        max_tensors_key=torch.zeros((num_layers, num_tokens)),
        max_tensors_value=torch.zeros((num_layers, num_tokens)),
        num_heads=num_heads,
        head_size=head_size,
    )
    decoded_key = torch.zeros((num_layers, num_tokens, channels))
    decoded_value = torch.ones((num_layers, num_tokens, channels))

    monkeypatch.setattr(
        cachegen_decoder.CacheGenGPUEncoderOutput,
        "from_bytes",
        staticmethod(lambda _: encoded),
    )
    monkeypatch.setattr(
        cachegen_decoder,
        "decode_function_gpu",
        lambda *_args: (decoded_key, decoded_value),
    )
    monkeypatch.setattr(
        cachegen_decoder,
        "do_dequantize",
        lambda value, *_args: value,
    )

    deserializer = object.__new__(cachegen_decoder.CacheGenDeserializer)
    deserializer.dtype = torch.float32
    deserializer.chunk_size = num_tokens
    deserializer.output_buffer = None
    deserializer.key_bins = torch.zeros(num_layers)
    deserializer.value_bins = torch.zeros(num_layers)

    memory_obj = deserializer.deserialize(BytesBufferMemoryObj(b"encoded"))
    assert memory_obj.get_ref_count() == 1

    with caplog.at_level(logging.WARNING):
        memory_obj.ref_count_down()

    assert memory_obj.get_ref_count() == 0
    assert "Double free occurred" not in caplog.text
