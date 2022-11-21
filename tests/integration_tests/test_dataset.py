# The MIT License (MIT)
# Copyright © 2021 Yuma Rao

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated 
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation 
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, 
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of 
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION 
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
# DEALINGS IN THE SOFTWARE.

import bittensor
from . import constant
from unittest.mock import MagicMock
logging = bittensor.logging()

def test_construct_text_corpus():
    dataset = bittensor.dataset(num_batches = constant.dataset.num_batches, dataset_name = constant.dataset.dataset_name)
    dataset.close()

def test_change_data_size():
    data_sizes = [dict(batch_size=10,sequence_length=64, block_size=1000, buffer_size=2000),
                        dict(batch_size=20,sequence_length=128, block_size=2000, buffer_size=3000),
                        dict(batch_size=30,sequence_length=256, block_size=4000, buffer_size=1000) ]
    dataset = bittensor.dataset(num_batches = constant.dataset.num_batches, dataset_name = constant.dataset.dataset_name, no_tokenizer=False)
    for data_size in data_sizes:
        dataset.set_data_size(**data_size)
        dataset.no_tokenizer = False
        sample = next(dataset)
        assert sample.shape[0] == data_size['batch_size']
        assert sample.shape[1] == data_size['sequence_length']
        assert dataset.block_size == data_size['block_size']
        assert dataset.buffer_size == data_size['buffer_size'] == len(dataset.sample_buffer)
        dataset.no_tokenizer = True
        sample = next(dataset)
        assert len(sample)== data_size['batch_size']
        assert len(sample[0].split()) == data_size['sequence_length']

    dataset.close() 




def test_next_tokenized_sample():
    batch_size = 10
    sequence_length = 128
    block_size = 500

    
    dataset = bittensor.dataset (
        block_size = block_size,
        batch_size = batch_size,
        sequence_length = sequence_length,
         dataset_name = constant.dataset.dataset_name,
        no_tokenizer=False
    )

    input = next(dataset)
    assert input.shape[0] == batch_size
    assert input.shape[1]  == sequence_length
    dataset.close()

def test_next_raw_sample():
    batch_size = 10
    sequence_length = 128
    block_size = 500
    dataset = bittensor.dataset (
        block_size = block_size,
        batch_size = batch_size,
        sequence_length = sequence_length,
        dataset_name = constant.dataset.dataset_name,
        no_tokenizer = True
    )

    input = next(dataset)
    assert len(input) == batch_size
    for i in range(len(input)):
        assert len(input[i].split()) == sequence_length

    dataset.close()


def test_fail_IPFS_server():
    dataset = bittensor.dataset(num_batches = constant.dataset.num_batches, dataset_name = constant.dataset.dataset_name)
    dataset.requests_retry_session = MagicMock(return_value = None)
    next(dataset)
    next(dataset)
    next(dataset)
    dataset.close()


if __name__ == "__main__":
    test_change_data_size()