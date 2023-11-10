
"""Detailed DCDCompressors, e.g., top-k/random-k, quantization, sign-based quantization."""
"""taken from ChocoSGD"""
"""the entry for DCDCompressor."""

import torch


from utils.data_utils import (
    get_n_bits,
    get_data,
    flatten,
    apply_gradient,
    recover_device,
)

from utils.tensor_buffer import (
    TensorBuffer
)

from compression.compressors_simple import (
    SparsificationCompressor,
    QuantizationCompressor,
)

from .message_define import MyMessage



class DCDCompressor(object):
    def __init__(self, **kargs):
        # assign compressor class.
        if "topk" in kargs["comm_op"] or "randomk" in kargs["comm_op"]:
            self.compressor_fn = DCDSparsificationCompressor(**kargs)
        elif "quantize" in kargs["comm_op"]:
            self.compressor_fn = DCDQuantizationCompressor(**kargs)
        elif "sign" in kargs["comm_op"]:
            self.compressor_fn = DCDSignCompressor(**kargs)
        else:
            raise NotImplementedError

    def compress(self, *args, **kargs):
        return self.compressor_fn.compress(*args, **kargs)

    def uncompress(self, *args, **kargs):
        return self.compressor_fn.uncompress(*args, **kargs)

    def uncompress_direct(self, *args, **kargs):
        return self.compressor_fn.uncompress_direct(*args, **kargs)


class DCDSparsificationCompressor(object):
    def __init__(
        self,
        comm_op,
        compress_ratio,
        quantize_level,
        is_biased,
        **kargs
    ):
        # assign the common hyper-parameters
        self.comm_op = comm_op
        self.compress_ratio = compress_ratio
        self.quantize_level = quantize_level
        self.is_biased = is_biased
        self.kargs = kargs
        self.compressor_fn = SparsificationCompressor()


    def compress(self, sync_buffer):
        # get the sign/magnitude for the tensor (to be transmitted).
        selected_values, selected_indices = [], []

        for half_param, hat_param in zip(
            sync_buffer["flatten_half_params"], sync_buffer["flatten_params"]
        ):
            _selected_values, _selected_indices = self.compressor_fn.compress(
                half_param - hat_param,
                self.comm_op,
                self.compress_ratio,
                self.is_biased,
            )
            selected_values.append(_selected_values)
            selected_indices.append(_selected_indices)

        # get selected shapes.
        selected_shapes = [len(_value) for _value in selected_values]

        # flatten selected values/indices.
        flatten_selected_values = TensorBuffer(selected_values)
        flatten_selected_indices = TensorBuffer(selected_indices)

        # get n_bits to transmit.
        n_bits = get_n_bits(flatten_selected_values.buffer) + get_n_bits(
            flatten_selected_indices.buffer
        )

        # update shared dict.
        sync_buffer["selected_shapes"] = selected_shapes
        sync_buffer["flatten_selected_values"] = flatten_selected_values
        sync_buffer["flatten_selected_indices"] = flatten_selected_indices
        sync_buffer["n_bits"] = n_bits

    def sync(self, sync_buffer):
        # get the flatten values.
        message_to_send = torch.cat(
            [
                sync_buffer["flatten_selected_values"].buffer,
                sync_buffer["flatten_selected_indices"].buffer,
            ]
        )

        # sync.
        # synced_message = self.aggregator_fn._agg(
        #     message_to_send, op="get_raw_sync_data", force_wait=True
        # )

        # update sync_buffer.
        # sync_buffer["synced_message"] = synced_message
        # sync_buffer["sycned_message_size"] = len(message_to_send)

    def uncompress(self, msg_params, hat_params, selected_shapes, original_shapes):
        """
            values:             compressed values of multiple tensors
            indices:            indices of selected values in their original tensors 
                                (means that indices are not unique)
            selected_shapes:    list of sizes of each compressed tensor
            original_shapes:    list of sizes of each original tensor
            hat_params:         need to be recovered
        """

        # deal with unbalanced values/indieces
        assert original_shapes is not None
        assert selected_shapes is not None
        q_values, q_indices = self.compressor_fn.uncompress(
            values=msg_params.get(MyMessage.MSG_ARG_KEY_SPARSE_PARAMS_1),
            indices=msg_params.get(MyMessage.MSG_ARG_KEY_SPARSE_INDEX_1),
            selected_shapes=selected_shapes,
            original_shapes=original_shapes,
        )

        # update the flatten hat params.
        hat_params.buffer[q_indices.to(hat_params.buffer.device)] += \
            q_values.to(hat_params.buffer.device)

    def uncompress_direct(self, sync_buffer, hat_params, selected_shapes, original_shapes):
        """
            values:             compressed values of multiple tensors
            indices:            indices of selected values in their original tensors 
                                (means that indices are not unique)
            selected_shapes:    list of sizes of each compressed tensor
            original_shapes:    list of sizes of each original tensor
            hat_params:         need to be recovered
        """
        # deal with unbalanced values/indieces
        q_values, q_indices = self.compressor_fn.uncompress(
            values=sync_buffer["flatten_selected_values"].buffer,
            indices=sync_buffer["flatten_selected_indices"].buffer,
            selected_shapes=selected_shapes,
            original_shapes=original_shapes,
        )

        # update the flatten hat params.
        hat_params.buffer[q_indices] += q_values



    def uncompress_old(self, sync_buffer, neighbor_hat_params):
        sycned_message_size = int(sync_buffer["sycned_message_size"] / 2)

        # uncompress and update.
        for rank, hat_params in neighbor_hat_params.items():
            _message = recover_device(
                sync_buffer["synced_message"][rank], device=hat_params.buffer.device
            )
            values = _message[:sycned_message_size]
            indices = _message[sycned_message_size:]

            # deal with unbalanced values/indieces
            q_values, q_indices = self.compressor_fn.uncompress(
                values,
                indices,
                sync_buffer["selected_shapes"],
                sync_buffer["original_shapes"],
            )

            # update the flatten hat params.
            hat_params.buffer[q_indices] += q_values


class DCDQuantizationCompressor(object):
    def __init__(
        self,
        comm_op,
        compress_ratio,
        quantize_level,
        is_biased,
        **kargs
    ):
        # assign the common hyper-parameters
        self.comm_op = comm_op
        self.compress_ratio = compress_ratio
        self.quantize_level = quantize_level
        self.is_biased = is_biased
        self.kargs = kargs
        self.compressor_fn = QuantizationCompressor()

    def compress(self, sync_buffer):
        # get the sign/magnitude for the tensor (to be transmitted).
        quantized_values = []

        for half_param, hat_param in zip(
            sync_buffer["flatten_half_params"], sync_buffer["flatten_params"]
        ):
            _quantized_values = self.compressor_fn.compress(
                half_param - hat_param,
                self.comm_op,
                self.quantize_level,
                self.is_biased,
            )
            quantized_values.append(_quantized_values)

        # flatten selected values/indices.
        flatten_updates = TensorBuffer(quantized_values)

        # get n_bits to transmit.
        n_bits = get_n_bits(flatten_updates.buffer) * self.quantize_level / 32

        # update shared dict.
        sync_buffer["flatten_updates"] = flatten_updates
        sync_buffer["n_bits"] = n_bits

    def sync(self, sync_buffer):
        # prepare the sync.
        to_sync_message = sync_buffer["flatten_updates"].buffer

        # sync.
        # synced_message = self.aggregator_fn._agg(
        #     to_sync_message, op="get_raw_sync_data", force_wait=True
        # )

        # update sync_buffer.
        # sync_buffer["synced_message"] = synced_message

    def uncompress(self, sync_buffer, neighbor_hat_params):
        # uncompress and update.
        for rank, hat_params in neighbor_hat_params.items():
            # map the tensors to the correct location.
            _message = recover_device(
                sync_buffer["synced_message"][rank], device=hat_params.buffer.device
            )

            # update the flatten hat params.
            hat_params.buffer.add_(_message)


class DCDSignCompressor(object):
    def __init__(
        self,
        comm_op,
        compress_ratio,
        quantize_level,
        is_biased,
        **kargs
    ):
        # assign the common hyper-parameters
        self.comm_op = comm_op
        self.compress_ratio = compress_ratio
        self.quantize_level = quantize_level
        self.is_biased = is_biased
        self.kargs = kargs
        # self.compressor_fn = SignCompressor()

    def compress(self, sync_buffer):
        # get the sign/magnitude for the tensor (to be transmitted).
        norms, updates = [], []
        for half_param, hat_param in zip(
            sync_buffer["flatten_half_params"], sync_buffer["flatten_params"]
        ):
            _update = half_param - hat_param
            updates += [_update]
            norms += [_update.norm(p=1)]

        # flatten selected values/indices.
        flatten_norms = TensorBuffer(norms)
        flatten_updates = TensorBuffer(updates)
        signs, sign_size = self.compressor_fn.compress(flatten_updates.buffer)

        # get n_bits to transmit.
        n_bits = get_n_bits(flatten_norms.buffer) + get_n_bits(signs)

        # update shared dict.
        sync_buffer["flatten_norms"] = flatten_norms
        sync_buffer["flatten_updates"] = flatten_updates
        sync_buffer["signs"] = signs
        sync_buffer["sign_size"] = sign_size
        sync_buffer["n_bits"] = n_bits

    def sync(self, sync_buffer):
        # prepare sync.
        to_sync_flatten_norms = sync_buffer["flatten_norms"].buffer
        to_sync_signs = sync_buffer["signs"]

        # sync.
        # synced_flatten_norms = self.aggregator_fn._agg(
        #     to_sync_flatten_norms, op="get_raw_sync_data", force_wait=True
        # )
        # synced_signs = self.aggregator_fn._agg(
        #     to_sync_signs, op="get_raw_sync_data", force_wait=True
        # )

        # update sync_buffer.
        # sync_buffer["synced_flatten_norms"] = synced_flatten_norms
        # sync_buffer["synced_signs"] = synced_signs

    def uncompress(self, sync_buffer, neighbor_hat_params):
        # uncompress and update.
        for rank, hat_params in neighbor_hat_params.items():
            # recover the message and the corresponding device.
            sync_buffer["flatten_norms"].buffer = recover_device(
                sync_buffer["synced_flatten_norms"][rank],
                device=hat_params.buffer.device,
            )
            sync_buffer["flatten_updates"].buffer = self.compressor_fn.uncompress(
                recover_device(
                    sync_buffer["synced_signs"][rank], device=hat_params.buffer.device
                ),
                sync_buffer["sign_size"],
            )

            # update hat_params.
            for hat_param, norm, sign in zip(
                hat_params, sync_buffer["flatten_norms"], sync_buffer["flatten_updates"]
            ):
                # update the flatten hat params.
                hat_param.add_(norm / sign.nelement(), sign)


