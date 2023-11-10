import torch
import logging

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



class CHOCO_SGDCompressor(object):
    def __init__(self, **kargs):
        # assign compressor class.
        if "topk" in kargs["comm_op"] or "randomk" in kargs["comm_op"]:
            self.compressor_fn = CHOCO_SGDSparsificationCompressor(**kargs)
        elif "quantize" in kargs["comm_op"]:
            self.compressor_fn = CHOCO_SGDQuantizationCompressor(**kargs)
        elif "sign" in kargs["comm_op"]:
            self.compressor_fn = CHOCO_SGDSignCompressor(**kargs)
        else:
            raise NotImplementedError

    def compress(self, *args, **kargs):
        return self.compressor_fn.compress(*args, **kargs)

    def uncompress(self, *args, **kargs):
        return self.compressor_fn.uncompress(*args, **kargs)

    def uncompress_direct(self, *args, **kargs):
        return self.compressor_fn.uncompress_direct(*args, **kargs)


class CHOCO_SGDSparsificationCompressor(object):
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

        # for half_param, hat_param in zip(
        #     sync_buffer["flatten_half_params"], sync_buffer["flatten_params"]
        # ):
        for half_param, hat_param in zip(
            sync_buffer["flatten_params"], sync_buffer["flatten_hat_params"]
        ):
            _selected_values, _selected_indices = self.compressor_fn.compress(
                half_param - hat_param,
                self.comm_op,
                self.compress_ratio,
                self.is_biased,
            )
            # logging.debug("_selected_indices: {}, _selected_indices.dtype:{}".format(
            # _selected_indices, _selected_indices.dtype))
            # if (_selected_indices == 51034000).any():
            #     raise RuntimeError
            selected_values.append(_selected_values)
            selected_indices.append(_selected_indices)

        # get selected shapes.
        selected_shapes = [len(_value) for _value in selected_values]

        # flatten selected values/indices.
        flatten_selected_values = TensorBuffer(selected_values)
        flatten_selected_indices = TensorBuffer(selected_indices)
        # logging.debug("========flatten_selected_indices.buffer.dtype: {}".format(flatten_selected_indices.buffer.dtype))

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
        pass

    # def uncompress(self, msg_params, weight, neighbor_hat_params, selected_shapes, original_shapes):
    def uncompress(self, msg_params, neighbor_hat_params, selected_shapes, original_shapes):
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
        # neighbor_hat_params.buffer[q_indices] += weight * q_values
        neighbor_hat_params.buffer[q_indices.to(neighbor_hat_params.buffer.device)] += \
            q_values.to(neighbor_hat_params.buffer.device)

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
        # logging.debug("hat_params.buffer.shape:{}, q_indices: {}, q_values:{} len(q_indices):{}, len(q_values):{}".format(
        # hat_params.buffer.shape, q_indices, q_values, len(q_indices), len(q_values)))
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



#  Not complete now
class CHOCO_SGDQuantizationCompressor(object):
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
        pass

    def uncompress(self, sync_buffer, neighbor_hat_params):
        # uncompress and update.
        for rank, hat_params in neighbor_hat_params.items():
            # map the tensors to the correct location.
            _message = recover_device(
                sync_buffer["synced_message"][rank], device=hat_params.buffer.device
            )

            # update the flatten hat params.
            hat_params.buffer.add_(_message)


#  Not complete now
class CHOCO_SGDSignCompressor(object):
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
        pass

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


