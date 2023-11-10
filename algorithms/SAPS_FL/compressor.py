import logging
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



class SAPS_FLCompressor(object):
    def __init__(self, **kargs):
        # assign compressor class.
        if "topk" in kargs["comm_op"] or "randomk" in kargs["comm_op"]:
            self.compressor_fn = SAPS_FLSparsificationCompressor(**kargs)
        elif "quantize" in kargs["comm_op"]:
            raise NotImplementedError
        elif "sign" in kargs["comm_op"]:
            raise NotImplementedError
        else:
            raise NotImplementedError

    def compress(self, *args, **kargs):
        return self.compressor_fn.compress(*args, **kargs)

    def uncompress(self, *args, **kargs):
        return self.compressor_fn.uncompress(*args, **kargs)

    def uncompress_direct(self, *args, **kargs):
        return self.compressor_fn.uncompress_direct(*args, **kargs)


class SAPS_FLSparsificationCompressor(object):
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

        for hat_param in sync_buffer["flatten_params"]:
            _selected_values, _selected_indices = self.compressor_fn.compress(
                hat_param,
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


    def uncompress(self, msg_params, weight, hat_params, selected_shapes, original_shapes, device):
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
        logging.debug("###################################")
        logging.debug("hat_params is on device: {} \n q_values is on device: {}".format(
            hat_params.buffer[q_indices].device, q_values.device
        ))
        logging.debug("###################################")


        # update the flatten hat params.
        hat_params.buffer[q_indices] += weight * q_values.to(hat_params.buffer[q_indices].device)

    def uncompress_direct(self, sync_buffer, aggregated_params, selected_shapes, original_shapes):
        """
            values:             compressed values of multiple tensors
            indices:            indices of selected values in their original tensors 
                                (means that indices are not unique)
            selected_shapes:    list of sizes of each compressed tensor
            original_shapes:    list of sizes of each original tensor
            hat_params:         need to be recovered
        """
        # deal with unbalanced values/indieces
        _, q_indices = self.compressor_fn.uncompress(
            values=sync_buffer["flatten_selected_values"].buffer,
            indices=sync_buffer["flatten_selected_indices"].buffer,
            selected_shapes=selected_shapes,
            original_shapes=original_shapes,
        )

        # update the flatten hat params.
        sync_buffer["flatten_params"].buffer[q_indices] = \
            aggregated_params.buffer[q_indices]

