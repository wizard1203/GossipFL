import os
import logging

import torch

from configs.chooses import EPOCH, ITERATION

def setup_losses_track_config(args):
    save_losses_track_config = {}
    # save_losses_track_config["losses_track_level"] = args.losses_track_level
    # save_losses_track_config["losses_track_client_list"] = args.losses_track_client_list
    return save_losses_track_config

class losses_tracker(object):

    def __init__(self, args=None):
        self.things_to_track = ["losses"]
        self.save_losses_track_config = setup_losses_track_config(args)

    def check_config(self, args, **kwargs):
        if args.losses_track:
            pass
        else:
            return False
        return True

    def generate_record(self, args, **kwargs):
        """ Here args means the overall args, not the *args """
        info_dict = {}
        if "losses" in kwargs:
            info_dict["losses"] = kwargs["losses"]
        logging.info('Losses TRACK::::   {}'.format(
            info_dict
        ))
        return info_dict

    def get_things_to_track(self):
        return self.things_to_track














