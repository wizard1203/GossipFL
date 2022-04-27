from .losses_track import losses_tracker

def create_trackers(args, **kwargs):
    """
        create the recorder to record something during training.
    """
    recorder_dict = {}
    if args.losses_track:
        recorder_dict["losses_track"] = losses_tracker(args)

    return recorder_dict
















