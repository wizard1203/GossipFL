from algorithms.basePS.message_define import MyMessage


class MyMessage(MyMessage):
    """
        message type definition
    """
    # message to neighbor
    MSG_TYPE_INIT = 1
    MSG_TYPE_SEND_MSG_TO_NEIGHBOR = 2
    MSG_TYPE_METRICS = 3

    # message to coordinator
    MSG_TYPE_CLIENT_TO_COORDINATOR = 4
    MSG_TYPE_COORDINATOR_TO_CLIENT = 5

    """
        message payload keywords definition
    """

