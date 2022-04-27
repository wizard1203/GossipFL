import logging

import numpy as np

from fedml_core.distributed.topology.symmetric_topology_manager import SymmetricTopologyManager
from .minmax_commuication_cost import SAPS_gossip
from .utils import generate_bandwidth


class SAPSTopologyManager(SymmetricTopologyManager):
    """
    """

    def __init__(self, args=None):
        super().__init__(n=args.client_num_in_total, neighbor_num=1)
        # super:
        # self.n = n
        # self.neighbor_num = neighbor_num
        # self.topology = []
        self.bandwidth = generate_bandwidth(args)
        self.SAPS_gossip_match = SAPS_gossip(self.bandwidth, args.B_thres, args.T_thres)

    # override
    def generate_topology(self, t):
        match, self.real_bandwidth_threshold = self.SAPS_gossip_match.generate_match(t)
        logging.debug("match: %s" % match)
        self.topology = np.zeros([self.n, self.n])
        for i in range(self.n):
            self.topology[i][i] = 1/2
            self.topology[i][match[i]] = 1/2




