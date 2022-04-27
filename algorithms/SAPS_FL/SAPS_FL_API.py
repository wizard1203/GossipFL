import logging

from mpi4py import MPI


from utils.perf_timer_with_cuda import Perf_Timer
from utils.metrics import Metrics
from utils.logger import Logger
from fedml_core.distributed.topology.symmetric_topology_manager import SymmetricTopologyManager

from .decentralized_worker import DecentralizedWorker
from .decentralized_worker_manager import DecentralizedWorkerManager

from .SAPS_topology_manager import SAPSTopologyManager

track_time = True


def FedML_init():
    comm = MPI.COMM_WORLD
    process_id = comm.Get_rank()
    worker_number = comm.Get_size()
    return comm, process_id, worker_number


def FedML_SAPS_FL(process_id, worker_number, device, comm, model, train_data_num, train_data_global, test_data_global,
                train_data_local_num_dict, train_data_local_dict, test_data_local_dict, args, model_trainer=None):
    # args = Args(job)
    # initialize the topology (ring)
    model_trainer.set_id(process_id)

    # configure logger.
    # conf.logger = logging.Logger(conf.checkpoint_dir)

    # configure timer.
    perf_timer = Perf_Timer(
        verbosity_level=1 if track_time else 0,
        log_fn=Logger.log_timer
    )
    metrics = Metrics([1], task=args.task)

    # tpmgr = SymmetricTopologyManager(worker_number, 2)
    # tpmgr.generate_topology()

    # generate topology in the loop
    tpmgr = SAPSTopologyManager(args=args)
    tpmgr.generate_topology(t=0)
    logging.info(tpmgr.topology)

    # initialize the decentralized trainer (worker)
    client_index = process_id
    # self.topology_manager.generate_topology(t=self.global_round_idx)
    worker = DecentralizedWorker(client_index, tpmgr, train_data_global, test_data_global, train_data_num,
                train_data_local_dict, test_data_local_dict, train_data_local_num_dict, worker_number, 
                device, model, args, model_trainer, perf_timer, metrics)

    client_manager = DecentralizedWorkerManager(args, comm, process_id, worker_number, worker, tpmgr, model_trainer,
                                                perf_timer, metrics)
    client_manager.run()
