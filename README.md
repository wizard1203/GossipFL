# GossipFL.


## Code Structure.

The ``algorithms'' folder includes implementations of tested algorithms.
The ``data_preprocessing'' folder includes IID dataloader and non-IID dataloder.
The ``fedml_core'' folder includes the low-level communication core.
The ``model'' folder includes the architectures of deep neural networks.
The ``experiments'' folder includes the configs of different algorithms and launch files of experiments.


## Launch Experiments.

### Defining Communication Nodes

The communication nodes are indicated by PS_MPI_HOST=$NameOfMachine1:$NumberOfClients,$NameOfMachine2:$NumberOfClients

Assuming you have 4 GPUs per machine, the GPU Mappings are indicated by PS_GPU_MAPPINGS=$NameOfMachine1:$NumberOfClientsOnGPU0,$NumberOfClientsOnGPU1,$NumberOfClientsOnGPU2,$NumberOfClientsOnGPU3;$NameOfMachine2:$NumberOfClients0,$NumberOfClientsOnGPU1,$NumberOfClientsOnGPU2,$NumberOfClientsOnGPU3.

You can find some usage examples in /experiments/mpi_based/batch_experiment_scripts_for_GossipFL.




### Running scripts
You can find some example running scripts in the /experiments/mpi_based/batch_experiment_scripts_for_GossipFL for the according to algorithms.























