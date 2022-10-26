# Experiment directory

Location described by configs["exp_dir"] as an absolute path.

Each expirment directory contains:

- Tensorboard directories:
    Loss/, CPU_Utilization/, Memory_GB/, Iteration_time/

- Configs file, specific to the training session:

    configs.json

- Output file:

    output.out

- PyTorch model file saved to disk:

    torch_model

- Training statistics needed to denormalize data set

    train_stats.json