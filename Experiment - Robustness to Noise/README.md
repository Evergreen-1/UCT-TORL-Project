# Robustness of Decision Transformers to Gaussian Noise Corruption in Offline Reinforcement Learning

This project investigates the robustness and performance of CDT, DT, and CQL when under observation and reward noise. This is achieved via the Minari dataset API, the original CDT paper by, the CORL implementations of DT and CQL.

## Usage
[1] It's recommended to create a virtual environment with the required python libraries:

        python -m venv venv
        venv\Scripts\Activate.ps1

This may vary depending on what terminal you are using, this example is for Windows Powershell. You may need to use "Set-ExecutionPolicy -Scope CurrentUser RemoteSigned" in order to activate environment.

[2] Required installs: (Use pip to install libraries)
    There must be a version of python 3.10+.
    PyTorch
        CUDA (example: 12.1):
        pip install torch --index-url https://download.pytorch.org/whl/cu121
        CPU only:
        pip install torch --index-url https://download.pytorch.org/whl/cpu

    Dependency list:
        pip install numpy tqdm wandb pyrallis "gymnasium[mujoco]" minari[all] "gymnasium[other]" wandb
        pip install matplotlib        //For DatasetVerify.py
        

[3] Additional Note:
    If "pip install minari[all]" fails then use "pip install minari" and add additional dependencies for Minari manually.
    You will have to setup your own Weights and Biases account and follow their instructions. You will have to edit the wandb.init() in run_single() if you want to log data to your own projects.

[4] How to setup experiment:
    
    Experiment flags:
    --algo          Used to specify which algorithm to run. (DT, CDT, CQL)
    --steps         Contains how many timesteps you want the algorithm to train for.
    --noise         Contains how much % of the dataset you want to put noise into. (From range 0.0 - 1.0)
    --seed          Stores and uses the seed value specified.
    --record_video  Enables video recording at end of run.
    --rew or --obs  The noise type you want to have in the dataset. If left out, then no noise will be applied.
    --full          Runs algo with all noise levels and seeds. (You need to have a --rew or --obs for there to be noise)
    --checkpoint    Loads a checkpoint file from specified directory and reruns evaluation.
    --resume        Loads a checkpoint from which you can resume training for X steps.
                    (X steps is through --steps where X is total steps you want your model to train for)
    --dataset       Used to specify which dataset you want to run. "walk", "halfch" for Walker2d and HalfCheetah respectively.
    Example run commands:
        python ExperimentA_HPC.py --algo dt --device cuda --noise 0.0 --seed 0 --steps 100000 --rew --dataset walk
        python ExperimentA_HPC.py --full --algo dt --device cuda --steps 100000 --dataset walk  --obs
        python ExperimentA_HPC.py --checkpoint folder_location/cql_noise_0.75_seed_1_rew.pt --dataset walk --device cpu  

[5] Documentation links:

        Weights & Biases:        https://wandb.ai/TORL-team/Experiment-A-Final/?nw=nwuserlckjos003
        Github:                  https://github.com/Evergreen-1/Robustness-of-Decision-Transformers-to-Gaussian-Noise-Corruption-in-Offline-RL-Hons-Project-/
