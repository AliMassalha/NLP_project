import wandb
YOUR_WANDB_USERNAME = "technions"
project = "NLP2024_PROJECT_AliMassalha"
command = [
        "${ENVIRONMENT_VARIABLE}",
        "${interpreter}",
        "StrategyTransfer.py",
        "${project}",
        "${args}"
    ]
sweep_config = {
    "name": "LSTM: SimFactor=0/4 for any features representation",
    "method": "random",
    "metric": {
        "goal": "maximize",
        "name": "AUC.test.max"
    },
    "parameters": {
        "ENV_HPT_mode": {"values": [False]},
        "architecture": {"values": ["LSTM"]},
        "seed": {"values": list(range(1, 6))},
        "online_simulation_factor": {"values": [0, 4]},
        "features": {"values": ["EFs", "GPT4", "BERT"]},
        "basic_nature": {"values": [18, 19, 20, 21]},
    },
    "command": command,
}

# wandb.login()
# wandb.init(project="NLP2024_PROJECT_AliMassalha")
# Initialize a new sweep
sweep_id = wandb.sweep(sweep=sweep_config, project=project)
print("run this line to run your agent in a screen:")
print(f"screen -dmS \"sweep_agent\" wandb agent {YOUR_WANDB_USERNAME}/{project}/{sweep_id}")
