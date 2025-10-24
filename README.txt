F1TENTH PPO Agent Simulation (SB3 + f110-gym)

This project trains and runs a Proximal Policy Optimization (PPO) reinforcement learning agent to drive a simulated F1TENTH car in a basic 2D track environment, using the [f110-gym](https://github.com/f1tenth/f1tenth_gym) simulator. The agent learns to move forward while avoiding crashes using 1D LiDAR scan data as observation input.

(BE AWARE: THE PROJECT IS STILL IN DEVELOPMENT) 

 Content

-f110_rl_env.py`: Custom Gym-compatible environment wrapping the F1TENTH simulator
-train_f110_ppo.py`: Trains a PPO agent (using Stable-Baselines3)
-run_f110_ppo.py`: Runs the trained agent in simulation
-ppo_f110_agent.zip`: Trained model checkpoint
-example_map.png`: Track map used for training and evaluation
-train_pendulum.py` and `test__cartpole.py`: Preliminary SB3 training tests (Pendulum-v1 and CartPole-v1)
