
CUDA_VISIBLE_DEVICES=1 python3 main.py --dataset abilene --n_rollout_threads 10 --episode_length 100 --algorithm_name ppo --gamma 0.01 --train_size 1 --use_eval --n_path 5 --n_eval_rollout_threads 10
CUDA_VISIBLE_DEVICES=1 python3 main.py --dataset geant --n_rollout_threads 10 --episode_length 100 --algorithm_name ppo --gamma 0.01 --train_size 1 --use_eval --n_path 5 --n_eval_rollout_threads 10
CUDA_VISIBLE_DEVICES=1 python3 main.py --dataset gnnet-40 --n_rollout_threads 10 --episode_length 100 --algorithm_name ppo --gamma 0.01 --train_size 1 --use_eval --n_path 5 --n_eval_rollout_threads 10
CUDA_VISIBLE_DEVICES=1 python3 main.py --dataset germany --n_rollout_threads 10 --episode_length 100 --algorithm_name ppo --gamma 0.01 --train_size 1 --use_eval --n_path 5 --n_eval_rollout_threads 10
