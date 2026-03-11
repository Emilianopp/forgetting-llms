
### Evals:
GPQA
RG-mix
AIME
LiveCodeBenchv6
SimpleQA
HumanEvalPlus
MBPP+
LitQA2
BFCL
Safety
Sequentially we should evaluate on backward tasks 


# setups 
Mix vs Sequential


# Experiments 
Qwen 3 1.7b 
Olmo 3 7b 


# Lora vs No Lora


# 2 permutations reversed vs not reversed & IID 



# Hyper parameters

Standard 
temp 1 
bs 256 
k (number of sampels per group) = 8
seq = 8192 


# self-distillation vs forward samples from student 

reverse-kl vs fowrard kl with teacher (low prio)


# Task distribution 

Set this all up on prime RL

* verifiers env that works with olmo dataset
* eval script for all the evals taht picks up checkpoints
* if were running on mila cluster send things to scratch
* Claude skill to automate this process
* Some joint board for us to monitor experiments


# SFT 

dataset make sure its balanced 
* Filter out impossible questions
* Make sure each solvable question has the same number of solutions
* simple hueristic 2 right questions on pass@16 
* sample using 32b version of models

