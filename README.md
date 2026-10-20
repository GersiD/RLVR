# RLVR
Reinforcement Learning from Verifiable Rewards

<!-- TODO: need to add discussion -->

## Layout Of Project
* `src/rlvr`: Contains the main code for the RLVR framework.
* `src/rlvr/llm.py` : Contains the implementation of the LLM (Large Language Model) used as our RL agent.
* `src/rlvr/grpo.py` : Contains the implementation of GRPO (Group Regularized Policy Optimization) algorithm.
* `src/envs`: Contains the implementations of the environments used for training and evaluation.
* `rlvr.conf` : Configuration file for the RLVR framework.

### Security Considerations
The RLVR framework described here is designed to be an MVP (Minimum Viable Product) and is not indented for production use. There are current know security vulnerabilities such as privelege escalation and arbitrary code execution enabled by the reward function found in `rewards.py`. 

Saftey considerations are left for future work and this project should be viewed as a learning tool for understanding
RLVR in practice in relatively few lines of code.

### About The Project
Studying risk averse RL I have wondered what some of the more practical applications of RL might be especially in the
age of LLMs. I have done distillation and LLM optimization in the past and was interested to see what could be done on a
small scale with an LLM as an agent. Combining these two passions of mine through this project has been a fun and
education experience.

Development was done on a Macbook Pro M2 with 32GB of RAM and 8 cores. 
