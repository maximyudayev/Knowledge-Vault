Created: 25-04-2022 00:21
Status: #summary 
Tags: [[Reinforcement Learning]] [[Production RL Summit 2022]] 

# Reinforcement Learning-Based Simulation of Markets
- How do you model systems where environment is made of multiple interacting agents, with policies of their own?
- Building agent-based simulators using RL.
- Behavior of an agent depends on the quantity, composition, and actions of all other agents in the environment.
- Shared policy learning shares parameters and results in stable faster learning of the optimal policy.
- By varying the agent-specific parameter, a complex behavior of the environment can be achieved quite simply.
- Instead of calculating calibration loss for the multi-agent composition, which requires convergence of learning at each iteration, use an RL calibrator agent that receive reward based on the calibation loss.
- The learning is done in two-time scale, with the calibrator learning at a slower pace than the agents to provide sufficient time for them to tend in the optimal direction such that agents can adapt to the new composition.
## References
1. [Reinforcement learning-based simulation of markets](https://www.youtube.com/watch?v=Om9A-k29GyE)