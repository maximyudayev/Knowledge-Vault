Created: 24-04-2022 13:35
Status: #summary 
Tags: [[Reinforcement Learning]] [[Production RL Summit 2022]] [[Sergey Levine]]

# Production RL Summit 2022 Keynote - Offline Reinforcement Learning
- Regular ML:
	- big data + big model.
	- independent datapoints.
	- outputs do not influnce inputs.
	- availability of ground truth labels.
- Most decision-making domains do not fit into these assumptions:
	- current actions influence future observations.
	- goal is to maximize a KPI.
	- you have to figure out optimal actions.
- RL is learning-based decision making that maps observations to action decisions that have consequences which are used to reason about taret KPI maximization.
- Active online RL in a non-simulated environment is difficult to deal with in complex high-stakes environments where it can cause harm or losses for the underlying business (incurring millions in liabilities due to poor actions of a cold-start exploring/learning RL model).
- Offline RL is doing RL off of previously collected historical observation-action data under a stable policy (i.e. a hand engineering legacy control policy).
- Model collects data under a behavioral policy (any combination of decison making policies: humans, existing system, random) that it can not influence into a buffer. That buffer is used to learn the best policy possible without any extra interaction with the environment. The policy is later deployed in the environment as is, or as an Online RL model.
- Unlike online RL, after algorithm iteration, no need to recollect the data from the environment. If more data is needed, it is simply appended to the buffer.
## References
1. [Offline reinforcement learning](https://www.youtube.com/watch?v=o76FwLvsb3U)