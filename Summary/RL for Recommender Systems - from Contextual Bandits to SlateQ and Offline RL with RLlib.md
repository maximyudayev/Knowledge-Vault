Created: 24-04-2022 11:01
Status: #summary 
Tags: [[Reinforcement Learning]] [[Production RL Summit 2022]] [[RLlib]] [[Ray.io]] [[Recommender System]]

# RL for Recommender Systems - from Contextual Bandits to SlateQ and Offline RL with RLlib
- Team develops [ray.io](https://docs.ray.io), one library of which is RLlib.
- Jupyter Notebooks tutorial of RLlib use.
- Agenda:
	1. Why use RL for Recommender Systems.
	2. Hands-on with RLlib and Google RecSim simulator.
	3. Offline RL and production deployment with Ray Serve.
## Introduction
### Recommender Systems
- Recommender systems personalize product recommendations to users based on what they are most likely to enjoy.
### Reinforcement Learning
- In RL, agent interacts with the environment. The goal is for agent to learn to be smarter overtime - to learn policy.
- It is not critical for the immediate reward to be the highest, but the overall cumulative sum of future rewards be the highest.
- There are no labels that indicate if an action was good or bad.
- Not guaranteed to converge because RL bootstraps on previous predictions.
- If you go to the usual place, potentially better actions will not be explored.
- Data collection step, and policy-update step. In RL, both happend at the same time.
- RL algorithms are use case agnostic because it simply looks at the interface of observations, rewards, and actions.
## Hands-on
- RecSim lets you define your own recommender system simulator by creating individual components that are composed into a ready-made OpenAI Gym environment.
## Offline RL (Batch RL) and Production Deployment
- Offline RL is often used when it is not possible to sample from the environment and to explore actions:
	- for safety reasons, to avoid the agent from performing potentially harmful actions.
	- when the users act too stochastically/unpredicatbly that it doesn't make sense to write a simulator.
	- when it is not possible to interact, learn, update on the fly.
- Learns from historic logs of observations and recommendations.
- Types of Offline RL:
	- [[Behavioral Cloning]] - copying the same policy as the one used for interacting with the environment.
	- [[Offline RL]] - slightly improving over historical actions, which requires records of action probabilities for the historic action logs.
## References
1. [RL for recommender systems: from Contextual Bandits to SlateQ and Offline RL with RLlib](https://www.youtube.com/watch?v=L5WjqPYHkqE)