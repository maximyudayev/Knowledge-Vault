Created: 04-05-2022 17:14
Status: #summary #done 
Tags: [[Machine Learning]] [[Data Science]] [[Compute Infrastructure]]

# The Ultimate Guide to Building Scalable Machine Learning Infrastructure
Production nature and maturity of ML specific capabilities are demanded for infrastructure to achieve scalability and operationalization.

Base of the ML platform:
1. Manage;
2. Monitor;
3. Track experiments and models;

It must be built for scalability and with visibility (auditability?), with as little technical debt as possible.
Should include solutions for data management, version control and ML workbench for simple and fast experimenting, training, research, optimization.
A critical component of a scalable infrastructure is deployment.

Because of the abundance of resources and fragmentation of infrastructure, cloud or on-premise, most of the human resources are spent on configuration and resource management rather than on data science and model improvement.

2 workflows in ML systems:
1. MLOps - managing resources, infrastructure, orchestration; visualization of in-production models; integration to the rest of the IT stack.
2. Data science - data selection; data preparation; model research; running experiments; training, validation, tunning of models; deployment.

Having a scalable ML infrastructure is then having a streamline across all teams, projects, and regions.

MLOps attempts to systematize the entire development lifecycle of ML, so that engineering teams can focus more on delivering high impact ML models and not on operations surrounding it.

ML infrastructure should be compute agnostic. The same resources should be accessible to be shared and scheduled to perform diverse tasks depending on their availability and the user needs.

Containerization provide portability and flexibility of the same compute task across resource configurations (types of resources, amounts of resources, their interfaces, etc.).

You should consider how to manage all the orchestration resource from one place/interface for all your data scientists with one click. That requires designing your infrastructure or even systems with interfaces that can configure and manage the diverse resources you have: from GPU clusters, to Spark/Hadoop clusters, to individual CPUs or custom accelerators.

The design of the infrastructure should be easily extendable to keep up with the evolution of the ML models, applications, user demands, new hardware and software, etc, without reconfiguring your entire infrastructure.

Think of the interface to data scientists and design infrastructure that capitalizes on their strengths while abstracting the burden of managing the operational complexities from them. 

Goals of the infrastructure:
1. Maximize productivity of your data scientists;
2. Maximize resource utilization;

Use insights from visibility tools to guide evolution of your infrastructure.
## References
1. https://cnvrg.io/building-scalable-machine-learning-infrastructure/