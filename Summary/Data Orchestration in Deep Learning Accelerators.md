Created: 04-04-2022 10:27
Status: #summary #todo
Tags: [[AI Accelerators]] [[Books]]

## Bookmark: 3.4
# Data Orchestration in Deep Learning Accelerators
- The higher the computational efficiency, the more memory limitations are exacerbated. Therefore a lot of area is spent for buffer memories.
- Domain expertise and workload knowledge can be leveraged to optimize these accelerators.
- Lack of consistent data movement, staging and synchronization abstraction, causes a plethora of ~~incompatible~~ toolchains.
- Coupled data orchestration is request-response in load/store architectures, where the requester alternates between requesting and consuming data (master-slave idiom). Decoupled on the other hand is a duplex of producer and consumer.
- DSA do not consider heuristic cache replacement policies acceptable because of high area and enery overhead for features like tag matching and associative sets.
- The set of access patterns for data orchestration is limited in DSAs because of the workload knowledge and limited set of high-level operands.
- Buffets are recommended as the data orchestration-aware memory abstraction.
- 
## References
1. 20, 23-29 - smth
2. 33 - GPUs scratchpad explicit orchestration
3. 34-35 - ping-pong buffers
4. 36 - decoupled access-execute architecture
5. 46, 47 - auto-derivation of control FSMs for single-layer tiling specifications (non-generalized)
6. 