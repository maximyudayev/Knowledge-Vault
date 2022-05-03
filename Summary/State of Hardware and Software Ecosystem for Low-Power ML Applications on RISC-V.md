Created: 28-04-2022 13:05
Status: #summary #todo
Tags: [[AI Accelerators]] [[RISC-V]] [[TinyML]]

# State of Hardware and Software Ecosystem for Low-Power ML Applications on RISC-V
## Introduction
- Introduction to the RISC-V world.
- 80% of execution happens on 6 instructions.
- Performance gain from custom instructions is marginal.
- Inflection point for alternative architectures is also a function of business value.
- Are we hitting another inflection point today?
	- Yes, real use cases force to operate under miliwatt.
	- Throughput in the millions.
	- Is RISC-V an architecture to cater for it?
- Can design without royalties, is modular, everyone can contribute to the standard.
- How to identify and characterize existing cores:
	- ISA spec
		- Instructions
			- Unpriveleged
			- Priveleged
	- Debug and trace spec
		- How an external debugger can be connected
		- Interface and encoding algorithm to get trace data
- Task group sees if there are instructions that can accelerate graphics or AI applications -> IP vendor develops a core and tapes out a chip -> IC given to software people.

RV32I at the base -> base ISA on which everything is built
RV128I (not ratified but present in silicon)
RV32E (not ratified but present in silicon)

Everything else is an extension to the base ISA.

RISC-V is simplicity in the base ISA. 39 arithmetic and logical instructions. Other stuff is added to enhance and cater for more complex use cases.

M - multiply divide, present in all silicon
F - floating point
E - atomic operations
C - some base instruction but in 16-bit instructions -> 20-25% code size reduction. Great for embedded and MCUs. Can freely mix 16-bit and 32-bit instructions. (not a problem to misalign instructions).

B - bit manipulation instructions (DSP applications)
Z* - integer registers as floating-points
V - vector instructions (SIMD on 128 bits, 300+ instructions, frozen spec). designed as base for other vector extensions for crypto an ML workloads.

N - user-level interrupts
H - supervisors
_(...) - continue note taking from slides after the event_

RISC-V Developer Board Program gives out boards to early adopters to advance software ecosystem development.

## Software
- Not mature enough.
- Libraries and Frameworks available in the ecosystem:
	- Tensorflow Lite
	- NNCN
	- Andes NN Library
- IREE (Google) High-level optimizations of ML models before passing to low-level LLVM. Operation fusion. 

SIG Graphics and ML.
December 13-14, 2022 RISC-V summit.
Join SIGs to contribute: RISC-V Labs, SIG Architecture Tests, Graphics and ML.

IR promotes reusability.
Optimization on HL and LL have trade-offs and advantages
LL targeted to microarchitectures.
Some optimizations are language-dependent.

## References
1. tinyML Talks webcast Muhammad Kamran.
talks@tinyml.com