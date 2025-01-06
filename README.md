## Mountain car: MDP to OCP
This repository contains the implementation and formal verification of the mountain car problem, which involves transforming the Markov Decision Process (MDP) formulation into an Optimal Control Problem (OCP). 

## Project goals
1. Transform a MDP problem to a OCP using Rockit/Casadi. The original MDP problem is from here: [OpenAI Gymnasium](https://gymnasium.farama.org/main/environments/classic_control/mountain_car_continuous/).
2. Solve the OCP to determine optimal solutions.
3. Model the problem using a hybrid automaton.
4. Verify the system's safety properties.

## Links
* [Document](https://www.overleaf.com/project/66436a0ceb9f1831561bd728)
* [Interactive Code](https://colab.research.google.com/drive/1PCe_csdQoR-v4dSQG7tzR7IcOUcfPtsV?authuser=0#scrollTo=kErYfzm6xwkK)

## Repository structure
* `mountain_car_continuous.py`: Python implementation of the OCP using Rockit/CasADi.
* `D.cfg`, `D.model`, `D.xml`: Hybrid automaton files defining the system and its behavior.
* `property.yml`: Safety properties to be verified.
