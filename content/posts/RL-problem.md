---
title: "Deep Reinforcement Learning: An overview"
date: 2022-06-08T12:26:35-07:00
author: "Jerry Liu"
---

# RL Problem

## Reinforcement Learning Overview

This section provides background of RL briefly. 

### Problem setup

One biggest difference between *RL* and other *machine learning* problem is that the dataset is generated in a very different way.

We first consider the components that are included in a typical RL problem:

- Agent: The agent serves as the subject of learning. It's the one that conducts the learning behavior and the receiver of the consequences. It interacts with an environment overtime.
- Action: Actions are the behavior which forms the interaction between agent and environment. The action space $A$ serves as a set of all the actions that can be taken by the agent.
- State: States model the environment. The agent typicall fires the action based on the state it's in. 
- Policy: A policy $\pi$ is normally a mapping between state $s_t$ and action $a_t$. It's modeled as a probablity distribution for the action in every single state.
- Reward: Reward is a signal generated when the agent interacts with the environment. It's the most important standard for the agent to evaluate the learning process. In another word, it seeks to maximize the expectation of accumulated reward: $max  R_t = \sum_{k=0}^\infty \gamma^k r_{t+k}$

## Prediction Problem and Control Problem

