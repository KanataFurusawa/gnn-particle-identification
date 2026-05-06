# gnn-particle-identification
Research project on particle identification using Graph Neural Networks (GNN), focusing on K/π classification with Cherenkov detector simulation data.

## Overview
This project focuses on particle identification using Graph Neural Networks (GNN).  
The goal is to classify charged particles (Kaon / Pion) based on simulated Cherenkov detector data.

## Background
In high energy physics experiments such as TOP counter systems, distinguishing between particles like kaons and pions is an important task.  
This project explores the application of GNNs to improve classification performance.

## Method
- Graph construction from photon detection events
- Node features:
  - Momentum
  - Hit positions (x, z)
  - Detection channel (x, y)
  - Detection time
- k-nearest neighbor (k-NN) graph
- Model: Graph Neural Network (PyTorch)

## Key Contributions
- Investigated impact of feature normalization
- Improved performance by optimizing graph structure (k-NN vs fully connected)
- Analyzed factors affecting classification accuracy

## Tech Stack
- Python
- PyTorch
- NumPy

## Status
Work in progress (research project)

## Future Work
- Feature engineering improvements
- Selecting the ideal hidden layer
- Application to real experimental data
