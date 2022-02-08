# Metamaterial Graph Network (MGN)

This repository contains an implementation of our manuscript "Graph networks for dynamical simulation of cellular mechanical metamaterials under finite deformation". Here is an overview of the method:

<p align="center">
  <img src="https://user-images.githubusercontent.com/45647025/153072829-37ce7049-2a47-45c2-ae9a-715d35a374b9.png">
</p>


There are three major steps: 1. Collect ground truth data by performing finite element simulation via _FEniCS_; 2. Train a graph network enabled by either a neural network or a Gaussian process regression model via _JAX_; 3. Deploy the trained graph network for dynamical simulation via _JAX_.

Here are a few examples.

Longitudinal wave propagation by direct numerical simulation using the finite element method:

https://user-images.githubusercontent.com/45647025/149664118-8d47b93a-1f61-44c2-ad31-2eadb57b81aa.mp4

Longitudinal wave propagation by MGN:

https://user-images.githubusercontent.com/45647025/149663444-2922bd5b-f7e5-45e6-8333-7b399e63cf5d.mp4

Shear wave propagation by MGN for a fully connected cross-spring system:

https://user-images.githubusercontent.com/45647025/149663587-be518d22-44b2-43cb-ad32-b86dd2630361.mp4

Shear wave propagation by MGN for a partially connected cross-spring system:

https://user-images.githubusercontent.com/45647025/149663599-c5494df5-cc08-4b52-b720-d9bef802f7d0.mp4
