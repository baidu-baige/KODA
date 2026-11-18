# Quickstart

[English](quickstart.md) | [简体中文](quickstart.zh-CN.md)

Choose the deployment guide that matches your environment. Both use the same agent workflow: a model generates kernel code, the sandbox compiles/evaluates/profiles it, and the feedback drives the next iteration.

| Guide | Choose this when | Requirements |
| --- | --- | --- |
| [Docker Quickstart](docker-quickstart.md) | You want to run a sandbox on one GPU host without Kubernetes | Linux, local Docker daemon, NVIDIA Container Toolkit, dedicated GPU UUIDs |
| [Kubernetes Quickstart](kubernetes-quickstart.md) | You have a GPU-enabled Kubernetes cluster | Cluster access, kubeconfig, CPU/GPU templates, registry access |

For both guides, prepare a compatible model API and dataset, then start the Sandbox Server before launching evaluation. The evaluation command creates the GPU pool and CPU sessions automatically. The model service and agent runner can run on other machines.

Docker mode is selected with `backend: docker`; omitting `backend` preserves the default Kubernetes backend. The Docker guide currently supports `cpu_gpu` sessions on one local Docker host. See each guide for configuration, validation and cleanup.

Container image addresses and private-registry login instructions are in the [README](../README.md#container-images).
