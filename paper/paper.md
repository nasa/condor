---
title: 'Condor, a mathematical modeling framework for engineers with a deadline'
tags:
  - Python
  - mathematical modeling
  - optimization
  - metaprogramming
  - domain-specific language
authors:
  - name: Benjamin W. L. Margolis
    orcid: 0000-0001-5602-1888
    affiliation: 1
  - name: Kenneth R. Lyons
    orcid: 0000-0002-9143-8459
    affiliation: 1
affiliations:
 - name:  NASA Ames Research Center, Systems Analysis Office
   index: 1
date: 07 August 2025
bibliography: references.bib
---


# Summary

Numerical modeling is an important part of the engineering workflow, providing understanding of engineering systems without the often prohibitive cost of fabricating physical models and prototypes. Condor is a mathematical modeling framework in Python that enables the rapid deployment of models for analysis and design. Condor uses metaprogramming to provide an mathematical domain-specific language (DSL) that reduces the software development effort needed to develop models. It also features a modular model-solver-backend architecture, analogous to the model-view-controller (MVC) pattern in application development, making it easy to deploy off-the-shelf solvers.


# Statement of Need

Condor was developed at NASA Ames Research Center's Systems Analysis Office to solve a variety of analysis and design problems in aeronautics [@Margolis2024gascon, @Listgarten2025, @Pham2025, @Park2025, @zelinski2025ttbw, @Margolis2026gascon], orbital trajectory design [@Koehler2024, @Margolis2024techniques, @Margolis2024coopt,], and subsystem design [@Margolis2026bwb, @Margolis2026npss]. Condor's modular framework makes it feasible to develop algorithms using exising models as test examples like gradient methods for solutions to ordinary differential equation with events [@Margolis2023sweeping] or uncertain differential equations [@Margolis2026sigma].

A variety of existing libraries work towards unifying solvers and optimization tools under a single interface,however to the best of the author's knowledge, no library provides such a convenient modeling language with a modular framework to leverage existing solvers.

After assessing the available tools, we found that nothing met the requirements to allow engineers to have a single, easy-to-use interface to any existing solver and define new models as conceptual analysis demands arose. We built Condor to provide an interface for engineers that used computational tools from the AI/ML community as the "backend" of the framework in the Python programming language. Using Python as the base language offers access to a huge community of practice, especially in scientific computing and numerical methods, where the majority of the AI/ML tools originated. This means that even with a small development team, we could provide features like parallel computing, file interfaces, and more by leveraging the open-source ecosystem.


# Description

Condor is a general-purpose engineering-mathematical modeling framework used to build conceptual design and analysis capability. It is related to multi-disciplinary analysis (MDA) frameworks (e.g. Simulink, SimuPy, Model Center, NPSS, Modelica), computational tools developed by the Artificial Intelligence/Machine Learning (AI/ML) community (e.g. CasADi, JAX, Aesara, PyTorch, TensorFlow), and specific numerical solvers (e.g., IPOPT/SLSQP for optimization, SUNDIALS for trajectory integration, etc). The goal was to provide a single, easy-to-use interface to the best in-class numerical solvers so engineers can focus on engineering rather than learning solver-specific interfaces or spend their time with intensive coding or algorithm tuning for new models. The computational tools from the AI/ML communities took some steps to address these challenges, but their interfaces did not lend themselves to general engineering approaches to modeling physical systems. Similarly, the existing MDA frameworks did not satisfy the system analysis needs, either because they were too domain focused (e.g. dynamical systems), or because they used their own language so had limited community of practice and could not benefit from recent advancements in best practices and computational techniques.

A major development strategy of Condor is to leverage the existing "competitors" to do the work for us. We use CasADi as our backend, but thanks to Condor's modular architecture, it would be relatively straightforward to swap with alternatives like JAX or Aesara. We also leverage existing best-in-class solvers for optimization (IPOPT, SNOPT, other CasADi-enabled optimizers, sequential least squares SLSQP), differential equations (Runge-Kutta implementations in SciPy, Lawrence Livermore National Lab's SUNDIALS), and efficient array-arithmetic and linear algebra from the backend. Since Condor makes it easy to wrap external solvers into the modeling framework, we have also used NASA-developed solvers like CBAero (for hypersonic aerodynamics and aerothermal modeling), VSPAero (for subsonic and supersonic aerodynamics modeling), and NPSS (for propulsion).

Since Condor uses standard python data structures, we also leverage general purpose scientific tools and libraries. Virtually all utility functions come from the larger python ecosystem:
- NumPy's built-in file format to read and store model results
- Compatible with Python parallel processing libraries such as the built-in multiprocessing or the third-party joblib and Dask projects
- For most low-level operations (creating identity matrix, concatenating vectors,etc), we follow the Python array API standard, promoting user familiarity compatibility with existing libraries potential backends.
- Compatible with any plotting capability like matplotlib, seaborn, and PyQtGraph


# References
