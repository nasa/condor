---
title: 'Condor, a mathematical modeling framework for engineers with a deadline'
tags:
  - Python
  - mathematical modeling
  - metaprogramming
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
date: 06 September 2020
bibliography: references.bib
---

# Summary

Numerical modeling is an important part of the engineering workflow,
providing understanding of enginering systems without the often
prohibitive cost of fabricating physical models and prototypes. Condor is
a mathematical modeling framework in Python that enables the rapid
deployment of models for analysis and design. Condor uses metaprogramming to provide an mathematical domain-specific language to reduce the software development effort for developing models with a modular framework making it easy to deploy off-the-shelf solvers.


# Statement of Need

Condor was developed at NASA Ames Research Center's Systems Analysis Office to solve a variety of analysis and design problems in aeronautics [@Margolis2024gascon, @Listgarten2025, @Pham2025, @Park2025, @zelinski2025ttbw, @Margolis2026gascon],  orbital trajectory design [@Koehler2024, @Margolis2024techniques, @Margolis2024coopt,], and subsystem design [@Margolis2026bwb, @Margolis2026npss]. Condor's modular framework makes it feasible to develop algorithms using exising models as test examples like gradient methods for solutions to ordinary differential equation with events [@Margolis2023sweeping] or unerctain differential equations [@Margolis2026sigma].

A variety of libraries work towards unifying solvers and optimization tools under a single interface. However, to the best of the author's knowledge no library provides such a convenient modeling language with a modular framework to leverage existing solvers.

[From SOY]
After assessing the available tools, we found that nothing met the requirements to allow engineers to have a single, easy-to-use interface to any existing solver or define new models as conceptual analysis demands arose. We built Condor to provide an interface for engineers that used computational tools from the AI/ML community as the “backend” of the framework in the Python programming language. Using Python as the base language gives access to a huge community of practice especially in scientific computing and numerical methods where the majority of the AI/ML tools originated. This means that even with a small development team, we could provide features like parallel computing, file interfaces, and more by leveraging the open-source ecosystem.


# Description

[From SOY}
Condor is a general-purpose engineering-mathematical modeling framework used to build conceptual design and analysis capability. It is related to multi-disciplinary analysis (MDA) frameworks (e.g., Simulink, SimuPy, Model Center, NPSS, Modelica), computational tools developed by the Artificial Intelligence/Machine Learning (AI/ML) community (e.g., CasADi, JAX, Aesara, PyTorch, TensorFlow), and specific numerical solvers (e.g., IPOPT/SLSQP for optimization, SUNDIALS for trajectory integration, etc). The goal was to provide a single, easy-to-use interface to the best in-class numerical solvers so engineers can focus on engineering rather than learning solver-specific interfaces or spend their time with intensive coding or algorithm tuning for new models. The computational tools from the AI/ML communities took some steps to address these challenges, but their interface did not lend themselves to general engineering approaches to modeling of physical systems. Similarly, the existing MDA frameworks did not satisfy the system analysis needs, either because they were too domain focused (e.g., dynamical systems) or because they used their own language so had limited community of practice and could not benefit from recent advancements in best practices and computational techniques. 

A major development strategy of Condor is to leverage the existing “competitors” to do the work for us. We use CasADi as our backend but thanks to Condor’s modular architecture, it would be relatively straightforward to swap with alternatives like JAX or Aesara. We also leverage existing best-in-class solvers for optimization (IPOPT, SNOPT, other casadi-enabled optimizers, sequential least squares SLSQP), differential equations (runge-kutta implementations in SciPy, Lawrence-Livermore National Lab’s Sundials), and efficient array-arithmetic and linear algebra from the backend. Since Condor makes it easy to wrap external solvers into the modeling framework, we have also used NASA-developed solvers like CBAero (for hypersonic aerodynamics and aerothermal modeling), VSPAero (for subsonic and supersonic aerodynamics modeling), and NPSS (for propulsion).

Since Condor uses standard python data structures, we also leverage general purpose scientific tools and libraries. Virtually all utility functions come from the larger python ecosystem:
- NumPy’s built-in file format to read and store model results
- Compatible with Python parallel processing libraries such as the built-in multiprocessing or the third-party joblib and DASK projects
- For most low-level operations (creating identity matrix, concatenating vectors,etc), we follow the “Array API Standard” protocol so users are essentially familiar with the API and so we are more compatible with existing libraries and potential backend interfaces.
- Compatible with any plotting capability like matplotlib, seaborne, and pqtgraph

We have also taken efforts to learn design patterns from and verify our approach with experts at NASA, particularly Dave Kinney (developer of CBAero and VSPAero) and Jeff Bowles (senior engineer known for rapidly developing engineering analysis) at Ames Research Center and Jim Felder and Tom Lavelle (part of the original NPSS architecture team) at Glenn Research Center.



# References
