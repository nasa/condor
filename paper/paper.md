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

Numerical modeling is an important part of the engineering workflow, providing understanding of engineering systems without the often prohibitive cost of fabricating physical models and prototypes. Condor is a mathematical modeling framework in Python that enables the rapid deployment of models for analysis and design. Condor uses metaprogramming to provide a mathematical domain-specific language (DSL) that reduces the software development effort needed to deploy models for solving engineering analysis and design problems. It also features a modular model-solver-backend architecture, analogous to the model-view-controller (MVC) pattern in web-application development, facitilitating usage of off-the-shelf solvers.


# Statement of Need

Condor was developed at NASA Ames Research Center's Systems Analysis Office to solve a variety of analysis and design problems in aeronautics [@Margolis2024gascon; @Listgarten2025; @Pham2025; @Park2025; @zelinski2025ttbw; @Margolis2026gascon], orbital trajectory design [@Koehler2024; @Margolis2024techniques; @Margolis2024coopt], and subsystem design [@Margolis2026bwb; @Margolis2026npss]. Condor's modular framework makes it feasible to develop algorithms using existing models as test examples like gradient methods for solutions to ordinary differential equation with events [@Margolis2023sweeping] or uncertain differential equations [@Margolis2026sigma].

A variety of existing libraries work towards unifying solvers and optimization tools under a single interface.
After assessing the available tools, we found that no existing tool provided an interface that we felt improved the engineering workflow.
Condor is unique, to the best of the authors' knowledge, for providing a mathematically-focused DSL to facilitate rapid prototyping of engineering models. Condor's modular architecture supports rapid deployment of new or existing solvers for engineering problems. These features improve the engineering workflow by providing a single engineering-focused interface to any existing solver, facilitating the creation of new models as conceptual analysis demands arise.

We built Condor to provide an interface for engineers that used computational tools from the AI/ML community as the "backend" of the framework in the Python programming language.
Using Python as the base language offers access to a huge community of practice, especially in scientific computing and numerical methods, where the majority of the AI/ML tools originated. This means that even with a small development team, we could provide features like parallel computing, file interfaces, and more, by leveraging the open-source ecosystem.


# Description

Condor is a general-purpose engineering-mathematical modeling framework used to build conceptual design and analysis capability. It is related to multi-disciplinary analysis (MDA) frameworks (e.g., Simulink [@simulink], SimuPy [@Margolis2017simupy], ModelCenter [@modelcenter], NPSS [@Curlett1995], Modelica [@Fritzson2002]), computational tools developed by the Artificial Intelligence/Machine Learning (AI/ML) community (e.g., CasADi [@Andersson2018], JAX [@Bradbury2018], Aesara [@Willard2023], PyTorch [@Paszke2017], TensorFlow [@MartinAbadi2015]), and specific numerical solvers (e.g., IPOPT [@Waechter2005] / SLSQP [@Kraft1994] for optimization, SUNDIALS [@Hindmarsh2005] for trajectory integration, etc.) The goal was to provide a single, easy-to-use interface to the best in-class numerical solvers so engineers can focus on engineering rather than learning solver-specific interfaces or spend their time with intensive coding or algorithm tuning for new models. The computational tools from the AI/ML communities took some steps to address these challenges, but their interfaces did not lend themselves to general engineering approaches to modeling physical systems. Similarly, the existing MDA frameworks did not satisfy the system analysis needs, either because they were too domain focused (e.g., dynamical systems), or because they used their own language so had limited community of practice and could not benefit from recent advancements in best practices and computational techniques.

Condor was designed and built following modern software development best-practices. We leveraged CasADi as our backend, but thanks to Condor's modular architecture, changing backends (e.g., to JAX or Aesara) would require providing a few interface files for the backend-shim infrastructure. We also leverage existing best-in-class solvers for optimization (IPOPT, other CasADi-enabled optimizers, sequential least squares SLSQP), differential equations (Runge-Kutta implementations in SciPy [@Virtanen2020; @Hairer1993], Lawrence Livermore National Lab's SUNDIALS), and efficient array-arithmetic and linear algebra from the backend. Since Condor makes it easy to wrap external solvers into the modeling framework, we have also used NASA-developed solvers like CBAero [@Kinney2007] (for hypersonic aerodynamics and aerothermal modeling), VSPAero [@Kinney2025] (for subsonic and supersonic aerodynamics modeling), and NPSS (for propulsion).

Since Condor uses standard Python data structures, we also leverage general purpose scientific tools and libraries. Virtually all utility functions come from the larger Python ecosystem:

- NumPy's [@Harris2020] built-in file format to read and store model results
- Compatible with Python parallel processing libraries such as the built-in multiprocessing or the third-party joblib [@JDT2020]  and Dask [@DDT2016] projects
- For most low-level operations (creating identity matrix, concatenating vectors, etc.), we follow the Python array API standard to reduce user and development burden
- Compatible with any plotting capability like Matplotlib [@Hunter2007], seaborn [@Waskom2021], and PyQtGraph [@Moore2023]


# References
