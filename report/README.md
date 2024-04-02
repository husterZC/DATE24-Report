# DATE24 Conference Report



# Opening Keynote

Date: Monday, 25 March 2024
Time: 09:00 CET - 09:45 CET

## CHIPLET STANDARDS: A NEW ROUTE TO ARM-BASED CUSTOM SILICON
*Presenter*: 
Robert Dimond, ARM, GB

**Short Summary**

The talk introduces chiplets as an emerging solution on how to improve performance while managing manufacturing costs and efficiency . Chiplets are small, separate chips that can be combined to create complex systems, offering a more flexible and cost-effective alternative to traditional monolithic chip designs. This approach allows for custom silicon solutions and opens up new design possibilities. The speaker discuss two main strategies for utilizing chiplets: decomposing existing systems into multiple chiplets and aggregating peripherals into a single package. Both strategies highlight the importance of collaboration on standards for chiplet integration. The talk also cover the standards framework being developed by Arm and its partners, including specific specifications like the Arm Chiplet System Architecture (Arm CSA) and the role of industry standards such as UCIe, to facilitate this new approach in chip design.

**Notes**

- Usually, software used to lead innovation (besides technology)
- Chiplets: potential new opportunity
    - Vertical integration 
        - If you break a chip in multiple chips: HIGHER YIELD
        - In a SoC there are different federated functions that scale differently with processes
    - Optimize the process for each chiplet
    - Offer lower cost when building custom silicon. Pre-build chiplets, build system in a package at lower costs. Unluckily to be superior to fully custom integration
- Chiplets enable using different “IPs” already designed or designed by myself. Principle: maximize the reuse of the design
- Where is the cost of building a chip?
    - Software cost the most when designing custom silicon
    - firmware
    - security updates along the whole lifespan of the product
- If software's gonna cost a lot when using chiplets, we will not have an economic advantage.
    - We need compiler support and CPU architecture (system architecture) to enable low-cost chiplet development
- Arm academic access: free of royalties if you just use it for academic purposes
- How to design chiplets (technology)
    - Interconnect
    - BW connections vs connection density
    - Co design of the blocks, or integrated design
    - SERDES based
- Chiplet-based chiplets today on the market
    - AWS GRAVITON 3
        - (check -> Arm SVE2 - based!)
        - 16% energy less  than competitors
    - Nvidia Grace-hopper with HBM chiplets
- If we think about an open-market chiplet
    - Today: co-design of all the chiplets in a chip
    - Tomorrow: adopt standards for low-level protocols between chiplets
        - We need standardization
        - Trying to find sweet spots for the standards - between re-usaiblity and optimization
- How to design a chiplet-based chip?
    - Aggregation on a motherboard can be a problem
        - Protocol bridging - full-symmetric multi-processing
        - Full-coherency - real-time sync
        - Problem with chiplets today!
    - Disaggregation of an existing SoC into chiplets
        - The on-chip busses become inter-chiplets
        - For Arm - AMBA
        - Combine CPUs and other components free of charge
- The best starting point for chiplet interco 
    - Separate into layers
        - Physical layer reliably among chiplets
        - Protocol layer
        - System architecture layer (meaning to data)
- Different approaches
    - Integration on  a motherboard
        - Re-use the known protocols
        - This is also the starting point on which companies agree on
        - Maybe there will be a process of optimization
    - Disaggregation
        - Re-use protocols of the SoC we are disaggregating
        - We have AMBA for protocol layer (starting point)
        - First gap: AMBA works with wires - how to transport onto chiplets physical layer (we need to transform into packets)
            - Arm’s solution: CHI C2C architecture
            - packetized - layer on top of chiplets
            - interoperability (standard)
            - performance-sensitive - cooperation with partners using AMBA to get feedback for CHI C2C development
        - Second gap: reusability is impossible if we build on millions of different chiplets.
            - Arm’s solution: WIP early work
            - Group with +20 companies
            - Build consensus on chiplet and system types
            - Consensus on the interfaces
                - Will interrupts be wires?
                - Many arbitrary choices that need to be settled with a consensus among companies
- Final Stack for Chiplets:
    - Aggregation:
        - re-use (and optimize existing protocols)
    - Disaggregation
        - Arm’s solution to fill the gaps of the current stack



## ENLIGHTEN YOUR DESIGNS WITH PHOTONIC INTEGRATED CIRCUITS
*Presenter*:
Luc Augustin, SMART Photonics, NL

**Short Summary**

This talk delves into the significant potential of integrated photonics to address key societal challenges across various sectors including data and telecom, autonomous driving, and healthcare, focusing on cost, performance, and scalability. It draws parallels between the evolution of photonics and the semiconductor industry, emphasizing the need for platform integration in photonics to enable the creation of compact, efficient devices with diverse functionalities. A particular focus is placed on Indium Phosphide (InP) as the preferred material for long-distance communication lasers due to its proven efficacy and capacity for monolithic integration of lasers, amplifiers, modulators, and passives. This enables the development of complex Photonic Integrated Circuits (PICs). The talk also explores the transition to a foundry business model similar to that of the electronics sector, the importance of heterogenous integration with other technologies, and the role of SMART PHOTONICS as an EU-based foundry offering Process Design Kits (PDKs) to designers. This approach not only aims at advancing the photonic integration platforms but also highlights the strategic shift towards manufacturing scalability and the broader applicability of photonics in addressing the embedded system requirements of the future.

**Notes**

- Photonics to address embedded system requirements
- Photonic integration - Light use in chips to “solve problems”
- Basic elements of photonic integration
    - Polarization - Amplitude - Phase - Waveguides
    - While, in regular electronics -> Capacitor - Resistor - Transistor - Interconnect
- Use PDKs of these basic elements and then building on top
- Development stack
    - Integration
    - Manufacturing
    - Economic of scale
    - Industry structure
    - Foundry knowledge
- InP (indium phosphate) - monolithic integration of all the functionalities
    - Mature technology already!
- SMART photonics company
    - Foundry
    - Offer PDK to designers
    - EU-based
- Technology has different platforms to offer with different strengths
    - InP (more performant, simple packaging, monolithic chip)
- Hard to do non-monolithic blocks with connections between them
    - Monolithic design is better now! InP shines here
    - Si (cost per square mm, better scaling)
    - SiN
- Technology enablers
    - Performance target? We need tuneable lasers
        - Use != wavelengths to pack much information on a single wire or medium
        - To do so, we need to control precisely the wavelengths to avoid interference between != wavelengths
        - >40nm tuning range (SMART photonics)
    - MZI Modulators
        - Bring them into chips
- Increased demand for photonics in chips from companies
- We therefore need process maturity
    - PDK
        - Build complex blocks starting from basic ones
            - With the same technology, we can build sensing chips and datacom chips
        - Make it generic and reusable!
    - SMART is a new player
        - They saw a fragmented supply chain
        - Many players, all fragmented
        - They tried to defragment it with the JePPIDC EU Project
    - Need for virtual test benches 
        - Design test upfront before going to test
    - IP blocks
        - SMART offers pre-design components
    - Automated test
        - Align fibers (unique challenges to photonics)
    - Heterogeneous integration
        - Fundamental to integrate photonics and non-photonics
            - Optical IO
            - High-end electronic drivers
    - uTransfer printing on InP on Si
        - Integration at photonic level (Si-Photonics)
        - Cooperation between SMART and IMEC
        - Silicon cannot generate light but can transport it
- Where is optics used today:
    - Long distance communication: Electronics here can be fully photonics for multiplexing, demultiplexing, initial processing, etc. 
    - 5G: fiber infrastructure - mux-demux to antennas
    - Gas sensor: gas detection

<br>

## IEEE Ceda Distinguished Lecturer Lunchtime Keynote

Date: Monday, 25 March 2024
Time: 13:15 CET - 14:00 CET

Speaker and Author:
Hai (Helen) Li, Duke University, US

### AI MODELS FOR EDGE COMPUTING: HARDWARE-AWARE OPTIMIZATIONS FOR EFFICIENCY

**Notes**

- Big era of AI
- Edge devices fialed ot keep up with the pace of AI models
- Typical approach in DNN model simplification is through pruning and quantization
- **Pruning**: remove redundant parameters, up to 60% of the parameters can be removed without affecting the MSE
  - This way, a NN becomes sparse
  - Lasso regulazer is the sum over the weight parametrs, alternative to L0 norm which is less suited for gradient descends
  - How does modern HW handle sparsification? Current way is to have a xbar of DNN accelerator (when dynamics are dense) -> idea: what about handle sparsity by clustering sparse objects into dense blocks?
  - Iterative clustering concentrates the connections and minimizes the outliers
  - Structured sparsity + group LASSO: l2 norm followerd by l0 norm over gorups
  - Further improvement: L1/L2 norm as regularizer (DeepHoyer) -> auto-training effect
- **Quantization**:
  - Insight: reduce data precision from FP32 to INT4
  - Quantization scheme can be exploited with structural 1-bit sparsity
- **Communication**: for many devices used in deployment -> communication is the bottleneck
  - Distributed learning: transfer the distribution instead of the raw values
  - Federated learning (FedMask): the devices train and transmit 1-bit binary mask instead of the whole 32-bit weights of the model

<br>

## Special Day on Sustainable Computing Lunchtime Keynote

Date: Tuesday, 26 March 2024
Time: 13:15 CET - 14:00 CET

Speaker and Author:
Ayse Coskun, Boston University, US

### DATA CENTER DEMAND RESPONSE FOR SUSTAINABLE COMPUTING: MYTH OR OPPORTUNITY?

**Notes**

- Energy efficiency improve is a today trend, but: is it enough?
- "Performance per watt is the new Moore's Law"
- Data centers is an ecosystem that interacts with users and electricity grids (a system on its own)
- An electricity grid is where we get our power from.
- Match between power demand and requests to change demand
  - Better management of peak demand
- Data centers can be the actors to handle the demand response
- Provider dictates power regulation targets to the data centers, which follows it
  - Example: Google announced DR pilots in Europe (shifting non-urgent compute)
  - LLNL provided 8MW of flexibility
- Figure of merits: QoS degradation as the % of degradation in execution time from the max speed
- Runtime jobs can play with several parameters to control the servers
  - Optimization problem with a cost function that meet constraints
    - Like this you optimize more threads with a lot of load
    - Transformer model to learn the relation between parameters, decresing convergence time

<br><br><br>

# Technical Sessions

## Architectural And Microarchitectural Design Solutions

**Date**: Tuesday, 26 March 2024 
**Time**: 16:30 CET - 18:00 CET

### ViTA: A Highly Efficient Dataflow and Architecture for Vision Transformers

**Summary**

This work underscores the challenge of deploying large-scale Transformer-based models, such as GPT-4, due to their compute- and data-intensive nature, particularly highlighting the efficiency issues faced in both matrix multiplication and non-linear function operations which are core components of these models. To address these challenges, especially for Vision Transformer (ViT) models, the paper proposes ViTA, a scalable architecture that focuses on a memory-centric, hardware-efficient dataflow design to reduce memory usage and data transfer significantly. ViTA demonstrates remarkable improvements in area and power efficiency, outperforming state-of-the-art Transformer accelerators by substantial margins. The paper’s contributions include the development of a scalable architecture for ViT workloads, the introduction of a novel fused module for optimizing non-linear function operations, a comprehensive design space exploration to balance area and power trade-offs, and a comparative performance analysis with existing Transformer accelerators.

**Proposal Highlights:**

- ViTA Architecture: Introduces a scalable architecture designed to efficiently accelerate the entire Vision Transformer (ViT) workload, focusing on reducing memory usage and data transfer.
- Memory-Centric Dataflow: Proposes a memory-centric, hardware-efficient dataflow to address the inefficiencies in handling matrix multiplication and non-linear function operations in Transformer models.
- Fused Special Function Module: Develops a novel module for non-linear functions in ViT models, enhancing resource sharing and significantly improving area and power efficiency metrics.
- Design Space Exploration: Conducts a thorough design space exploration of ViTA Kernels and VMUs (Vector Memory Units) to facilitate optimal area and power efficiency trade-offs.
- Performance Benchmarking: Demonstrates ViTA's superior performance, achieving 16.384 TOPS with remarkable area efficiency (2.13 TOPS/mm²) and power efficiency (1.57 TOPS/W) at 1 GHz in a 28 nm technology node, substantially outperforming existing Transformer accelerators in both metrics.

![](figures/Architectural%20And%20Microarchitectural%20Design%20Solutions/1b.png)
![](./figures/Architectural%20And%20Microarchitectural%20Design%20Solutions/1c.png)

### Sava: A Spatial- and Value-Aware Accelerator for Point Cloud Transformer

**Summary**

This work addresses the challenge of achieving real-time processing with point cloud transformers, which combine traditional Point Cloud Neural Networks (PCNNs) and transformers for tasks like autonomous driving and augmented reality. Despite their superior accuracy, existing PCNN accelerators focus mainly on the front-end, neglecting the back-end transformers and thus failing to optimize the entire network. To address this, the paper introduces a novel spatial-and-value-aware hybrid pruning strategy for the attention mechanism, reducing redundant computations and enabling hardware acceleration by converting dense patterns to sparse modes. The proposed Sava architecture leverages this pruning strategy, employing a reconfigurable mixed-precision systolic array for efficient computation and introducing mechanisms to handle workload imbalance in sparse computations. The contributions include the hybrid pruning strategy that combines spatial information and value importance to minimize computational redundancy without significant accuracy loss, the innovative Sava architecture for accelerating point cloud transformers, and the demonstration of Sava's superior accuracy and performance through extensive evaluations.

**Proposal Highlights:**

- Hybrid Pruning Strategy: Introduces a novel pruning strategy that combines spatial information from the front-end and value importance from the back-end of point cloud transformers, optimizing computation by reducing redundancy.
- Sava Architecture: Proposes the first accelerator specifically designed for point cloud transformers, featuring a reconfigurable mixed-precision systolic array and a data rearrangement mechanism to efficiently handle sparse computations.
- Spatial-and-Value-Aware Pruning: Utilizes a dual pruning approach to identify and eliminate less important computations, thereby transforming dense patterns into sparse modes for enhanced hardware acceleration.
- Reconfigurable Mixed-Precision Computation: Employs a systolic array that adjusts precision levels for different computational needs, balancing accuracy and speed in the feature computation and attention mask generation.
- Data Rearrangement Mechanism: Introduces a mechanism to pack rows with similar sparsity levels, optimizing resource utilization and minimizing latency in processing unstructured sparse data.
- Superior Accuracy and Performance: Demonstrates through extensive experiments that Sava achieves notable improvements in accuracy and performance over existing PCNN accelerators, validating the effectiveness of the hybrid pruning strategy and the architectural innovations.



### Efficient Design of a Hyperdimensional Processing Unit for Multi-Layer Cognition

**Summary**

This work discusses the importance of hierarchical cognition in developing dynamic intelligent systems, particularly through integrating sensory perception, reasoning, and control into a unified computational model. This model aims to benefit various applications, such as intelligent mobile robots and adaptive prosthetic systems. The paper identifies the challenge of combining traditional neural networks for perception and logic programming for reasoning into a seamless framework. Hyperdimensional Computing (HDC) is introduced as a solution to bridge this gap, offering a new paradigm where computational elements are represented by high-dimensional holographic vectors that can be manipulated algebraically to implement cognitive functions. Despite HDC's advantages in robustness and efficiency, its high-dimensional nature presents challenges for implementation on classical compute platforms like GPUs, due to inefficiencies in handling wide vectors and lack of optimization for on-chip dimensionality scaling. To address these challenges and the limitations of existing HDC processors, the paper proposes the design of the HPU (Hyperdimensional Processing Unit), the first general-purpose HDC processor capable of implementing multi-layer cognition with configurable performance trade-offs. The contributions include a structured design method for transforming HDC-based multi-layer cognition specifications into kernels and dataflow representations and an evaluation of HPU's hardware design and control framework.

**Proposal Highlights:**

- Unified Framework for Hierarchical Cognition: Proposes integrating sensory perception, reasoning, and control into a unified computational model for dynamic intelligent systems.
- Hyperdimensional Computing (HDC): Introduces HDC as a solution to seamlessly combine neural networks and logic programming by representing computational elements as high-dimensional vectors.
- Challenges with Classical Compute Platforms: Highlights the inefficiencies of implementing HDC on traditional computing platforms due to their inability to efficiently manage high-dimensional data.
- HPU Design: Presents the design of the Hyperdimensional Processing Unit (HPU), the first general-purpose HDC processor that supports multi-layer cognition with extensive configurability.
- Structured Design Method: Proposes a structured design method to translate HDC-based cognition specifications into practical kernels and dataflow representations for hardware implementation.
- Configurable Performance: Describes HPU's control framework, which allows tuning performance trade-offs, addressing the need for a new processor architecture tailored for HDC functions.


### EcoFlex-HDP: High-Speed and Low-Power and Programmable Hyperdimensional-Computing Platform with CPU Co-processing

**Summary**

This work outlines the advantages and challenges of Hyperdimensional Computing (HDC), a computing paradigm inspired by the brain's information processing capabilities, which uses high-dimensional vectors for data representation and classification. HDC's computational efficiency, parallelization capabilities, and performance in cognitive tasks across various fields are highlighted. However, challenges arise from the specialized arithmetic operations HDC requires, which are inefficient on general-purpose CPUs and problematic in energy-efficient devices without suitable vector computation mechanisms. To address these challenges, the paper introduces EcoFlex-HDP, a computational platform that enhances processing speed and energy efficiency while ensuring compatibility with existing software and adaptability to new algorithms. EcoFlex-HDP integrates a Hyperdimensional Processing Unit (HPU) with a CPU to enable efficient HDC operations. The contributions include the design of an HPU architecture optimized for HDC, the development of EcoFlex-HDP for flexible task execution, a software stack facilitating CPU-HPU collaboration, and the demonstration of EcoFlex-HDP's improved energy–delay product (EDP) compared to commercial CPUs, showcasing significant advancements in energy efficiency and processing speed for HDC applications.

**Proposal Highlights:**

- Hyperdimensional Computing (HDC): Outlines HDC's efficiency in mimicking the brain's capability to process multidimensional information, making it suitable for classification tasks across various fields.
- Challenges in Implementing HDC: Discusses the inefficiencies of executing specialized HDC operations on general-purpose CPUs and the need for a dedicated computational mechanism for HDC.
- EcoFlex-HDP Platform: Introduces a computational platform that combines the efficiency of an HPU with a CPU, optimizing processing speed and energy consumption for HDC tasks.
- HPU Architecture: Describes the design of an HPU architecture with instructions optimized for HDC operations, allowing for more efficient computation than traditional CPUs.
- CPU and HPU Collaboration: Establishes a computational environment where the CPU and HPU work in sync, enabling flexible and efficient execution of tasks.
- Software Stack for CPU-HPU Integration: Develops a software stack that eases the integration of existing software with HDC operations, facilitating task sharing between the CPU and HPU.
- Validation of EcoFlex-HDP: Demonstrates EcoFlex-HDP's ability to integrate with existing software and adapt to new algorithms through collaborative execution of various recognition tasks.
- Performance Evaluation: Evaluates EcoFlex-HDP on an FPGA platform, showing a significant improvement in the energy–delay product (EDP) compared to commercial CPUs, highlighting its potential in HDC-enabled software development and the growth of HDC ecosystems.

### Accelerating Chaining in Genomic Analysis Using RISC-V Custom Instructions

**Summary**

This work highlights the pivotal role of DNA sequence analysis in fields like medicine and agriculture, emphasizing Minimap2 as the leading software tool for third-generation DNA sequence analysis. Despite Minimap2's effectiveness, its compute-intensive nature traditionally confines it to high-performance computing systems. Recent initiatives aim to run DNA analysis on embedded systems for enhanced portability and energy efficiency, particularly valuable in remote areas. However, the performance on embedded platforms, such as ARM Cortex-A57, substantially lags behind HPC systems, with the chaining stage identified as a bottleneck in embedded processor executions. To address this, the paper proposes accelerating the chaining stage of Minimap2 on RISC-V processors using custom instructions, leveraging RISC-V’s extensibility for embedded systems. The novel contributions include the acceleration of the chaining step on a RISC-V based ASIP with custom instructions, improving speed by up to 2.4× without compromising mapping accuracy; the introduction of a novel architectural template for complex custom instructions; a heuristic algorithm for mapping C code segments to this template; and the implementation of these custom instructions in HDL on a Rocket Chip configured on a Xilinx Zynq UltraScale+ FPGA.

**Proposal Highlights:**

- Acceleration of DNA Sequence Analysis: Focuses on enhancing the speed of Minimap2's chaining stage on embedded systems through custom RISC-V instructions.
- RISC-V Custom Instructions: Proposes a novel approach by introducing custom instructions on a RISC-V processor to improve the performance of DNA analysis software.
- Performance Improvement: Achieves up to 2.4× speedup for the chaining step of Minimap2 on an ASIP without losing final mapping accuracy.
- Novel Architectural Template: Introduces a new template for designing custom instructions that can accommodate more complex operations beyond the standard two-input limit.
- Heuristic Algorithm for C Code Mapping: Presents an algorithm to efficiently map segments of C code to the newly designed architectural template for custom instructions.
- Implementation on FPGA: Implements the extracted custom instructions from Minimap2's chaining step in HDL on a Rocket Chip configured on Xilinx Zynq UltraScale+ FPGA, demonstrating the practicality and effectiveness of the proposed solution.


### ONE-SA: Enabling Nonlinear Operations in Systolic Arrays For Efficient and Flexible Neural Network Inference

**Summary**

This work outlines the challenges associated with deploying deep neural networks (DNNs) due to their significant computational demands and high power consumption, particularly on resource-constrained platforms. While ASIC accelerators offer a solution by streamlining computations through specialized units for linear and nonlinear operations, their inherent specificity often limits their flexibility across different DNN models, leading to performance inefficiencies. This paper proposes ONE-SA, a novel systolic array architecture designed to support a broad spectrum of nonlinear computations alongside linear computations, providing a versatile and efficient computing unit for DNNs. ONE-SA addresses the common limitations of ASIC-based accelerators by maintaining continuous computation and avoiding idle periods, demonstrating computational and power efficiency on par with application-specific accelerators while retaining the flexibility to accommodate various DNN models. The contributions include the introduction of a systolic array architecture for nonlinear operations, the design of micro-architectures for function units, and FPGA implementation results showing ONE-SA's efficiency and flexibility in executing diverse DNN models.

**Proposal Highlights:**

- ONE-SA Architecture: Introduces a systolic array architecture that supports a wide range of nonlinear computations, enabling seamless execution of versatile DNN computations through a single computing unit.
- Nonlinear Operations in Systolic Arrays: Facilitates nonlinear operations through capped piecewise linearization and computation by Matrix Hadamard Products (MHPs) with intermediately fetched parameters.
- Micro-Architectures Design: Details the meticulous design of function units, including the processing element and L3 buffer, balancing between computation performance and hardware resource cost.
- Computational and Power Efficiency: Demonstrates that ONE-SA achieves levels of computational and power efficiency previously exclusive to state-of-the-art ASIC-based accelerators, while providing flexibility for a wide range of DNN models.
- FPGA Implementation and Evaluation: Showcases FPGA implementation results, indicating the efficiency and adaptability of ONE-SA in executing diverse DNN models.

### DeepFrack: A Comprehensive Framework for Layer Fusion, Face Tiling, and Efficient Mapping in DNN Hardware Accelerators

**Summary**

This work highlights the significant advancements in Machine Learning (ML) and the increasing demand for high-throughput, energy-efficient hardware accelerators, especially in the context of Deep Neural Networks (DNNs). Given the slower global energy production rates compared to computational demands, there's a critical need for frequent redesigns of ML accelerators to adapt to the evolving nature of ML algorithms and their application in power-limited devices. The paper addresses the challenge of optimally fusing DNN layers to reduce energy consumption, particularly in DRAM communication, a major bottleneck in current hardware accelerators. Existing tools fall short in optimally fusing workloads and deciding the best partitioning strategy for these fused layers. The proposed solution, DeepFrack, is an algorithmic framework designed to identify the optimal layer fusion strategy and data flow for execution in fused layer sets, catering to various layer types. Through experiments, DeepFrack demonstrates significant reductions in energy consumption and latency on state-of-the-art DNN accelerators by optimally scheduling fused layers compared to processing layers individually. The framework is made available as an open-source tool, providing a resource for optimizing DNN accelerator designs.

**Proposal Highlights:**

- Addressing ML Hardware Accelerator Design Challenges: Emphasizes the need for high-throughput, energy-efficient hardware accelerators in ML, particularly for DNNs, due to increasing operational demands and global energy constraints.
- Optimal Layer Fusion Strategy: Introduces DeepFrack, an algorithmic framework to find the best layer fusion strategy and data flow for execution in fused layer sets, aiming to reduce DRAM communication energy.
- Support for Various Layer Types: DeepFrack supports a wide range of DNN layer types, including 2D and 3D convolution, ReLU, and pooling layers.
- Significant Energy and Latency Reductions: Demonstrates that DeepFrack can reduce energy consumption and latency by 20% to 60% on state-of-the-art DNN accelerators through optimal scheduling of fused layers.
- Case Studies on Simba and Eyeriss Architectures: Includes case studies on the application of DeepFrack to the Simba and Eyeriss DNN accelerator architectures.
- Open-Source Tool Availability: Makes DeepFrack available as an open-source framework for the community, encouraging further research and optimization in DNN accelerator design. Link: [https://github.com/TomGlint/DeepFrack](https://github.com/TomGlint/DeepFrack)


### S-LGCN: Software-hardware co-design for accelerating LightGCN

This work emphasizes the significant role of Graph Convolutional Neural Networks (GCNs) in learning intricate node features within graph structures, highlighting LightGCN as a notable model for personalized recommendation and molecular property prediction. Despite its successes, LightGCN faces challenges in layer combination parameter optimization and time-consuming inference, particularly in scenarios with linear or polynomial growth interactions. To address these challenges, the paper proposes algorithmic optimizations through Q-learning to refine layer connection parameters and incorporates hardware-friendly activation functions. Furthermore, to tackle the inference latency, a specialized FPGA-based architecture, S-LGCN, is introduced, leveraging High Bandwidth Memory (HBM), efficient data compression, and parallel computation strategies to accelerate the inference phase of LightGCN. This approach not only improves performance on various datasets but also significantly enhances inference speed, making LightGCN more adaptable to real-time tasks. The contributions include the optimization of LightGCN for hardware efficiency, the design of the S-LGCN accelerator for rapid inference, and the substantial performance improvement over traditional CPU and GPU implementations, establishing S-LGCN as the first hardware accelerator for LightGCN.

**Proposal Highlights:**

- Algorithmic Optimizations for LightGCN: Utilizes Q-learning to optimize layer connection parameters and introduces nonlinear functions at the output stage of each layer to enhance the model's performance.
- S-LGCN Architecture: Proposes an FPGA-based accelerator that includes efficient data storage and access, a new data compression format, and three levels of parallel computation to address the inference speed bottleneck.
- Performance Improvement: Achieves a significant increase in inference speed, making LightGCN more suitable for real-time applications with a performance improvement of 1576.4x and 21.8x over CPUs and GPUs, respectively.
- Adaptability to Real-time Tasks: Demonstrates S-LGCN's capability to significantly reduce the time-consuming phases of LightGCN, addressing real-world demands for rapid processing.
- First Hardware Accelerator for LightGCN: Introduces S-LGCN as the pioneering hardware accelerator designed specifically for the LightGCN model, targeting improvements in both algorithmic efficiency and hardware performance.


<br>

## FPGA Solution
Date: Monday, 25 March 2024
Time: 11:00 CET - 12:30 CET

### Unveiling the Black-Box: Leveraging Explainable AI for FPGA Design Space Optimization

**Summary**
This work discusses the increasing popularity and potential of Field-programmable gate arrays (FPGAs) due to their reconfigurability, high performance, and cost efficiency. However, it highlights a significant challenge in the FPGA design process: the complexity and large design space of synthesis, place, and route (SP&R) decisions, exacerbated by the unpredictable behavior of electronic design automation (EDA) tools. To address these challenges, the paper introduces a novel approach utilizing explainable artificial intelligence (XAI) to refine the design space by determining effective tool parameters. This approach aims to interpret the "chaotic behavior" of EDA tools, enabling a more efficient optimization process. The study focuses on timing optimization for FPGA-based designs and claims to improve the maximum operating frequency of test designs by an average of 26% while also achieving faster convergence with fewer FPGA compilations. The main contributions include a framework for refining the design space for timing optimization, a two-stage contribution score estimation for tool parameters using XAI, and the demonstration of improved optimization performance.

**Proposal Highlights:**

- The study proposes a novel approach to refine the FPGA design space using explainable artificial intelligence (XAI), aiming to tackle the challenges of complexity and unpredictability in electronic design automation (EDA) tools.
- It introduces a framework that refines the design space specifically for FPGA timing optimization, making it the first of its kind to focus on optimizing the impact of various tool parameters on the final timing results.
- A two-stage contribution score estimation method is implemented to assess the impact of each tool parameter on the optimization process. This method leverages XAI to explain the correlation between input features and the output of machine learning-based predictions.
- By utilizing the refined design space generated through this framework, the study achieved an average improvement of 26% in the maximum operating frequency across six test designs. Additionally, it demonstrated faster optimization convergence with fewer FPGA compilations required.


### An Agile Deploying Approach for Large-Scale Workloads on CGRA-CPU Architecture 

**Summary**
The work addresses the growing demand for computational power, flexibility, and parallelism for large-scale workloads in areas like Deep Learning (DL) and High-Performance Computation (HPC). It highlights Coarse-Grained Reconfigurable Architectures (CGRAs) as a suitable solution due to their ability to adapt configurations for diverse computational tasks efficiently. However, the challenge lies in the heterogeneous nature of CGRA-CPU systems, especially in compiler design to handle diverse kernel types and large kernel sizes. To overcome these challenges, the paper proposes CGRV-OPT, an MLIR-Based optimizing and deploying framework for large-scale data-flow applications, targeting CGRA and RISC-V heterogeneous architecture. This framework aims to enhance the utilization of System on Chip (SoC) through automated optimization techniques and supports various high-level programming models. It introduces an open-source compilation framework, optimizations for computing and memory access, and an automated hardware/software partitioning explorer. The results show that CGRV-OPT achieves a significant speedup in computing efficiency over exclusive CPU execution.

**Proposal Highlights:**

- CGRV-OPT: An end-to-end, MLIR-Based compilation framework designed to optimize and deploy applications on CGRA-CPU SoCs, enhancing computing efficiency for large-scale data-flow applications.
- The framework is open-source, supports multiple high-level programming models, and aims to automate the optimization process for applications targeting heterogeneous architectures.
- Introduces a series of Intermediate Representation (IR) transformations and optimizations that incorporate hardware-specific information to improve computing and memory access efficiency.
- Features an automated hardware/software partitioning explorer that efficiently maps different kernels to the CPU or CGRA based on performance models, optimizing workload distribution.
- Demonstrates a significant improvement in performance, with CGRV-OPT achieving an average speedup of 2.14× over exclusive CPU execution for heterogeneous architectures.



### Cuper: Customized Dataflow and Perceptual Decoding for Sparse Matrix-Vector Multiplication on HBM-Equipped FPGAs


**Summary**

This work discusses the importance of Sparse matrix-vector multiplication (SpMV) in various fields such as scientific computing, engineering, graph computation, and cloud computing, emphasizing its performance impact. FPGAs are identified as attractive platforms for accelerating SpMV due to their ability to customize data flow and memory structure, along with their low power consumption. However, challenges such as limited memory channels and irregular memory access patterns hinder the full exploitation of FPGAs. The paper introduces a high-performance SpMV accelerator named Cuper, designed for HBM-equipped FPGAs, to overcome these challenges. Cuper incorporates a custom sparse storage format, a two-step reordering algorithm, and a perceptual decoder to improve bandwidth utilization, mitigate read-after-write conflicts, and enhance vector reusability and on-chip memory utilization. The evaluation of Cuper on a Xilinx Alveo U280 FPGA shows it outperforms existing solutions like HiSparse, GraphLily, Sextans, and Serpens in throughput, bandwidth efficiency, and energy efficiency. Comparisons with an Nvidia Tesla K80 GPU on SuiteSparse matrices also demonstrate superior performance.

**Proposal Highlights:**

- Custom Sparse Storage Format: Cuper employs a newly proposed sparse storage format to customize the dataflow for HBM, supporting vectorized and streaming accesses to enhance bandwidth utilization.
- Two-step Reordering Algorithm: A novel algorithm that combines conflict-aware row reordering and reuse-aware column reordering to reduce read-after-write (RAW) conflicts and improve vector reusability.
- Perceptual Decoder Design: Innovates with a perceptual decoder that boosts vector reusability and on-chip memory utilization through a flexible reuse register design and skipping redundant vector writes.
- Superior Performance: Implemented on a Xilinx Alveo U280 FPGA, Cuper demonstrates significant improvements over existing FPGA accelerators and outperforms an Nvidia Tesla K80 GPU in throughput, bandwidth efficiency, and energy efficiency when evaluated on a large set of SuiteSparse matrices.



### Towards High-throughput Neural Network Inference with Computational BRAM on Nonvolatile FPGAs

**Summary**

This work highlights the increasing relevance of Field-Programmable Gate Arrays (FPGAs) in artificial intelligence (AI) and big data applications due to their flexibility and energy efficiency. As AI models evolve, the demand for computation and memory resources on FPGAs grows, prompting a shift from traditional SRAM-based FPGAs to those incorporating emerging nonvolatile memories (NVMs) for improved performance, energy efficiency, and device lifetime. This paper proposes leveraging the computation-in-memory (CiM) capability of NVMs to enhance the architecture and synthesis flow of nonvolatile FPGAs (NV-FPGAs). By introducing a computational function into BRAMs, termed computational BRAM (C-BRAM), and developing an operator allocation strategy, the paper aims to utilize the high computing parallelism of C-BRAMs effectively. This approach is demonstrated through neural network inference applications, showcasing the benefits over existing NV-FPGA architectures. The paper claims to be the first to explore computational BRAM within NV-FPGAs, contributing to reduced area overhead and maximized computational density.

**Proposal Highlights:**

- Introduction of CiM into NV-FPGA: Integrating computation-in-memory (CiM) architecture into NV-FPGAs using NVM-based technology to enhance computational efficiency and reduce area overhead.
- Computational BRAM (C-BRAM) Design: Proposing a novel architecture that adds computational functions to BRAMs within NV-FPGAs, aiming to leverage the high parallelism available in these memory arrays for computation.
- Operator Allocation Strategy: Developing a computational density-aware operator allocation strategy that selects the most appropriate resource type for different operations, fully exploiting the computational potential of C-BRAM.
- Synthesis Tool-Chain Integration: Incorporating the proposed architecture and strategy into an open-source FPGA synthesis tool-chain, enabling evaluation and demonstrating the practical benefits of the approach for neural network inference applications.




### On-FPGA Spiking Neural Networks for Integrated Near-Sensor ECG Analysis

**Summary**

The paper addresses the challenge of real-time monitoring of heart function, particularly focusing on arrhythmia recognition using electrocardiograms (ECG), a crucial but resource-constrained task in the wearable domain. Spiking Neural Networks (SNNs) are identified as a promising solution for their energy-efficient, event-based processing capabilities. However, this necessitates dedicated computational architectures, where Field Programmable Gate Arrays (FPGAs) are ideal due to their configurability, allowing for efficient processing by targeting only active neurons. The study explores the implementation of an SNN-based classification system on the Lattice iCE40-UltraPlus FPGA, achieving state-of-the-art accuracy for arrhythmia detection. This includes an effective encoding method translating ECG signals into spike traces and an efficient FPGA implementation that significantly lowers energy consumption and inference time compared to existing FPGA-based solutions.

**Proposal Highlights:**

- SNN-Based Arrhythmia Detection: Implementation of an SNN-based system for the accurate detection of arrhythmia, achieving accuracy levels comparable to the state-of-the-art in SNN applications.
- Effective Encoding Method: Exploration of delta modulation on original ECG signals and their derivatives as an encoding method to generate spike traces, resulting in a performance gain.
- Ultra-Low-Power FPGA Implementation: Efficient deployment of the system on a Lattice iCE40-UltraPlus FPGA, demonstrating a significant improvement in inference time and energy consumption, achieving over 6× energy efficiency improvement compared to existing FPGA-based alternatives.





<br>

## New Circuits and Devices

Date: Monday, 25 March 2024
Time: 14:00 CET - 15:30 CET

### Dynamic Realization of Multiple Control Toffoli Gate

**Summary**

In the context of the Noisy Intermediate Scale Quantum (NISQ) era, a major challenge is mapping quantum algorithms to noisy quantum computers efficiently, particularly with fewer qubits. Dynamic Quantum Circuits (DQC) have emerged as a solution, enabling the realization of quantum algorithms with significantly fewer qubits by allowing n data qubits and m answer qubits to be operated using only m + 1 qubits. This advancement not only makes fault-tolerant computation more feasible but also reduces mapping overhead. The paper focuses on the dynamic realization of Multiple Control Toffoli (MCT) gates, which are crucial for quantum algorithms. It proposes two schemes for MCT gate transformation: one using an existing ancilla-free decomposition and another with a new decomposition optimized for DQCs. The study compares these approaches based on resource constraints and evaluates the fidelity using the Deutsch-Jozsa algorithm. Experimental results indicate that the new decomposition method offers better computational accuracy, albeit requiring additional gates.

**Proposal Highlights:**

- Efficient Mapping of Quantum Algorithms: Introduction of Dynamic Quantum Circuits (DQC) as a solution for efficient mapping of quantum algorithms to NISQ era hardware, using fewer qubits.
- Dynamic Realization of MCT Gates: Exploration and proposal of two schemes for the dynamic realization of MCT gates, crucial for enhancing quantum computation.
- Ancilla-Free Decomposition Method: Utilization of an existing ancilla-free decomposition structure for DQC transformation, ensuring efficient computation without additional qubits.
- New Decomposition for DQCs: Introduction of a novel decomposition method specifically designed for DQCs, aimed at improving computational accuracy.
- Comparative Evaluation: Comparison of both decomposition methods in terms of resource constraints (gate, depth, nearest neighbor overhead) and computational accuracy using the Deutsch-Jozsa algorithm.
- Enhanced Computational Accuracy: Experimental results showing that the new decomposition method provides better computational accuracy, with a trade-off of additional gates required.


### A FeFET-based Time-Domain Associative Memory for Multi-bit Similarity Computation

**Summary**

This work outlines the significance of similarity computation (SC) across various fields, emphasizing its role in machine learning, bioinformatics, and database management. Traditional in-memory computing (IMC) designs, while addressing the inefficiencies of Von Neumann architectures, face challenges related to analog-digital signal conversion, static power consumption, and precision-versus-efficiency trade-offs. Time-domain (TD) computing emerges as a solution, offering digital compatibility, robustness against variations, and efficiency in quantitative SC. However, existing TD-IMC designs, particularly those based on SRAM and early-stage NVM technologies, encounter limitations in area cost and lack comprehensive robustness analysis. The paper proposes a FeFET-based TD associative memory (AM) for quantitative SC, employing a novel variable-capacitance delay chain structure that enhances robustness and maintains a linear relationship between delay and computing results. This design aims at energy-constrained AI applications, with innovations in hardware utilization, multi-level storage, and computing. SPICE simulations and Monte Carlo analysis validate the design's energy efficiency and robustness, and benchmarking in hyperdimensional computing tasks demonstrates significant improvements in speed and energy efficiency.

**Proposal Highlights:**

- Time-Domain (TD) Computing for SC: Introduces TD computing as a promising paradigm for addressing the challenges of analog IMC designs in similarity computation.
- FeFET-Based TD Associative Memory (AM): Proposes a novel FeFET-based TD-AM design optimized for quantitative SC, particularly focusing on energy-constrained AI scenarios.
- Variable-Capacitance Delay Chain: Implements a unique variable-capacitance delay chain structure, improving robustness against variations and ensuring accuracy in SC.
- 2-Step Computation Principle: Introduces a computation principle that maximizes hardware utilization and reduces latency, enhancing overall computational efficiency.
- Multi-Level Storage and Computing: Leverages multi-domain FeFET devices for multi-level storage and computing, enabling higher precision and efficiency in data processing.


### RVCE-FAL: A RISC-V Scalar-Vector Custom Extension for Faster FALCON Digital Signature

**Summary**

This work outlines the vulnerabilities of existing public-key cryptography (PKC) infrastructures to quantum computer attacks, highlighting the necessity for post-quantum cryptography (PQC) standards. With the National Institute of Standards and Technology (NIST) announcing finalists for PQC algorithms, the FALCON digital signature scheme emerges as a promising candidate, especially for Internet of Things (IoT) applications due to its fast verification speed and low communication overhead. However, existing hardware implementations focus primarily on accelerating signature verification without optimizing signature generation. The paper proposes RVCE-FAL, a comprehensive acceleration solution for FALCON, leveraging RISC-V scalar-vector custom instructions for both signature generation and verification. By exploiting data-level parallelism (DLP) and designing efficient custom hardware, including the first hardware implementation of a FALCON Gaussian sampler, RVCE-FAL significantly improves computational efficiency. The solution achieves up to 18× and 6.9× speedup in signature generation and verification, respectively, when evaluated on a Xilinx UltraScale+ ZCU104 platform.

**Proposal Highlights:**

- RVCE-FAL: A comprehensive FALCON acceleration solution using RISC-V scalar-vector custom instructions for both signature generation and verification, addressing the need for optimized implementations in light of pending PQC standards.
- Data-Level Parallelism (DLP): Exploitation of DLP in polynomial operations over rings to improve computation speed, addressing the significant execution time these operations account for in signature verification.
- Custom Hardware Design: Introduction of custom hardware, including the first full hardware implementation of a FALCON Gaussian sampler, to break computational bottlenecks and enhance efficiency.
- HW/SW Co-Design: Utilization of hardware and software co-design to enable custom instructions for accelerating modular arithmetic, SHA-3 functions, and discrete Gaussian sampling.
- Performance Gains: The implementation of RVCE-FAL achieves significant speedups, with up to 18× for signature generation and 6.9× for signature verification, demonstrating its potential for efficient PQC applications in IoT and beyond.
- Simulation and Evaluation: Integration of RVCE-FAL into an RVV-enabled gem5-RTL simulation platform, with performance evaluations conducted on the Xilinx UltraScale+ ZCU104 platform, showcasing the solution's effectiveness and efficiency improvements over reference C implementations.


<br>

## Emerging Design Technologies For Future Computing

Date: Monday, 25 March 2024
Time: 16:30 CET - 18:00 CET

### SuperFlow: A Fully-Customized RTL-to-GDS Design Automation Flow for Adiabatic Quantum-Flux-Parametron Superconducting Circuits

**Summary**

This work emphasizes the superior energy efficiency of superconducting logic circuits, specifically Adiabatic Quantum-Flux-Parametron (AQFP) logic, over traditional Complementary Metal–Oxide–Semiconductor (CMOS) technology. AQFP, designed for significant reductions in power consumption through adiabatic switching, shows potential for substantial energy efficiency gains compared with CMOS. However, the distinct characteristics of AQFP, such as its components, logic gates, and power consumption patterns, render CMOS-focused design automation tools inapplicable. Previous efforts in design automation for AQFP have been fragmented, focusing either on the logic synthesis or placement stages without covering the complete design flow. Addressing this gap, the paper introduces a fully-customized RTL-to-GDS (Register-Transfer Level to Graphic Design System) design automation flow, SuperFlow, specifically developed for AQFP circuits. SuperFlow offers comprehensive design optimization capabilities, including mixed-cell-size constraints, a layer-wise routing strategy, and wirelength and timing optimization. The experimental results demonstrate that SuperFlow outperforms existing state-of-the-art placers for AQFP circuits in terms of wirelength and timing quality, marking a significant advancement in design automation for energy-efficient AQFP technology.

**Proposal Highlights:**

- SuperFlow: A fully-customized RTL-to-GDS design automation tool tailored for AQFP circuits, marking the first non-commercial effort in this area.
- Optimized Placement: Incorporates simultaneous optimization of wirelength and timing, adhering to clocking and mixed-cell-size constraints.
- Layer-wise Routing Strategy: Introduces a routing strategy that accommodates space expansion, effectively addressing potential routability issues.
- Improved Design Metrics: Demonstrates a 12.8% improvement in wirelength and a 12.1% enhancement in timing quality over previous solutions, highlighting SuperFlow's effectiveness in optimizing Power, Performance, and Area (PPA) for AQFP circuits.


### Para-ZNS: Improving Small-zone ZNS SSDs Parallelism through Dynamic Zone Mapping

**Summary**

The introduction discusses the NVMe Zoned Namespace (ZNS) interface as a solution to mitigate write amplification issues in flash-based SSDs by effectively placing data at the zone level. Despite its benefits in exploiting die-level parallelism for enhanced I/O performance, ZNS SSDs face challenges in efficiently mapping zones to die and plane-level parallelism, primarily due to the small size of zones and the host's limited ability to manage these mappings. To address these limitations, this paper introduces a novel dynamic zone mapping scheme for small-zone ZNS SSDs named Para-ZNS, aimed at improving both die-level and plane-level parallelism. Prior studies have attempted to enhance I/O parallelism in ZNS SSDs but often at the cost of substantial overhead or under restricted scenarios, failing to fully exploit plane-level parallelism. Para-ZNS dynamically maps blocks across multiple dies to open zones, optimizing the use of die and plane-level parallelism without the complexities and scalability issues of previous approaches. The paper's primary contributions include a parallel block grouping mechanism, a die-parallelism identification module to utilize idle dies based on zone states, and a dynamic zone mapping mechanism to maximize die utilization for open zones, significantly enhancing die-level parallelism in ZNS SSDs.

**Proposal Highlights:**

- Dynamic Zone Mapping: Introduces Para-ZNS, a novel design for small-zone ZNS SSDs, enhancing die-level and plane-level parallelism through dynamic zone-address mapping.
- Parallel Block Grouping: Proposes a mechanism to manage mapping units, ensuring access to these units exploits parallelism among multiple dies and at the plane level.
- Die-Parallelism Identification: Designs a device-side module to identify idle dies based on zone states, facilitating their further utilization.
- Maximizing Die Utilization: Implements a dynamic zone mapping mechanism to increase the number of dies utilized by open zones, fully leveraging the potential of die-level parallelism in ZNS SSDs.


### A3PIM: An Automated, Analytic and Accurate Processing-in-Memory Offloader

**Summary**

The introduction delineates the "memory wall" issue, where a significant discrepancy between CPU speeds and main memory system speeds leads to inefficiencies, particularly pronounced in data-intensive applications like graph processing and machine learning. Processing-in-Memory (PIM) technologies, enabled by advancements in 3D-stacked memory, offer a promising solution by integrating computational units within memory to minimize data movement. However, the challenge remains in effectively offloading workloads between CPU and PIM cores, considering factors such as compatibility of memory access patterns and the costs associated with inter-segment data movement and context-switching. This paper presents A3PIM, an automated, analytic, and accurate PIM offloader designed for CPU-PIM systems, which evaluates the intrinsic memory access pattern of each code segment. By considering the key factors influencing offloading decisions, A3PIM significantly improves workload execution speed, demonstrating substantial speedups over CPU-only and PIM-only executions across various benchmarks.

**Proposal Highlights:**

- Challenge of Effective Offloading: Emphasizes the need for an intelligent offloader to optimally distribute workloads between CPU and PIM cores, factoring in compatibility and data movement costs.
- A3PIM: Introduces an automated, analytic, and accurate offloader for CPU-PIM systems, which utilizes static code analysis to assess memory access patterns and decide on the best offloading strategy.
- Key Factors in Offloading: Identifies compatibility with PIM and CPU cores, inter-segment data movement overhead, and context-switching time as crucial considerations for offloading decisions.
- Performance Evaluation: Demonstrates significant speed improvements with A3PIM in CPU-PIM hybrid executions, achieving average speedups of 2.63x and 4.45x (and up to 7.14x and 10.64x) compared to CPU-only and PIM-only executions, respectively.


### BlockAMC: Scalable In-Memory Analog Matrix Computing for Solving Linear Systems

**Summary**

This work highlights the challenge of solving linear systems in the context of modern scientific computing and data-intensive tasks, especially given the resource-intensive nature of matrix computations on conventional digital computers. The advent of in-memory Analog Matrix Computation (AMC) with nonvolatile resistive memory, recognized for its high speed and low complexity, offers a promising solution by allowing basic matrix operations like matrix-vector multiplication (MVM) and matrix inversion (INV) to be performed efficiently in one step. Despite the theoretical potential for optimizing time complexity to approach O(1), practical limitations such as manufacturability, yield of resistive memory arrays, and non-ideal factors like cell variations and interconnect resistances pose significant challenges for implementing large-scale AMC circuits. To overcome these limitations, this work introduces BlockAMC, a method for solving large-scale linear systems using crosspoint resistive memory arrays by partitioning matrices into smaller blocks for sequential INV or MVM operations. The approach includes novel macro designs for circuit reconfiguration, operational amplification, sample-and-hold circuits, and analog-digital/digital-analog interfaces, leading to improved area and energy efficiencies as well as enhanced computing accuracy due to minimized error accumulation.

**Proposal Highlights:**

- BlockAMC Method: A novel approach for solving large-scale matrix inversion problems by partitioning the original matrix into smaller blocks for efficient processing.
- Compact Algorithm: A streamlined algorithm that performs INV or MVM operations on each block matrix sequentially, addressing the challenges of large-scale AMC implementations.
- Macro Design for AMC: Introduction of BlockAMC macro designs incorporating transmission gates for circuit reconfiguration, a clock controller, operational amplifiers for shared INV or MVM operations, sample-and-hold circuits, and ADC/DAC interfaces.
- One-stage and Two-stage BlockAMC: Implementation of both one-stage and two-stage BlockAMC configurations to optimize computational efficiency and accuracy.
- Improved Efficiencies: Demonstrated substantial improvements in area and energy efficiencies over traditional AMC circuits, alongside an analysis of the impact of intrinsic conductance variations and interconnect resistances.
- Enhanced Computing Accuracy: Achievement of improved computing accuracy through the use of smaller array sizes and strategies to reduce error accumulation.

<br>

## System Simulation, Validation And Verification

Date: Tuesday, 26 March 2024
Time: 08:30 CET - 10:00 CET

### EvilCS: An Evaluation of Information Leakage through Context Switching on Security Enclaves

**Summary**

This work articulates the critical challenge in System-on-Chip (SoC) design of balancing between area, power, performance, and security, especially given the vulnerability of modern processors to side-channel attacks due to performance enhancement features. Security enclaves or trusted execution environments (TEEs) have been developed to isolate sensitive applications from potential breaches, leveraging hardware functionalities. However, the process of context switching within these enclaves, particularly with RISC-V's implementation of Simultaneous Multithreading (SMT) and hardware threads (harts), introduces a new vulnerability termed "EvilCS". This vulnerability arises during the context switch process, where the power signature changes significantly, allowing an adversary with physical access to extract sensitive data from the trusted application within the security enclave. To counteract this, the paper proposes a statistics-based strategy for assessing the hardware and firmware of security enclave implementations, introducing techniques to maximize side-channel sensitivity, isolate power signatures related to context switching, and evaluate the correlation between register values and power consumption during this process. The evaluation of sixteen security enclave configurations demonstrates the existence of the EvilCS vulnerability and the effectiveness of the proposed countermeasures.

**Proposal Highlights:**

- EvilCS Vulnerability Identification: Identification of a new side-channel vulnerability, "EvilCS", which exploits the context switch process in security enclaves to recover sensitive data.
- Security Enclave Kernels and Hardware Threads: Discussion on the role of security enclave kernels like Keystone, Multizone, and OpenMZ in RISC-V architectures and the concept of hardware threads (harts) for enhancing processing throughput.
- Statistics-Based Assessment Strategy: Introduction of a comprehensive strategy for evaluating the security of enclave implementations against EvilCS through hardware and firmware analysis.
- Test Generation and Change Point Detection: Development of a test generation technique to enhance side-channel sensitivity during context switches and a change point detection method to isolate relevant power signatures.
- Power Analysis Technique: Implementation of a power analysis technique to establish a correlation between register values and power consumption during context switching.
- Evaluation of Security Enclave Configurations: Assessment of sixteen security enclave configurations to determine the presence and exploitability of the EvilCS vulnerability, demonstrating the approach's effectiveness.


### Selfie5: An autonomous, self-contained verification approach for high-throughput random testing of programmable processors

**Summary**

This work discusses the critical role of functional verification in the processor design cycle, highlighting the limitations of directed and random testing strategies due to the immense complexity of processor designs and the simulation runtime constraints. Traditional verification approaches, while necessary, struggle to cover the vast space of potential design bugs, especially those involving internal control mechanisms and dynamic system interactions. To address these challenges, the paper introduces Selfie5, a novel verification method that leverages the device under verification (DUV) itself to generate, execute, and verify random control sequences autonomously, eliminating the need for external test generation and checkers and thereby bypassing communication overheads. This self-contained approach enables verification across all stages, from simulation and FPGA-prototyping to hardware emulation and post-silicon, with the capability to run at the target speed of the DUV. Implemented on a RISC-V platform and tested on a 16 nm SoC running at 1 GHz, Selfie5 achieved a testing throughput of 13.8 billion instructions per hour—significantly higher than existing methods—successfully identifying real design bugs that other testing strategies had missed.

**Proposal Highlights:**

- Selfie5 Verification Approach: Introduces a self-contained, autonomous verification strategy that enables the device under verification to generate, execute, and check random control sequences internally.
- Elimination of Communication Overheads: By running the entire test process on the DUV, Selfie5 removes the need for external test components, avoiding the bottlenecks associated with communication channels.
- Versatile Application Across Verification Stages: The approach is applicable across various verification levels, including simulation, FPGA-prototyping, hardware emulation, and post-silicon testing.
- High Testing Throughput: Demonstrated on a 16 nm RISC-V SoC running at 1 GHz, Selfie5 achieved a throughput of 13.8 billion instructions per hour, substantially outperforming traditional methods.
- Effective Bug Detection: The massive testing capability of Selfie5 exposed real design bugs that had been missed by compliance testing and other common verification strategies.


### Heterogeneous Static Timing Analysis with Advanced Delay Calculator

**Summary**

This work discusses the limitations of traditional delay models like Elmore and NLDM in accurately modeling delays in advanced node designs, particularly beyond 45nm, due to their oversimplified assumptions which fail to account for complex RC networks and the resistive shielding effect. To enhance accuracy, standard timers like PrimeTime and OpenSTA have adopted model-order-reduction (MOR) techniques and effective capacitance formulas, notably incorporating the Arnoldi algorithm for more precise interconnect delay calculations. However, the improved accuracy comes at the cost of significantly increased runtime, limiting the use of Arnoldi algorithm in timing-driven optimization workflows. Addressing the need for better runtime efficiency without sacrificing accuracy, the paper proposes a GPU-accelerated delay calculator that leverages the parallelism of GPU computing to speed up the Arnoldi algorithm and effective capacitance computation for advanced interconnect modeling. This novel approach results in substantial speed improvements over traditional methods like PrimeTime and OpenSTA, achieving up to 7.27× and 14.03× speed-up, respectively, while maintaining strong correlation in accuracy with PrimeTime.

**Proposal Highlights:**

- GPU-Accelerated Delay Calculator: Introduction of a new delay calculator that utilizes GPU acceleration for advanced interconnect modeling, significantly improving runtime efficiency.
- Arnoldi Model Order Reduction: Incorporation of Arnoldi model order reduction and effective capacitance computation to enhance the accuracy of delay calculations.
- Design of Efficient GPU Kernels: Creation of optimized GPU kernels to accelerate various numerical tasks in interconnect modeling, including batched nodal analysis construction, LU factorization, Krylov subspace calculation, eigenvalue decomposition, and Newton-Raphson iterations.
- Integration into a Fully GPU-Accelerated STA Engine: The novel delay calculator is integrated into a comprehensive STA engine that is fully accelerated by GPU, offering substantial speed-ups in processing large circuit designs.
- Significant Speed Improvements: Demonstrated speed improvements of up to 7.27× over PrimeTime and 14.03× over OpenSTA, without compromising the accuracy of timing analysis.

### A RISC-V "V" VP: Unlocking Vector Processing for Evaluation at the System Level

**Summary**

This work emphasizes the importance of RISC-V's modularity and the significance of Single Instruction, Multiple Data (SIMD) instruction set extensions in enhancing performance for data-level parallelism tasks. The limitations of classical SIMD, such as fixed vector lengths and the challenge of adapting vector lengths to varying workloads, are discussed. The paper introduces a novel contribution to this area by extending an open-source SystemC Transaction Level Modeling (TLM) based RISC-V Virtual Prototype (VP) with the RISC-V "V" Vector Extension (RVV), enabling variable-length vectors and thereby overcoming the limitations of classical SIMD. This extension, termed RISC-V VP++, allows for the full exploitation of hardware capabilities by software without recompilation for different hardware configurations. By integrating RVV into the VP and employing code generation for the implementation, the paper achieves efficient support for RVV across RV32 and RV64 instruction set simulators, significantly reducing potential implementation errors and maintenance efforts. The utility of the extended VP is demonstrated through case studies, showing valuable insights for the design of RVV microarchitectures. This work stands out as the only open-source, SystemC-based VP supporting RVV and an instruction-accurate execution cycle model, providing a significant tool for system-level evaluation and architectural exploration.

**Proposal Highlights:**

- RISC-V's Modularity and SIMD Extensions: Highlights RISC-V's potential for customization and specialization, particularly for tasks requiring data-level parallelism through SIMD extensions.
- Limitations of Classical SIMD: Discusses the challenges associated with fixed vector lengths in classical SIMD architectures and the difficulty of adapting to specific workloads.
- RISC-V VP++ Extension: Introduces an extension to an open-source RISC-V Virtual Prototype to support the RVV extension, enabling variable-length vectors and addressing the limitations of classical SIMD.
- Efficient RVV Support in Virtual Prototypes: Details the integration of RVV into the Virtual Prototype, facilitating efficient support for RV32 and RV64 instruction set simulators and robust functional verification of RVV.
- Code Generation for Implementation: Utilizes code generation to implement over 600+ RVV instructions, significantly reducing the risk of errors and maintenance requirements.
- System-Level Evaluation and Architectural Insights: Demonstrates the utility of the RVV-enhanced Virtual Prototype in providing valuable insights for the design of RVV microarchitectures through case studies.
- Unique Contribution: Positions the RISC-V VP++ as the only open-source, SystemC-based Virtual Prototype supporting RVV and an instruction-accurate execution cycle model, filling a gap in system-level evaluation and architectural exploration tools.


<br>

## Microarchitectural And Side-Channel-Based Attacks And Countermeasures

Date: Monday, 25 March 2024
Time: 11:00 CET - 12:30 CET

### Three Sidekicks to Support Spectre Countermeasures

**Notes**
- Spectre exploits speculative execution on CPUs
  - Conditional branch prediction (PHT) is hard and expensive to mitigate
- Attacker on the same machine can mistrain the predictor with a malicious program to make the user access secrets in the uArch state
- Many countermeasures - stop speculative execution
- +50 countermeasures presented
- Performance penalty is high for many countermeasures
- Contributions:
  - Examine code transformations to reduce conditional branch predictions and limit the number of available gadgets
  - Control-flow linearization
  - zero overhead loops
  - instruction rescheduling
  - Analysis of the impact with gem5

### Detecting Backdoor Attacks in Black-Box Neural Networks through Hardware Performance Counters

**Notes**

- Black box access to Amazon web services, google cloud
- Detect backdoor attacks in black box neural networks
- Backdoor can be enabled by a single pixel fix
- HW performance counters - different distributions of inputs
  - Monitor caches

### Can Machine Learn Pipeline Leakage?

**Notes**

- Automated RNN framework simulates side-channel attacks in embedded devices.
- Reduces feature dimensionality by one-third for three-stage pipelines.
- Scalable model efficient for longer pipeline microprocessors.
- Matches MLP performance, surpassing on reduced datasets.
- Utilizes ABBY-CM0 dataset, compares CNN, LSTM, GRU architectures.

### A Deep-Learning Technique to Locate Cryptographic Operations in Side-Channel Traces

**Notes**

- Side-Channel power traces
- Side-channel attacks allow extracting secret information from the execution of cryptographic primitives by correlating the partially known computed data and the measured side-channel signal. However, to set up a successful side-channel attack, the attacker has to perform i) the challenging task of locating the time instant in which the target cryptographic primitive is executed inside a side-channel trace and then ii) the time-alignment of the measured data on that time instant. This paper presents a novel deep-learning technique to locate the time instant in which the target computed cryptographic operations are executed in the side-channel trace. In contrast to state-of-the-art solutions, the proposed methodology works even in the presence of trace deformations obtained through random delay insertion techniques. We validated our proposal through a successful attack against a variety of unprotected and protected cryptographic primitives that have been executed on an FPGA-implemented system-on-chip featuring a RISC-V CPU.
- Define any application in a trace


<br>

## Low-Power And Energy-Efficient Design

Date: Wednesday, 27 March 2024
Time: 08:30 CET - 10:00 CET


### Algorithm-hardware co-design for Energy-Efficient A/D conversion in ReRAM-based accelerators

**Notes**

- ReRAM-based accelerators improve DNN performance but suffer from high ADC power consumption.
- Over 60% of power is consumed by ADC, affecting efficiency.
- Proposed solution eliminates redundant ADC operations to maintain accuracy.
- Introduces algorithm-hardware co-design for efficient quantization and coding.
- Achieves 1.6 to 2.3 times ADC power reduction while retaining algorithm flexibility.

### An Efficient Asynchronous Circuits Design Flow with Backward Delay Propagation Constraint

**Notes**

- Asynchronous circuits gain popularity in IoT and neural networks for low power use.
- Design efficiency hampered by a lack of specialized EDA tools.
- Proposes a design flow with traditional EDA tools and a novel BDPC method.
- New flow improves timing analysis accuracy and efficiency significantly.
- Asynchronous RISC-V processor implementation shows 17.4% power optimization.

### Attention-Based EDA Tool Parameter Explorer: From Hybrid Parameters to Multi-QoR metrics

**Notes**

- Enhancing VLSI design QoR without changing design enablement is vital for IC designers.
- Parameter tuning for EDA tools is emerging to improve design outcomes.
- Proposed attention-based explorer uses self-attention for parameter importance.
- Optimizes continuous and discrete parameters via a hybrid Gaussian process model.
- Custom EHVI acquisition function enables multi-objective optimization and parallel evaluation.

### Parallel Multi-objective Bayesian Optimization Framework for CGRA Microarchitecture

**Notes**

- CGRA microarchitecture gains popularity in DNN acceleration due to its flexibility.
- Designing optimal CGRA microarchitecture is challenging due to vast design space.
- Introduces PAMBOF, a parallel multi-objective Bayesian optimization framework for design exploration.
- Uses high-precision models and DRKL-GP for fast, accurate design space exploration.
- PAMBOF outperforms prior methods in finding better area-performance CGRA designs quickly.

### Efficient Spectral-Aware Power Supply Noisy Analysis for Low-Power Design Verification

**Notes**

- Advanced methods needed for low-power design verification to mitigate power supply noise.
- Spectral methods reduce iterations in supply noise verification but face computational challenges.
- Proposes a two-stage spectral-aware algorithm for efficient preconditioner generation.
- Algorithm uses spectral-aware weights and eigenvalue transformation for quick, accurate solutions.
- Outperforms GRASS and feGRASS in preconditioner generation and solver acceleration.


### Compact Powers-of-Two: An Efficient Non-Uniform Quantization for Deep Neural Networks

**Notes**

- Proposes CPoT, an efficient non-uniform quantization scheme for DNNs to reduce computation and memory demands.
- CPoT adds a fractional part and biasing operation to improve quantization resolution near 0 and at edge regions.
- Optimizes dot product calculation for CPoT quantized DNNs by incorporating precomputable terms into bias.
- Designs a MAC unit for CPoT using shifters and LUTs, facilitating hardware implementation.
CPoT outperforms state-of-the-art methods in both data-free quantization and hardware efficiency.


<br>

## Adaptive And Sensing Systems

Date: Monday, 25 March 2024
Time: 11:00 CET - 12:30 CET

### Multi-Agent Reinforcement Learning for Thermally-Restricted Performance Optimization on Manycores

**Notes**

- Thermal restriction is one of the main design constriants
- SoP: thermal control circuitry (TCC), throttling f/V levels of all cores
- SoA: selective, per-core f/V -> this work
- TL;DR: use ML to learn complex relationships on complex SoCs
- Two ML ways:
  - Supervised Learning (SL) -> limited by training data, limited adaptability
  - Reinforcement learning (RL): good adaptability to runtime changes, but high overhead for large action space (e.g., many cores with several f/V levels -> exponential growth)
- This work uses RL and tackles the problem of the overhead
  - Multi-agent RL (kind of a distributed approach)
  - Action: per-core f/V
  - Reward: scalar function to represent the targeted object
  - Almost 35% performance gain


### Trace-enabled timing model synthesis for ROS2-based autonomous applications

**Notes**

- ROS2: middleware for developing autonomous applications
- Problem of having timing analysis for ROS2 systems, due to real-time requirements of applications like automotive
- Based of BPF
- Traces are collected and processed by a tool, part of the framework, which can measure, e.g., response time of callbacks
- Case study: benchmarks on autoware


### sLET for distributed aerospace landing system

**Notes**

- Aerospace: federated->IMAC->distributed architectures
- Need for new operational efficiency in the development of critical real-time embedded systems
- A time-aware computational model acts between temporal synchronization points (windows)
- Predictability, testability, and ultimately strong determinism are crucial high-level properties needed not only at equipment level but at the whole system scope
- This paper deals with an innovative solution, based on the sLET (synchronous logical execution time) paradigm, to bring drastic integration time reduction whatever the underlying architecture





<br>

## Technical Paper Session: Towards Assuring Safe Autonomous Driving

Date: Monday, 25 March 2024
Time: 14:00 CET - 15:30 CET

### Back To The Future: Reversible Runtime Neural Network Pruning For Safe Autonomous Systems

**Notes**

- Drop in accuracy due to pruning can compromise safety; sometimes fall-back and rapid recovery are not possible
- Pruned model is run during safe operation, full-accuracy model is run when an anomaly is detected
- Offline step:
  - Pruning happens offline in 2 steps: traditional pruning and fine-tuning down, then survived weights are regrowth and further fine-tuned up
  - Only layers affected by the pruning process are encoded
- Runtime step:
  - reverse() is used to swap between the two archi models, which are stored in memory at all times
  - Note: fine tuning on the sparse NN is needed otherwise accuracy drop would be too poor


### Adassure: Debugging Methodology For Autonomous Driving Control Algorithms

**Notes**

- How to improve the robustness of autonomous driving algos
- Development of a framework (ADAssure) for diagnosis of cyber-attacks and quick detection of weak-spots
- The framework is assertion-based



<br>

## Novel Architecture Solutions

Date: Monday, 25 March 2024
Time: 16:30 CET - 18:00 CET

### FusionArch: A Fusion-Based Accelerator for Point-Based Point Cloud Neural Networks

**Notes**

- Proposes 3 orthogonal algorithms (Fusion-FPS, Fusion-Computation, Fusion-Aggregation) for efficient PCNNs processing.
- Fusion-FPS reduces Farthest Point Sampling frequency and organizes neighbor search stages in parallel.
- Fusion-Computation eliminates redundant feature computations for "Filling Points" by borrowing nearest neighbor features.
- Fusion-Aggregation reduces memory access by clustering centroids with shared neighbors.
- FusionArch architecture significantly boosts PCNN performance and energy efficiency on various platforms, outperforming existing solutions.


### Efficient Exploration of Cyber-Physical System Architectures Using Contracts and Subgraph Isomorphism

**Notes**

- Propose ContrArc: explores cyber-physical system architectures to minimize costs and meet diverse constraints.
- Utilizes assume-guarantee contracts for system requirements and component interfaces.
- Translates exploration into a mixed integer linear programming problem for efficient solution searching.
- Employs contract decompositions and subgraph isomorphism for pruning infeasible architectures.
- Demonstrates significant acceleration in architectural exploration compared to other methods.


### Synthesizing Hardware-Software Leakage Contracts for RISC-V Open-Source Processors

**Notes**

- ISA do not provide any guarantees on side-channels
- Timing side channels: leakage of data via execution time
  - Caches
  - Branch predictors
  - Operand-dependent execution time
- Leakage contracts: instruction-level leakage specification -> no leakage as long as you respect the contract; contract stipulated after contract observations
- Contract = combination of multiple contract atoms


<br>

## ASD Technical Paper Session: Real-Time Aware Communication Systems for Autonomy

Date: Tuesday, 26 March 2024
Time: 08:30 CET - 10:00 CET

### End-to-End Latency Optimization of Thread Chains Under the DDS Publish/Subscribe Middleware

**Notes**

- Complex distributed applications based on publish-subscribe patterns implemented as chains of computations
- E.g. automotive: hundreds of interacting ECUs with SoA frameworks like Autosar and ROS2
- ROS2 middleware uses DDS for communication -> often neglected from optimization, while real-time analysis has been carried out
- The work focuses on FastDDS:
  - Flow controller threads -> middl.
  - Listener threads -> middl.
  - Publisher-subscriber threads -> appl. level
- Chains make things more complicated: chain of threads with data dependencies
- Holistic analysis for thread chains based on the Compositional Performance Analysis (CPA) approach
- How to optimize the DDS parameters to meet timing requirements at best -> minimization of the worst case thread latency
- Approach: systematic analysis-driven, non-convex optimization problem
- Threads with unique priority, threads scheduled by fixed priority partitioned scheduler
- The crafter optimizer is compared with a simulated annealing as baseline
- Evaluation on Autoware with a LIDAR chain

### Orchestration-aware optimization of ROS2 communication protocols

**Notes**

- Proposed ROS-SHC -> new protocol on shared memory based on a message queue
- Multi-subscriber allowed thanks to local buffers without additional copies


<br>

## Efficient And Secure Systems-On-Chip For The New Hyperconnected Environments

Date: Tuesday, 26 March 2024
Time: 11:00 CET - 12:30 CET

### Efficient Fast Additive Homomorphic Encryption Cryptoprocessor for Privacy-preserving Federated Learning Aggregation

**Notes**

- Addresses privacy leakage in collaborative deep learning model training with homomorphic encryption.
- Overcomes computing power and throughput limitations of the Paillier scheme.
- Introduces a high-throughput cryptoprocessor using the Fast Additive Homomorphic Encryption (FAHE) algorithm.
- Implements efficient FPGA mapping for encryption and rapid modular reduction for decryption.
- Achieves significantly higher throughput and lower latency than existing Paillier accelerators and FAHE software implementations.

### Cache Bandwidth Contention Leaks Secrets

**Notes**

- Deliberate blockages in CPU cache communication paths were created to form high-speed covert channels.
- Introduced three specific covert channels (L2CC, L3CC, LiCC) with transmission speeds up to 10.37 Mbps.
- The speeds of L2CC and L3CC outperform many existing covert channels, both memory and non-memory based.
- These channels can extract sensitive information from cryptographic algorithms like RSA and EdDSA.
- The methods used can circumvent common defenses against Spectre attacks, compromising security further.

### IOMMU Deferred Invalidation Vulnerability: Exploit and Defense

**Notes**

- A malicious DMA IO device can compromise the system
- Performance optimization: deferred invalidationch


<br>

## Focus Session: Smoothing Disruption Across The Stack: Tales Of Memory, Heterogeneity, And Compilers

Date: Wednesday, 27 March 2024
Time: 11:00 CET - 12:30 CET

### HETEROGENEOUS SCALING DRIVEN BY SYSTEM-TECHNOLOGY CO-OPTIMIZATION

**Notes**

- Use advanced 3D technologies for connectivity performance
- CMOS cost is exploding, PPA limits
- How can we provide versatile technology platforms?
- Device level: maximize efficiency by playing with device geometry, which is already stressed
- But at some point, the geometry solutions will be gone, because the device is still the same
- Litho processes are scaling (roadmap). Metal-based instead of damascene guarantees high aspect ratios
- Dream of 3D: stacking logic over logic; now technology is ready to welcome that
- Memory is moving to 3D as well: SRAM, DRAM and NAND
- CMOS2.0 -> new ow cap devices, extreme patterming, beyond Si devices…
- Technology for the future: IP folding
- Dense logic will drive scaling


<br>

## Emerging Machine Learning Techniques

Date: Monday, 25 March 2024
Time: 14:00 CET - 15:30 CET

### A Computationally Efficient Neural Video Compression Accelerator Based on a Sparse CNN-Transformer Hybrid Network

**Notes**

- neural video compression
- goal: real-time decoding
- The challenge is the enc-dec asymmetry
- They use Swin topology, compressing on motion, then on resolution

### MultimodalHD: Federated Learning Over Heterogeneous Sensor Modalities using Hyperdimensional Computing

**Notes**

- Multimodal FL
  - you don't have to assume all clients have all modalities
  - attention for multimodal fusion
- Fed: they aggregate based on the proximity of the model weights
  - they share model weights
  - quicker convergence that other (AvgFed-like works), but he still believes it could go faster

![](./figures/Emerging%20Machine%20Learning%20Techniques/2a.JPEG)

### DyPIM: Dynamic-inference-enabled Processing-In-Memory Accelerator

**Notes**

- structured pruning (n.b. pruning - one of the main topics present in DATE)
- generated based on the input

### MicroNAS: Zero-Shot Neural Architecture Search for MCUs

**Notes**

- ***Worth checking out, as they target zero-shot NAS.***
- They jointly consider performance indications + hardware metrics
  - Differentiable?
- Note also the work of Abas Rahimi, on zero-shot NAS
  - similarity kernel
  - using generative model - attributed dictionary and class attributes


### Accelerating DNNs using Weight Clustering on RISC-V Custom Functional Units

**Notes**

- DNN acceleration
- Compared to quantization, they additionally have hardware datapaths to process each cluster
- There is some sweet spot for #clusters vs #bits for precision, but they didn't analyse that