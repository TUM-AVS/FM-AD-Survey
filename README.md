# Foundation Models in Autonomous Driving: A Dual Survey on Scenario Generation and Scenario Analysis :car:
<div align="center">
<a href="https://example.com/paper-to-be-published"><img src="https://img.shields.io/badge/Paper-PDF-red.svg" alt="Paper Badge"/></a>
<a href="https://github.com/TUM-AVS/FM-AD-Survey/stargazers"><img src="https://img.shields.io/github/stars/TUM-AVS/FM-AD-Survey" alt="Stars Badge"/></a>
<a href="https://github.com/TUM-AVS/FM-AD-Survey/network/members"><img src="https://img.shields.io/github/forks/TUM-AVS/FM-AD-Survey" alt="Forks Badge"/></a>
<a href="https://github.com/TUM-AVS/FM-AD-Survey/pulls"><img src="https://img.shields.io/github/issues-pr/TUM-AVS/FM-AD-Survey" alt="Pull Requests Badge"/></a>
<a href="https://github.com/TUM-AVS/FM-AD-Survey/issues"><img src="https://img.shields.io/github/issues/TUM-AVS/FM-AD-Survey" alt="Issues Badge"/></a>
<a href="https://github.com/TUM-AVS/FM-AD-Survey/blob/main/LICENSE"><img src="https://img.shields.io/github/license/TUM-AVS/FM-AD-Survey" alt="License Badge"/></a>
</div>

This repository will collect research, implementations, and resources related to **Foundation Models for Scenario Generation and Analysis** in autonomous driving. The repository will be maintained by [TUM-AVS](https://www.mos.ed.tum.de/avs/startseite/) (Professorship of Autonomous Vehicle Systems at Technical University of Munich) and will be continuously updated to track the latest work in the community.

<p align="center">
<img src="Assets/00_concept_c.png" width="100%" height="auto"/>
</p>

## :fire: Updates
- [May.2024] Repository initialized

## 🤝 &nbsp; Citation
Please visit [Foundation Models in Autonomous Driving: A Dual Survey on Scenario Generation and Scenario Analysis](https://example.com/paper-to-be-published) for more details and comprehensive information (coming soon). If you find our paper and repo helpful, please consider citing it as follows:

<!-- Complete author list for when ready to publish:
Yuan Gao, Mattia Piccinini, Yuchen Zhang, Dingrui Wang, Korbinian Moller, Roberto Brusnicki, Baha Zarrouki, Alessio Gambi, Jan Frederik Totz, Kai Storms, Steven Peters, Andrea Stocco, Bassam Alrifaee, Marco Pavone and Johannes Betz
-->

```BibTeX
@article{Foundation-Models-AD-Dual-Survey,
  author={TBD},
  title={Foundation Models in Autonomous Driving: A Dual Survey on Scenario Generation and Scenario Analysis},
  journal={TBD},
  year={2024},
  pages={TBD},
  doi={TBD}
}
```

## :page_with_curl: Introduction
Foundation models are large-scale, pre-trained models that can be adapted to a wide range of downstream tasks. In the context of autonomous driving, foundation models offer a powerful approach to scenario generation and analysis, enabling more comprehensive and realistic testing, validation, and verification of autonomous driving systems. This repository aims to collect and organize research, tools, and resources in this important field.

<p align="center">
<img src="Assets/Sec2_FMs_page-0002.png" width="100%" height="auto"/>
</p>

## :chart_with_upwards_trend: Publication Timeline
The following figure shows the evolution of foundation model research in autonomous driving scenario generation and analysis over time:

<p align="center">
<img src="Assets/timeline.png" width="100%" height="auto"/>
</p>

## :mag: Search Methodology
The following list of keywords was used to search this survey's papers in the Google Scholar database. The keywords were entered either individually or in combination with other keywords in the list. The search was conducted until May 2025.

**Keywords:**
- **Foundation Model Types:** Foundation Models, Large Language Models (LLMs), Vision-Language Models (VLMs), Multimodal Large Language Models (MLLMs), Diffusion Models (DMs), World Models (WMs), Generative Models (GMs)
- **Scenario Generation & Analysis:** Scenario Generation, Scenario Simulation, Traffic Simulation, Scenario Testing, Scenario Understanding, Driving Scene Generation, Scene Reasoning, Risk Assessment, Safety-Critical Scenarios, Accident Prediction
- **Application Context:** Autonomous Driving, Self-Driving Vehicles, AV Simulation, Driving Video Generation, Traffic Datasets, Closed-Loop Simulation, Safety Assurance
## 🌟 Large Language Models for Autonomous Driving

<details>
<summary><strong>Scenario Generation (LLM)</strong></summary>

| Paper | Date | Venue | Code |
|:------|:-----|:------|:-----|
| [TARGET: Automated Scenario Generation from Traffic Rules for Testing Autonomous Vehicles](https://arxiv.org/abs/2305.06018) | 2023-05 | arXiv | - |
| [Language Conditioned Traffic Generation](https://arxiv.org/abs/2307.07947) | 2023-07 | CoRL 2023 | [GitHub](https://github.com/Ariostgx/lctgen/) |
| [A Generative AI-driven Application: Use of Large Language Models for Traffic Scenario Generation](https://ieeexplore.ieee.org/document/10415934) | 2023-11 | ELECO 2023 | - |
| [ChatGPT-Based Scenario Engineer: A New Framework on Scenario Generation for Trajectory Prediction](https://ieeexplore.ieee.org/document/10423819) | 2024-02 | IEEE Transactions on Intelligent Vehicles | - |
| [Enhancing Autonomous Vehicle Training with Language Model Integration and Critical Scenario Generation](https://arxiv.org/abs/2404.08570) | 2024-04 | arXiv | [GitHub](https://github.com/zachtian/CRITICAL) |
| [LLMScenario: Large Language Model Driven Scenario Generation](https://ieeexplore.ieee.org/document/10529537) | 2024-05 | IEEE Transactions on Systems, Man, and Cybernetics: Systems | - |
| [Automatic Generation Method for Autonomous Driving Simulation Scenarios Based on Large Language Model](https://link.springer.com/chapter/10.1007/978-981-96-3977-9_10) | 2024-05 | AIAT 2024 | - |
| [ChatScene: Knowledge-Enabled Safety-Critical Scenario Generation for Autonomous Vehicles](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10655362) | 2024-05 | CVPR 2024 | [GitHub](https://github.com/javyduck/ChatScene) |
| [Editable scene simulation for autonomous driving via collaborative llm-agents](https://ieeexplore.ieee.org/document/10656629) | 2024-06 | CVPR 2024 | [GitHub](https://github.com/yifanlu0227/ChatSim) |
| [Chat2Scenario: Scenario Extraction From Dataset Through Utilization of Large Language Model](https://ieeexplore.ieee.org/document/10588843) | 2024-06 | IV 2024 | [GitHub](https://github.com/ftgTUGraz/Chat2Scenario) |
| [SoVAR: Building Generalizable Scenarios from Accident Reports for Autonomous Driving Testing](https://arxiv.org/abs/2409.08081) | 2024-09 | ASE 2024 | - |
| [LeGEND: A Top-Down Approach to Scenario Generation of Autonomous Driving Systems Assisted by Large Language Models](https://arxiv.org/abs/2409.10066) | 2024-09 | ASE 2024 | [GitHub](https://github.com/MayDGT/LeGEND) |
| [Traffic Scene Generation from Natural Language Description for Autonomous Vehicles with Large Language Model](https://arxiv.org/abs/2409.09575) | 2024-09 | arXiv | [GitHub](https://github.com/basiclab/TTSG) |
| [Promptable Closed-loop Traffic Simulation](https://arxiv.org/abs/2409.05863) | 2024-09 | CoRL 2024 | [GitHub](https://ariostgx.github.io/ProSim/) |
| [Multimodal Large Language Model Driven Scenario Testing for Autonomous Vehicles](https://arxiv.org/abs/2409.06450) | 2024-09 | arXiv | - |
| [LLM-Driven Testing for Autonomous Driving Scenarios](https://ieeexplore.ieee.org/document/10852505) | 2024-11 | FLLM 2024 | - |
| [ChatSUMO: Large Language Model for Automating Traffic Scenario Generation in Simulation of Urban MObility](https://ieeexplore.ieee.org/document/10770822) | 2024-11 | IEEE Transactions on Intelligent Vehicles | - |
| [Generating Out-Of-Distribution Scenarios Using Language Models](https://arxiv.org/abs/2411.16554) | 2024-11 | arXiv | - |
| [Generating Traffic Scenarios via In-Context Learning to Learn Better Motion Planner](https://arxiv.org/abs/2412.18086) | 2024-12 | AAAI 2025 Oral | [GitHub](https://ezharjan.github.io/AutoSceneGen/) |
| [LLM-attacker: Enhancing Closed-loop Adversarial Scenario Generation for Autonomous Driving with Large Language Models](https://arxiv.org/abs/2501.15850) | 2025-01 | arXiv | - |
| [Risk-Aware Driving Scenario Analysis with Large Language Models](https://arxiv.org/abs/2502.02145) | 2025-02 | arXiv | [GitHub](https://github.com/TUM-AVS/From-Words-to-Collisions) |
| [CurricuVLM: Towards Safe Autonomous Driving via Personalized Safety-Critical Curriculum Learning with Vision-Language Models](https://arxiv.org/abs/2502.15119) | 2025-02 | arXiv | [GitHub](https://zihaosheng.github.io/CurricuVLM/) |
| [Text2Scenario: Text-Driven Scenario Generation for Autonomous Driving Test](https://arxiv.org/abs/2503.02911) | 2025-03 | arXiv | [GitHub](https://caixxuan.github.io/Text2Scenario.GitHub.io/) |
| [Seeking to Collide: Online Safety-Critical Scenario Generation for Autonomous Driving with Retrieval Augmented Large Language Models](https://arxiv.org/abs/2505.00972) | 2025-05 | arXiv | - |

</details>

<details>
<summary><strong>Scenario Analysis (LLM)</strong></summary>

| Paper | Date | Venue | Code |
|:------|:-----|:------|:-----|
| [Semantic Anomaly Detection with Large Language Models](https://arxiv.org/abs/2305.11307) | 2023-09 | Autonomous Robots | - |
| [LLM Multimodal Traffic Accident Forecasting](https://www.mdpi.com/1424-8220/23/22/9225) | 2023-11 | Sensors 2023 MDPI | - |
| [Reality Bites: Assessing the Realism of Driving Scenarios with Large Language Models](https://arxiv.org/abs/2403.09906) | 2024-03 | IEEE/ACM First International Conference on AI Foundation Models and Software Engineering (Forge) | [GitHub](https://github.com/Simula-COMPLEX/RealityBites) |
| [Driving with LLMs: Fusing Object-Level Vector Modality for Explainable Autonomous Driving](https://ieeexplore.ieee.org/abstract/document/10611018) | 2024-05 | ICRA 2024 | [GitHub](https://github.com/wayveai/Driving-with-LLMs) |
| [Generating Out-Of-Distribution Scenarios Using Language Models](https://arxiv.org/abs/2411.16554) | 2024-11 | arXiv | - |
| [SenseRAG: Constructing Environmental Knowledge Bases with Proactive Querying for LLM-Based Autonomous Driving](https://arxiv.org/abs/2501.03535) | 2025-01 | arXiv | - |
| [Risk-Aware Driving Scenario Analysis with Large Language Models](https://arxiv.org/abs/2502.02145) | 2025-02 | arXiv | [GitHub](https://github.com/TUM-AVS/From-Words-to-Collisions) |
| [CurricuVLM: Towards Safe Autonomous Driving via Personalized Safety-Critical Curriculum Learning with Vision-Language Models](https://arxiv.org/abs/2502.15119) | 2025-02 | arXiv | [GitHub](https://zihaosheng.github.io/CurricuVLM/) |
| [A Comprehensive LLM-powered Framework for Driving Intelligence Evaluation](https://arxiv.org/abs/2503.05164) | 2025-03 | arXiv | - |

</details>

## 🌟 Vision-Language Models for Autonomous Driving

<details>
<summary><strong>Scenario Generation (VLM)</strong></summary>

| Paper | Date | Venue | Code |
|:------|:-----|:------|:-----|
| [WEDGE: A multi-weather autonomous driving dataset built from generative vision-language models](https://arxiv.org/pdf/2305.07528) | 2023-05 | CVPR workshop 2023 | - |
| [DriveGenVLM: Real-world Video Generation for Vision Language Model based Autonomous Driving](https://ieeexplore.ieee.org/abstract/document/10786438) | 2024-08 | IAVVC 2024 | - |
| [Multimodal Large Language Model Driven Scenario Testing for Autonomous Vehicles](https://arxiv.org/abs/2409.06450) | 2024-09 | arXiv | - |
| [From Dashcam Videos to Driving Simulations: Stress Testing Automated Vehicles against Rare Events](https://arxiv.org/abs/2411.16027) | 2024-11 | arXiv | - |
| [Generating Out-Of-Distribution Scenarios Using Language Models](https://arxiv.org/abs/2411.16554) | 2024-11 | arXiv | - |
| [From Accidents to Insights: Leveraging Multimodal Data for Scenario-Driven ADS Testing](https://arxiv.org/abs/2502.02025) | 2025-02 | arXiv | - |
| [CurricuVLM: Towards Safe Autonomous Driving via Personalized Safety-Critical Curriculum Learning with Vision-Language Models](https://arxiv.org/abs/2502.15119) | 2025-02 | arXiv | [GitHub](https://zihaosheng.github.io/CurricuVLM/) |

</details>

<details>
<summary><strong>Scenario Analysis (VLM)</strong></summary>

| Paper | Date | Venue | Code |
|:------|:-----|:------|:-----|
| [Unsupervised 3D Perception with 2D Vision-Language Distillation for Autonomous Driving](https://arxiv.org/abs/2309.14491) | 2023-09 | ICCV 2023 | - |
| [OpenAnnotate3D: Open-Vocabulary Auto-Labeling System for Multi-modal 3D Data](https://arxiv.org/abs/2310.13398) | 2023-10 | ICRA 2024 | - |
| [On the Road with GPT-4V(ision): Early Explorations of Visual-Language Model on Autonomous Driving](https://arxiv.org/abs/2311.05332) | 2023-11 | ICIL 2024 Workshop on Large Language Models for Agents | [GitHub](https://github.com/PJLab-ADG/GPT4V-AD-Exploration) |
| [Talk2BEV: Language-enhanced Bird's-eye View Maps for Autonomous Driving](https://arxiv.org/abs/2310.02251) | 2023-11 | ICRA 2024 | [GitHub](https://github.com/llmbev/talk2bev) |
| [LLM Multimodal Traffic Accident Forecasting](https://www.mdpi.com/1424-8220/23/22/9225) | 2023-11 | Sensors 2023 MDPI | - |
| [NuScenes-MQA: Integrated Evaluation of Captions and QA for Autonomous Driving Datasets using Markup Annotations](https://ieeexplore.ieee.org/abstract/document/10495633) | 2024-01 | WACVW LLVM-AD 2024 | [GitHub](https://github.com/turingmotors/NuScenes-MQA) |
| [Is it safe to cross? Interpretable Risk Assessment with GPT-4V for Safety-Aware Street Crossing](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10597464) | 2024-02 | UR 2024 | - |
| [Multi-Frame, Lightweight & Efficient Vision-Language Models for Question Answering in Autonomous Driving](https://arxiv.org/abs/2403.19838) | 2024-03 | VLADR 2024 | [GitHub](https://github.com/akshaygopalkr/EM-VLM4AD) |
| [LATTE: A Real-time Lightweight Attention-based Traffic Accident Anticipation Engine](https://arxiv.org/abs/2504.04103) | 2024-04 | arXiv | - |
| [OmniDrive: A Holistic Vision-Language Dataset for Autonomous Driving with Counterfactual Reasoning](https://arxiv.org/abs/2405.01533) | 2024-05 | CVPR 2025 | [GitHub](https://github.com/NVlabs/OmniDrive) |
| [Reason2Drive: Towards Interpretable and Chain-based Reasoning for Autonomous Driving](https://arxiv.org/abs/2312.03661) | 2024-06 | ECCV 2024 | [GitHub](https://github.com/fudan-zvg/reason2drive) |
| [Large Language Models Powered Context-aware Motion Prediction in Autonomous Driving](https://arxiv.org/abs/2403.11057) | 2024-07 | IROS 2024 | [GitHub](https://github.com/AIR-DISCOVER/LLM-Augmented-MTR) |
| [DriveGenVLM: Real-world Video Generation for Vision Language Model based Autonomous Driving](https://ieeexplore.ieee.org/abstract/document/10786438) | 2024-08 | IAVVC 2024 | - |
| [Think-Driver: From Driving-Scene Understanding to Decision-Making with Vision Language Models](https://mllmav.github.io/papers/Think-Driver:%20From%20Driving-Scene%20Understanding%20to%20Decision-Making%20with%20Vision%20Language%20Models.pdf) | 2024-09 | ECCV 2024 Workshop | - |
| [Generating Out-Of-Distribution Scenarios Using Language Models](https://arxiv.org/abs/2411.16554) | 2024-11 | arXiv | - |
| [Automated Evaluation of Large Vision-Language Models on Self-driving Corner Cases](https://arxiv.org/abs/2404.10595) | 2024-12 | WACV 2025 | [GitHub](https://coda-dataset.github.io/coda-lm/) |
| [SFF Rendering-Based Uncertainty Prediction using VisionLLM](https://openreview.net/forum?id=q8ptjh1pDl) | 2024-12 | AAAI 2025 Workshop LM4Plan | - |
| [Are VLMs Ready for Autonomous Driving? An Empirical Study from the Reliability, Data, and Metric Perspectives](https://arxiv.org/abs/2501.04003) | 2025-01 | arXiv | [GitHub](https://drive-bench.github.io/) |
| [Enhancing Large Vision Model in Street Scene Semantic Understanding through Leveraging Posterior Optimization Trajectory](https://arxiv.org/abs/2501.01710) | 2025-01 | arXiv | - |
| [DriveLM: Driving with Graph Visual Question Answering](https://arxiv.org/abs/2312.14150) | 2025-01 | ECCV 2024 | [GitHub](https://github.com/OpenDriveLab/DriveLM) |
| [Scenario Understanding of Traffic Scenes Through Large Visual Language Models](https://arxiv.org/pdf/2501.17131) | 2025-01 | WACV 2025 | - |
| [INSIGHT: Enhancing Autonomous Driving Safety through Vision-Language Models on Context-Aware Hazard Detection and Edge Case Evaluation](https://www.arxiv.org/abs/2502.00262) | 2025-02 | arXiv | - |
| [CurricuVLM: Towards Safe Autonomous Driving via Personalized Safety-Critical Curriculum Learning with Vision-Language Models](https://arxiv.org/abs/2502.15119) | 2025-02 | arXiv | [GitHub](https://zihaosheng.github.io/CurricuVLM/) |
| [Evaluating Multimodal Vision-Language Model Prompting Strategies for Visual Question Answering in Road Scene Understanding](https://openaccess.thecvf.com/content/WACV2025W/LLVMAD/html/Keskar_Evaluating_Multimodal_Vision-Language_Model_Prompting_Strategies_for_Visual_Question_Answering_WACVW_2025_paper.html) | 2025-02 | WACV workshop 2025 | - |
| [NuGrounding: A Multi-View 3D Visual Grounding Framework in Autonomous Driving](https://arxiv.org/abs/2503.22436) | 2025-03 | arXiv | - |
| [AutoDrive-QA- Automated Generation of Multiple-Choice Questions for Autonomous Driving Datasets Using Large Vision-Language Models](https://arxiv.org/abs/2503.15778) | 2025-03 | arXiv | [GitHub](https://github.com/Boshrakh/AutoDrive-QA) |
| [DriveLMM-o1: A Step-by-Step Reasoning Dataset and Large Multimodal Model for Driving Scenario Understanding](https://arxiv.org/abs/2503.10621) | 2025-03 | arXiv | [GitHub](https://github.com/ayesha-ishaq/DriveLMM-o1) |
| [Vision Foundation Model Embedding-Based Semantic Anomaly Detection](https://arxiv.org/abs/2505.07998) | 2025-05 | ICRA 2025 Workshop | - |
| [OpenLKA: An Open Dataset of Lane Keeping Assist from Recent Car Models under Real-world Driving Conditions](https://arxiv.org/abs/2505.09092) | 2025-05 | arXiv | [GitHub](https://github.com/OpenLKA/OpenLKA) |
| [Bridging Human Oversight and Black-box Driver Assistance: Vision-Language Models for Predictive Alerting in Lane Keeping Assist systems](https://arxiv.org/abs/2505.11535) | 2025-05 | arXiv | - |

</details>

## 🌟 Multimodal Large Language Models for Autonomous Driving

<details>
<summary><strong>Scenario Generation (MLLM)</strong></summary>

| Paper | Date | Venue | Code | 
|:------|:-----|:------|:-----|
| [Realistic Corner Case Generation for Autonomous Vehicles with Multimodal Large Language Model](https://arxiv.org/pdf/2412.00243) | 2024-11 | arXiv | - |
| [LMM-enhanced Safety-Critical Scenario Generation for Autonomous Driving System Testing From Non-Accident Traffic Videos](https://arxiv.org/pdf/2406.10857) | 2025-01 | arXiv | [GitHub](https://anonymous.4open.science/r/CRISER/README.md) |

</details>

<details>
<summary><strong>Scenario Analysis (MLLM)</strong></summary>

| Paper | Date | Venue | Code | 
|:------|:-----|:------|:-----|
| [DriveGPT4: Interpretable End-to-end Autonomous Driving via Large Language Model](https://arxiv.org/abs/2310.01412) | 2023-10 | IEEE Robotics and Automation Letters 2024 | [GitHub](https://tonyxuqaq.github.io/projects/DriveGPT4/) |
| [Dolphins: Multimodal Language Model for Driving](https://arxiv.org/abs/2312.00438) | 2023-12 | ECCV 2024 | [GitHub](https://vlm-driver.github.io/) |
| [AccidentGPT: Accident analysis and prevention from V2X Environmental Perception with Multi-modal Large Model](https://arxiv.org/abs/2312.13156) | 2023-12 | IV 2024 | [GitHub](https://deepaccident.github.io/) |
| [Lidar-llm: Exploring the potential of large language models for 3d lidar understanding](https://arxiv.org/abs/2312.14074) | 2023-12 | AAAI 2025 | [GitHub](https://sites.google.com/view/lidar-llm) |
| [LingoQA: Visual Question Answering for Autonomous Driving](https://arxiv.org/abs/2312.14115) | 2023-12 | ECCV 2024 | [GitHub](https://github.com/wayveai/LingoQA) |
| [Holistic Autonomous Driving Understanding by Bird's-Eye-View Injected Multi-Modal Large Models](https://arxiv.org/abs/2401.00988) | 2024-01 | CVPR 2024 | [GitHub](https://github.com/xmed-lab/NuInstruct) |
| [MAPLM: A Real-World Large-Scale Vision-Language Benchmark for Map and Traffic Scene Understanding](https://openaccess.thecvf.com/content/CVPR2024/papers/Cao_MAPLM_A_Real-World_Large-Scale_Vision-Language_Benchmark_for_Map_and_Traffic_CVPR_2024_paper.pdf) | 2024-01 | CVPR 2024 | [GitHub](https://github.com/LLVM-AD/MAPLM) |
| [WTS: A Pedestrian-Centric Traffic Video Dataset for Fine-Grained Spatial-Temporal Understanding](https://arxiv.org/abs/2407.15350) | 2024-06 | ECCV 2024 | [GitHub](https://woven-visionai.github.io/wts-dataset-homepage/) |
| [Semantic Understanding of Traffic Scenes with Large Vision Language Models](https://ieeexplore.ieee.org/document/10588373) | 2024-06 | IV 2024 | [GitHub](https://github.com/sandeshrjain/lvlm-scene) |
| [VLAAD: Vision and Language Assistant for Autonomous Driving](https://ieeexplore.ieee.org/document/10495690) | 2024-06 | WACVW 2024 | [GitHub](https://ieeexplore.ieee.org/document/10495690) |
| [InternDrive: A Multimodal Large Language Model for Autonomous Driving Scenario Understanding](https://dl.acm.org/doi/10.1145/3690931.3690982) | 2024-07 | AIAHPC 2024 | - |
| [LingoQA: Visual Question Answering for Autonomous Driving](https://arxiv.org/abs/2312.14115) | 2024-09 | ECCV 2024 | [GitHub](https://github.com/wayveai/LingoQA) |
| [Using Multimodal Large Language Models for Automated Detection of Traffic Safety Critical Events](https://www.mdpi.com/2624-8921/6/3/74) | 2024-09 | Vehicles 2024 MDPI | - |
| [MLLM-SUL: Multimodal Large Language Model for Semantic Scene Understanding and Localization in Traffic Scenarios](https://arxiv.org/abs/2412.19406) | 2024-12 | arXiv | [GitHub](https://github.com/fjq-tongji/MLLM-SUL) |
| [TUMTraffic-VideoQA: A Benchmark for Unified Spatio-Temporal Video Understanding in Traffic Scenes](https://arxiv.org/abs/2502.02449) | 2025-02 | ICML 2025 | [GitHub](https://arxiv.org/abs/2502.02449) |
| [ScVLM: Enhancing Vision-Language Model for Safety-Critical Event Understanding](https://openaccess.thecvf.com/content/WACV2025W/LLVMAD/html/Shi_ScVLM_Enhancing_Vision-Language_Model_for_Safety-Critical_Event_Understanding_WACVW_2025_paper.html) | 2025-02 | WACV Workshop 2025 | [GitHub](https://github.com/datadrivenwheels/ScVLM) |
| [HiLM-D: Enhancing MLLMs with Multi-Scale High-Resolution Details for Autonomous Driving](https://arxiv.org/abs/2309.05186) | 2025-03 | International Journal of Computer Vision | - |
| [NuPlanQA: A Large-Scale Dataset and Benchmark for Multi-View Driving Scene Understanding in Multi-Modal Large Language Models](https://arxiv.org/abs/2503.12772) | 2025-03 | arXiv | - |
| [Tracking Meets Large Multimodal Models for Driving Scenario Understanding](https://arxiv.org/abs/2503.14498) | 2025-03 | arXiv | [GitHub](https://github.com/mbzuai-oryx/TrackingMeetsLMM) |
| [V3LMA: Visual 3D-enhanced Language Model for Autonomous Driving](https://arxiv.org/abs/2505.00156) | 2025-04 | arXiv | - |
| [Are Vision LLMs Road-Ready? A Comprehensive Benchmark for Safety-Critical Driving Video Understanding](https://arxiv.org/abs/2504.14526) | 2025-04 | arXiv | [GitHub](https://github.com/tong-zeng/DVBench) |

</details>

## 🌟 Diffusion Models for Autonomous Driving

<details>
<summary><strong>Scenario Generation (Diffusion Models)</strong></summary>

| Paper | Date | Venue | Code |
|:------|:-----|:------|:-----|
| [Guided Conditional Diffusion for Controllable Traffic Simulation](https://aiasd.github.io/ctg.github.io/) | 2022-10 | ICRA 2023 | [GitHub](https://github.com/NVlabs/CTG) |
| [Generating Driving Scenes with Diffusion](https://arxiv.org/abs/2305.18452) | 2023-05 | arXiv | - |
| [DiffScene: Guided Diffusion Models for Safety-Critical Scenario Generation](https://openreview.net/forum?id=hclEbdHida) | 2023-06 | AdvML-Frontiers 2023 | - |
| [BEVControl: Accurately Controlling Street-view Elements with Multi-perspective Consistency via BEV Sketch Layout](https://arxiv.org/abs/2308.01661) | 2023-09 | arXiv | - |
| [DriveSceneGen: Generating Diverse and Realistic Driving Scenarios From Scratch](https://arxiv.org/abs/2309.14685) | 2023-09 | IEEE Robotics and Automation Letters 2024 | - |
| [MagicDrive: Street View Generation with Diverse 3D Geometry Control](https://arxiv.org/abs/2310.02601) | 2023-10 | ICLR 2024 | [GitHub](https://github.com/cure-lab/MagicDrive) |
| [DrivingDiffusion: Layout-Guided multi-view driving scene video generation with latent diffusion model](https://arxiv.org/abs/2310.07771) | 2023-10 | ECCV 2024 | - |
| [Language-guided traffic simulation via scene-level diffusion](https://research.nvidia.com/labs/avg/publication/zhong.rempe.etal.corl23/) | 2023-11 | CoRL 2023 | - |
| [Scenario Diffusion: Controllable Driving Scenario Generation With Diffusion](https://neurips.cc/virtual/2023/poster/72611) | 2023-11 | NeurIPS 2023 | - |
| [Panacea: Panoramic and Controllable Video Generation for Autonomous Driving](https://arxiv.org/abs/2311.16813) | 2023-11 | CVPR 2024 | [GitHub](https://github.com/wenyuqing/panacea) |
| [SAFE-SIM: Safety-Critical Closed-Loop Traffic Simulation with Diffusion-Controllable Adversaries](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/03157-supp.pdf) | 2023-12 | ECCV 2024 | [GitHub](https://github.com/jxmmy7777/safe-sim) |
| [Text2Street: Controllable Text-to-image Generation for Street Views](https://arxiv.org/abs/2402.04504) | 2024-02 | ICPR 2024 | - |
| [GEODIFFUSION: Text-Prompted Geometric Control for Object Detection Data Generation](https://arxiv.org/pdf/2306.04607) | 2024-02 | LCLR 2024 | [GitHub](https://kaichen1998.github.io/projects/geodiffusion/) |
| [GenDDS: Generating Diverse Driving Video Scenarios with Prompt-to-Video Generative Model](https://arxiv.org/abs/2408.15868) | 2024-04 | ITSC 2024 | - |
| [Versatile Behavior Diffusion for Generalized Traffic Agent Simulation](https://arxiv.org/abs/2404.02524) | 2024-04 | RSS 2024 | [GitHub](https://github.com/SafeRoboticsLab/VBD) |
| [SceneControl: Diffusion for Controllable Traffic Scene Generation](https://waabi.ai/scenecontrol/) | 2024-05 | ICRA 2024 | - |
| [SLEDGE: Synthesizing Driving Environments with Generative Models and Rule-Based Traffic](https://arxiv.org/abs/2403.17933) | 2024-07 | ECCV 2024 | [GitHub](https://github.com/autonomousvision/sledge) |
| [DrivingGen: Efficient Safety-Critical Driving Video Generation with Latent Diffusion Models](https://ieeexplore.ieee.org/document/10688119) | 2024-07 | ICME 2024 | - |
| [AdvDiffuser: Generating Adversarial Safety-Critical Driving Scenarios via Guided Diffusion](https://ieeexplore.ieee.org/abstract/document/10802408) | 2024-10 | IROS 2023 | - |
| [Data-driven Diffusion Models for Enhancing Safety in Autonomous Vehicle Traffic Simulations](https://arxiv.org/abs/2410.04809) | 2024-10 | arXiv | - |
| [DiffRoad: Realistic and Diverse Road Scenario Generation for Autonomous Vehicle Testing](https://arxiv.org/abs/2411.09451) | 2024-11 | arXiv | - |
| [SceneDiffuser: Efficient and Controllable Driving Simulation Initialization and Rollout](https://arxiv.org/pdf/2412.12129) | 2024-12 | NeurIPS 2024 | [GitHub](https://scenediffuser.github.io/) |
| [Direct Preference Optimization-Enhanced Multi-Guided Diffusion Model for Traffic Scenario Generation](https://arxiv.org/abs/2502.12178) | 2025-02 | arXiv | - |
| [Causal Composition Diffusion Model for Closed-loop Traffic Generation](https://arxiv.org/abs/2412.17920) | 2025-02 | arXiv | - |
| [AVD2: Accident Video Diffusion for Accident Video Description](https://arxiv.org/pdf/2502.14801) | 2025-03 | ICRA 2025 | [GitHub](https://github.com/An-Answer-tree/AVD2) |
| [DualDiff+: Dual-Branch Diffusion for High-Fidelity Video Generation with Reward Guidance](https://arxiv.org/abs/2503.03689) | 2025-03 | arXiv | - |
| [Scenario Dreamer: Vectorized Latent Diffusion for Generating Driving Simulation Environments](https://arxiv.org/abs/2503.22496) | 2025-03 | arXiv | - |
| [DriveGen: Towards Infinite Diverse Traffic Scenarios with Large Models](https://arxiv.org/abs/2503.05808) | 2025-03 | arXiv | - |
| [DiVE: Efficient Multi-View Driving Scenes Generation Based on Video Diffusion Transformer](https://arxiv.org/abs/2504.00000) | 2025-04 | arXiv | - |
| [DualDiff: Dual-branch Diffusion Model for Autonomous Driving with Semantic Fusion](https://www.arxiv.org/abs/2505.01857) | 2025-05 | arXiv | - |
| [LD-Scene: LLM-Guided Diffusion for Controllable Generation of Adversarial Safety-Critical Driving Scenarios](https://arxiv.org/abs/2505.00000) | 2025-05 | arXiv | - |
| [Dual-Conditioned Temporal Diffusion Modeling for Driving Scene Generation](https://zzzura-secure.duckdns.org/dctdm) | 2025-05 | ICAR 2025 | [GitHub](https://github.com/PeteBai/DcTDM) |

</details>

<details>
<summary><strong>Scenario Analysis (Diffusion Models)</strong></summary>

| Paper | Date | Venue | Code |
|:------|:-----|:------|:-----|
| [AVD2: Accident Video Diffusion for Accident Video Description](https://arxiv.org/pdf/2502.14801) | 2025-03 | ICRA 2025 | [GitHub](https://github.com/An-Answer-tree/AVD2) |

</details>

## 🌟 World Models for Autonomous Driving

<details>
<summary><strong>World Models for Autonomous Driving</strong></summary>

| Paper | Date | Venue | Code | Application |
|:------|:-----|:------|:-----|:------------|
| [DriveDreamer: Towards Real-world-driven World Models for Autonomous Driving](https://arxiv.org/abs/2309.09777) | 2023-09 | ECCV 2024 | [GitHub](https://github.com/JeffWang987/DriveDreamer) | Scenario Generation |
| [GAIA-1: A Generative World Model for Autonomous Driving](https://arxiv.org/abs/2309.17080) | 2023-09 | arXiv Wayve | - | Scenario Generation |
| [Copilot4D: Learning Unsupervised World Models for Autonomous Driving via Discrete Diffusion](https://arxiv.org/abs/2311.01017) | 2023-11 | ICLR 2024 | - | Scenario Generation |
| [MUVO: A Multimodal Generative World Model for Autonomous Driving with Geometric Representations](https://arxiv.org/abs/2311.11762) | 2023-11 | IV 2025 | - | Scenario Generation |
| [Driving into the Future: Multiview Visual Forecasting and Planning with World Model for Autonomous Driving](https://arxiv.org/abs/2311.17918) | 2023-11 | CVPR 2024 | [GitHub](https://github.com/BraveGroup/Drive-WM) | Scenario Generation |
| [Vista: A Generalizable Driving World Model with High Fidelity and Versatile Controllability](https://arxiv.org/abs/2405.17398) | 2024-03 | NeurIPS 2024 | - | Scenario Generation |
| [MagicDrive: Street View Generation with Diverse 3D Geometry Control](https://arxiv.org/abs/2310.02601) | 2024-05 | arXiv | - | Scenario Generation |
| [DriveDreamer-2: LLM-Enhanced World Models for Diverse Driving Video Generation](https://arxiv.org/abs/2403.06845) | 2024-05 | AAAI 2025 | [GitHub](https://github.com/f1yfisher/DriveDreamer2) | Scenario Generation |
| [UniScene: Multi-Camera Unified Pre-training via 3D Scene Reconstruction for Autonomous Driving](https://arxiv.org/abs/2305.18829) | 2024-08 | RAL 2024 | - | Scenario Generation |
| [WoVoGen: World Volume-aware Diffusion for Controllable Multi-camera Driving Scene Generation](https://arxiv.org/abs/2312.02934) | 2024-08 | ECCV 2024 | - | Scenario Generation |
| [Panacea+: Panoramic and Controllable Video Generation for Autonomous Driving](https://arxiv.org/abs/2408.07605) | 2024-08 | CVPR 2024 | [GitHub](https://github.com/wenyuqing/panacea) | Scenario Generation |
| [DriveArena: A Closed-loop Generative Simulation Platform for Autonomous Driving](https://arxiv.org/abs/2408.00415) | 2024-08 | arXiv | [GitHub](https://github.com/PJLab-ADG/DriveArena) | Scenario Generation |
| [DriVerse: Navigation World Model for Driving Simulation via Multimodal Trajectory Prompting and Motion Alignment](https://arxiv.org/abs/2504.18576) | 2024-08 | arXiv | - | Scenario Generation |
| [DriveDreamer4D: World Models Are Effective Data Machines for 4D Driving Scene Representation](https://arxiv.org/abs/2410.13571) | 2024-11 | CVPR 2025 | [GitHub](https://github.com/GigaAI-research/DriveDreamer4D) | Scenario Generation |
| [ReconDreamer: Crafting World Models for Driving Scene Reconstruction via Online Restoration](https://arxiv.org/abs/2411.19548) | 2024-11 | arXiv | - | Scenario Generation |
| [MagicDrive3D: Controllable 3D Generation for Any-View Rendering in Street Scenes](https://arxiv.org/abs/2405.14475) | 2024-11 | arXiv | [GitHub](https://gaoruiyuan.com/magicdrive3d/) | Scenario Generation |
| [MagicDriveDiT: High-Resolution Long Video Generation for Autonomous Driving with Adaptive Control](https://arxiv.org/abs/2411.13807) | 2024-11 | arXiv | [GitHub](https://gaoruiyuan.com/magicdrive-v2/) | Scenario Generation |
| [ACT-Bench: Towards Action Controllable World Models for Autonomous Driving](https://arxiv.org/abs/2412.05337) | 2024-12 | arXiv | - | Scenario Generation |
| [GEM: A Generalizable Ego-Vision Multimodal World Model for Fine-Grained Ego-Motion, Object Dynamics, and Scene Composition Control](https://arxiv.org/abs/2412.11198) | 2024-12 | CVPR 2025 | [GitHub](https://github.com/vita-epfl/GEM) | Scenario Generation |
| [SceneDiffuser++: City-Scale Traffic Simulation via a Generative World Model](https://arxiv.org/abs/2412.12129) | 2024-12 | CVPR 2025 | - | Scenario Generation |
| [DrivingWorld: Constructing World Model for Autonomous Driving via Video GPT](https://arxiv.org/abs/2412.19505) | 2024-12 | arXiv | [GitHub](https://github.com/YvanYin/DrivingWorld) | Scenario Generation |
| [Driving in the Occupancy World: Vision-Centric 4D Occupancy Forecasting and Planning via World Models for Autonomous Driving](https://arxiv.org/abs/2408.14197) | 2025-01 | AAAI 2025 | [GitHub](https://github.com/yuyang-cloud/Drive-OccWorld) | Scenario Generation |
| [DualDiff+: Dual-Branch Diffusion for High-Fidelity Video Generation with Reward Guidance](https://arxiv.org/abs/2503.03689) | 2025-03 | ICRA 2025 | [GitHub](https://github.com/yangzhaojason/DualDiff) | Scenario Generation |
| [Cosmos-Reason1: From Physical Common Sense To Embodied Reasoning](https://arxiv.org/abs/2503.15558) | 2025-03 | arXiv | [GitHub](https://github.com/nvidia-cosmos/cosmos-reason1) | Scenario Generation |
| [GAIA-2: A Controllable Multi-View Generative World Model for Autonomous Driving](https://arxiv.org/abs/2503.20523) | 2025-03 | arXiv | - | Scenario Generation |
| [Cosmos-Transfer1: Conditional World Generation with Adaptive Multimodal Control](https://arxiv.org/abs/2503.14492) | 2025-04 | arXiv | [GitHub](https://github.com/nvidia-cosmos/cosmos-transfer1) | Scenario Generation |
| [OccSora: 4D Occupancy Generation Models as World Simulators for Autonomous Driving](https://arxiv.org/abs/2405.20337) | 2025-05 | arXiv | - | Scenario Generation |
| [PosePilot: Steering Camera Pose for Generative World Models with Self-supervised Depth](https://arxiv.org/abs/2505.00000) | 2025-05 | arXiv | - | Scenario Generation |

</details>

## 🌟 Datasets Comparison

<details>
<summary><strong>Datasets Comparison</strong></summary>

| Dataset | Year | Img | View | Real | Lidar | Radar | Traj | 3D | 2D | Lane | Weather | Time | Region | Company |
|:--------|:-----|:----|:-----|:-----|:------|:------|:-----|:---|:---|:-----|:--------|:-----|:-------|:--------|
| [CamVid](https://service.tib.eu/ldmservice/dataset/camvid-dataset) | 2009 | RGB | FPV | ✔ | ✖️ | ✖️ | ✖️ | ✖️ | ✔ | ✔ | ✔ | D | U | - |
| [KITTI](https://www.cvlibs.net/datasets/kitti/) | 2013 | RGB/S | FPV | ✔ | ✔ | ✖️ | ✔ | ✔ | ✔ | ✔ | ✔ | D | U/R/H | - |
| [Cyclists](https://www.ifi-mec.tu-clausthal.de/ctv-dataset) | 2016 | RGB | FPV | ✔ | ✖️ | ✖️ | ✖️ | ✖️ | ✖️ | ✖️ | ✖️ | D | U | - |
| [Cityscapes](https://www.cityscapes-dataset.com/login/) | 2016 | RGB/S | FPV | ✔ | ✖️ | ✖️ | ✖️ | ✔ | ✔ | ✔ | ✖️ | D | U | - |
| [SYNTHIA](https://service.tib.eu/ldmservice/dataset/bibtex/synthia) | 2016 | RGB | FPV | ✖️ | ✖️ | ✖️ | ✖️ | ✔ | ✔ | ✔ | ✔ | D/N | U | - |
| [Campus](https://paperswithcode.com/dataset/campus-shelf) | 2016 | RGB | BEV | ✖️ | ✖️ | ✖️ | ✖️ | ✖️ | ✖️ | ✖️ | ✖️ | D | C | - |
| [RobotCar](https://universe.roboflow.com/robotcar-lnnmb/robotcar-kj2cb) | 2016 | RGB | FPV | ✔ | ✖️ | ✖️ | ✖️ | ✖️ | ✖️ | ✖️ | ✖️ | D/N | U | - |
| [Mapillary](https://www.kaggle.com/c/mapillary-vistas-detection-challenge/data) | 2017 | RGB | FPV | ✔ | ✖️ | ✖️ | ✖️ | ✖️ | ✔ | ✔ | ✔ | D/N | U | - |
| [P.F.B.](https://scispace.com/pdf/brno-urban-dataset-the-new-data-for-self-driving-agents-and-3songw9bsn.pdf) | 2017 | RGB | FPV | ✔ | ✖️ | ✖️ | ✖️ | ✖️ | ✔ | ✔ | ✔ | D/N | U | - |
| [BDD100K](https://datasetninja.com/bdd100k) | 2018 | RGB | FPV | ✔ | ✖️ | ✖️ | ✖️ | ✔ | ✔ | ✔ | ✔ | D | U/H | - |
| [HighD](https://levelxdata.com/highd-dataset/) | 2018 | RGB | BEV | ✔ | ✖️ | ✖️ | ✖️ | ✖️ | ✔ | ✔ | ✖️ | D | H | - |
| [Udacity](https://www.kaggle.com/datasets/evilspirit05/cocococo-dataset) | 2018 | RGB | FPV | ✔ | ✖️ | ✖️ | ✖️ | ✖️ | ✖️ | ✖️ | ✖️ | D | U | - |
| [KAIST](https://msc.kaist.ac.kr/bbs/board.php?bo_table=CAG&amp;wr_id=25) | 2018 | RGB/S | FPV | ✔ | ✔ | ✖️ | ✖️ | ✖️ | ✔ | ✔ | ✔ | D/N | U | - |
| [Argoverse](https://docs.ultralytics.com/datasets/detect/argoverse/) | 2019 | RGB/S | FPV | ✔ | ✔ | ✖️ | ✖️ | ✖️ | ✔ | ✔ | ✔ | D/N | U | - |
| [TRAF](https://tum-traffic-dataset.github.io/tumtraf-v2x/) | 2019 | RGB | FPV | ✔ | ✖️ | ✖️ | ✖️ | ✖️ | ✔ | ✔ | ✔ | D | U | - |
| [ApolloScape](https://service.tib.eu/ldmservice/dataset/305aef79-4191-4717-bb6a-496fa1f5ac4c) | 2019 | RGB/S | FPV | ✔ | ✖️ | ✖️ | ✖️ | ✔ | ✔ | ✔ | ✔ | D | U | - |
| [ACFR](https://datasetninja.com/acfr-multifruit-2016) | 2019 | RGB | BEV | ✔ | ✖️ | ✖️ | ✖️ | ✖️ | ✖️ | ✖️ | ✖️ | D | RA | - |
| [H3D](https://paperswithcode.com/dataset/h3d) | 2019 | RGB | FPV | ✔ | ✖️ | ✖️ | ✖️ | ✖️ | ✔ | ✔ | ✔ | D | U | - |
| [INTERACTION](https://interaction-dataset.com/) | 2019 | RGB | BEV | ✔ | ✖️ | ✖️ | ✖️ | ✖️ | ✖️ | ✖️ | ✖️ | D | I/RA | - |
| [Comma2k19](https://github.com/commaai/comma2k19) | 2019 | RGB | FPV | ✔ | ✖️ | ✖️ | ✔ | ✔ | ✖️ | ✖️ | ✖️ | D/N | U/S/R/H | - |
| [InD](https://www.ind-dataset.com/) | 2020 | RGB | BEV | ✔ | ✖️ | ✖️ | ✖️ | ✖️ | ✖️ | ✖️ | ✖️ | D | I | - |
| [RounD](https://levelxdata.com/round-dataset/) | 2020 | RGB | BEV | ✔ | ✖️ | ✖️ | ✖️ | ✖️ | ✖️ | ✖️ | ✖️ | D | RA | - |
| [nuScenes](https://www.nuscenes.org/nuscenes) | 2020 | RGB | FPV | ✔ | ✔ | ✔ | ✖️ | ✔ | ✔ | ✔ | ✔ | D/N | U | - |
| [Lyft Level 5](https://hyper.ai/en/datasets/9036) | 2020 | RGB | FPV | ✔ | ✔ | ✔ | ✖️ | ✔ | ✔ | ✔ | ✔ | D/N | U/S | - |
| [Waymo Open](https://waymo.com/open/download/) | 2020 | RGB | FPV | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | D/N | U | - |
| [A*3D](https://hyper.ai/en/datasets/17161) | 2020 | RGB | FPV | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | D/N | U | - |
| [RobotCar Radar](https://oxford-robotics-institute.github.io/oord-dataset/) | 2020 | RGB | FPV | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | D/N | U | - |
| [Toronto3D](https://onedrive.live.com/?authkey=%21AKEpLxU5CWVW%2DPg&id=E9CE176726EB5C69%216398&cid=E9CE176726EB5C69&parId=root&parQt=sharedby&o=OneUp) | 2020 | RGB | BEV | ✔ | ✔ | ✖️ | ✔ | ✔ | ✖️ | ✔ | ✖️ | D/N | U | University of Waterloo |
| [A2D2](https://registry.opendata.aws/aev-a2d2/) | 2020 | RGB | FPV | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✖️ | ✔ | ✔ | D | U/H/S/R | Audi |
| [WADS](https://bitbucket.org/autonomymtu/dsor_filter) | 2020 | RGB | FPV | ✔ | ✔ | ✔ | ✔ | ✔ | ✖️ | ✖️ | ✔ | D/N | U/S/R | Michigan Technological University |
| [Argoverse 2](https://www.argoverse.org/av2.html) | 2021 | RGB/S | FPV | ✔ | ✔ | ✖️ | ✖️ | ✔ | ✔ | ✔ | ✔ | D/N | U | - |
| [PandaSet](https://scale.com/open-datasets/pandaset) | 2021 | RGB | FPV | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | D/N | U | - |
| [ONCE](https://www.once-for-auto-driving.com) | 2021 | RGB | FPV | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | D/N | U | - |
| [Leddar PixSet](https://leddartech.com/datasets/leddarpixset-download-form/) | 2021 | RGB | FPV | ✔ | ✔ | ✖️ | ✔ | ✔ | ✔ | ✖️ | ✔ | D/N | U/S/R | Leddar |
| [ZOD](https://zod.zenseact.com/) | 2022 | RGB | FPV | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | D/N | U/R/S/H | Zenseact |
| [IDD-3D](https://idd3d.github.io/) | 2022 | RGB | FPV | ✔ | ✔ | ✖️ | ✖️ | ✔ | ✔ | ✖️ | ✖️ | - | R | INAI |
| [CODA](https://paperswithcode.com/dataset/coda) | 2022 | RGB | FPV | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | D/N | U/S/R | Huawei |
| [SHIFT](https://vis.xyz/shift) | 2022 | RGB | FPV | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | D/N | U/S/R/H | ETH Zürich |
| [DeepAccident](https://arxiv.org/html/2304.01168v5) | 2023 | RGB/S | FPV/BEV | ✖️ | ✔ | ✖️ | ✖️ | ✔ | ✔ | ✔ | ✔ | D/N | U/S/R/H | HKU, Huawei, CARLA |
| [Dual_Radar](https://github.com/adept-thu/Dual-Radar) | 2023 | RGB | FPV | ✔ | ✔ | ✔ | ✔ | ✔ | ✖️ | ✔ | ✔ | D/N | U | Tsinghua University |
| [V2V4Real](https://mobility-lab.seas.ucla.edu/v2v4real/) | 2023 | RGB | FPV | ✔ | ✔ | ✖️ | ✔ | ✔ | ✖️ | ✔ | ✖️ | - | U/H/S | UCLA Mobility Lab |
| [SCaRL](https://fhr-ihs-sva.pages.fraunhofer.de/asp/scarl/) | 2024 | RGB/S | FPV/BEV | ✖️ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | D/N | U/S/R/H | Fraunhofer CARLA |
| [MARS](https://data.nasa.gov/dataset/ai4mars-a-dataset-for-terrain-aware-autonomous-driving-on-mars) | 2024 | RGB | FPV | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | D/N | U/S/H | NYU, MAY Mobility |
| [Scenes101](https://wayve.ai/science/wayvescenes101/) | 2024 | RGB | FPV | ✔ | ✖️ | ✖️ | ✔ | ✖️ | ✖️ | ✔ | ✔ | D/N | U/S/R/H | Wayve |
| TruckScenes | 2025 | RGB | FPV | ✔ | ✔ | ✔ | ✔ | ✔ | ✖️ | ✔ | ✔ | D/N | H/U | MAN |

*Notes: View: FPV=First-Person, BEV=Bird's-Eye; Time: D=Day, N=Night; Region: U=Urban, R=Rural, H=Highway, S=Suburban, C=Campus, I=Intersection, RA=Road Area; Img: RGB/S=RGB+Stereo*

</details>

## 🌟 Simulators

<details>
<summary><strong>Simulators</strong></summary>

| Simulator | Year | Back-end | Open Source | Realistic Perception | Custom Scenario | Real World Map | Human Design Map | Python API | C++ API | ROS API | Company |
|:----------|:-----|:---------|:------------|:---------------------|:----------------|:---------------|:-----------------|:-----------|:--------|:--------|:--------|
| [TORCS](http://torcs.sourceforge.net/) | 2000 | None | ✔ | ✔ | ✔ | ✖️ | ✖️ | ✖️ | ✖️ | ✖️ | - |
| [Webots](https://cyberbotics.com/) | 2004 | ODE | ✔ | ✔ | ✔ | ✔ | ✖️ | ✔ | ✔ | ✖️ | - |
| [CarRacing](https://www.gymlibrary.dev/environments/box2d/car_racing/) | 2017 | None | ✔ | ✖️ | ✖️ | ✖️ | ✔ | ✔ | ✖️ | ✖️ | - |
| [CARLA](http://carla.org/) | 2017 | UE4 | ✔ | ✔ | ✔ | ✖️ | ✔ | ✔ | ✔ | ✔ | - |
| [SimMobilityST](https://www.researchgate.net/publication/313289844_SimMobility_Short-term_An_Integrated_Microscopic_Mobility_Simulator) | 2017 | None | ✔ | ✖️ | ✖️ | ✖️ | ✖️ | ✖️ | ✖️ | ✖️ | - |
| [GTA-V](https://github.com/aitorzip/DeepGTAV) | 2017 | RAGE | ✖️ | ✔ | ✖️ | ✖️ | ✖️ | ✖️ | ✖️ | ✖️ | - |
| [highway-env](https://github.com/eleurent/highway-env) | 2018 | None | ✔ | ✖️ | ✔ | ✖️ | ✔ | ✔ | ✖️ | ✖️ | - |
| [Deepdrive](https://deepdrive.io/) | 2018 | UE4 | ✔ | ✔ | ✔ | ✖️ | ✔ | ✔ | ✔ | ✖️ | - |
| [esmini](https://cloe.readthedocs.io/en/latest/reference/plugins/esmini.html) | 2018 | Unity | ✔ | ✖️ | ✖️ | ✖️ | ✖️ | ✔ | ✖️ | ✖️ | - |
| [AutonoViSim](https://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w14/Best_AutonoVi-Sim_Autonomous_Vehicle_CVPR_2018_paper.pdf) | 2018 | PhysX | ✖️ | ✔ | ✔ | ✖️ | ✖️ | ✔ | ✖️ | ✖️ | - |
| [AirSim](https://microsoft.github.io/AirSim/) | 2018 | UE4 | ✔ | ✔ | ✔ | ✖️ | ✔ | ✔ | ✔ | ✖️ | - |
| [SUMO](https://www.eclipse.org/sumo/) | 2018 | None | ✔ | ✖️ | ✔ | ✔ | ✔ | ✖️ | ✔ | ✖️ | - |
| [Apollo](http://apollo.auto/) | 2018 | Unity | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✖️ | - |
| [Sim4CV](https://sim4cv.org/) | 2018 | UE4 | ✔ | ✔ | ✔ | ✖️ | ✔ | ✔ | ✖️ | ✖️ | - |
| [MATLAB](https://www.mathworks.com/products/automated-driving.html) | 2018 | MATLAB | ✖️ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | Mathworks |
| [Scenic](https://scenic-lang.org/) | 2019 | None | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✖️ | ✖️ | Toyota Research Institute, UC Berkeley |
| [SUMMIT](https://github.com/AdaCompNUS/summit) | 2020 | UE4 | ✔ | ✔ | ✔ | ✖️ | ✔ | ✔ | ✔ | ✖️ | - |
| [MultiCarRacing](https://github.com/ananya183/collaborative-multi-car-racing) | 2020 | None | ✔ | ✖️ | ✔ | ✖️ | ✔ | ✔ | ✖️ | ✖️ | - |
| [SMARTS](https://smarts.readthedocs.io/en/latest/) | 2020 | None | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✖️ | ✖️ | - |
| [LGSVL](https://www.lgsvlsimulator.com/) | 2020 | Unity | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | - |
| [CausalCity](https://arxiv.org/html/2306.03354v2) | 2020 | UE4 | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✖️ | ✖️ | - |
| [Vista](https://vista.csail.mit.edu/) | 2020 | None | ✔ | ✔ | ✔ | ✔ | ✖️ | ✔ | ✖️ | ✖️ | MIT |
| [MetaDrive](https://metadriverse.github.io/metadrive/) | 2021 | Panda3D | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✖️ | - |
| [L2R](https://github.com/learn-to-race/l2r) | 2021 | UE4 | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✖️ | - |
| [AutoDRIVE](https://autodrive-ecosystem.github.io/) | 2021 | Unity | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | - |
| [Nuplan](https://www.nuplan.org/) | 2021 | None | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✖️ | ✖️ | Motional |
| [AWSIM](https://autowarefoundation.github.io/AWSIM-Labs/) | 2021 | Unity | ✔ | ✔ | ✔ | ✔ | ✔ | ✖️ | ✖️ | ✔ | Autoware |
| [InterSim](https://tsinghua-mars-lab.github.io/InterSim/) | 2022 | None | ✔ | ✔ | ✔ | ✔ | ✖️ | ✔ | ✖️ | ✖️ | Tsinghua |
| [Nocturne](https://github.com/facebookresearch/nocturne) | 2022 | None | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✖️ | Facebook |
| [BeamNG.tech](https://beamng.tech/) | 2022 | Soft-body physics | ✖️ | ✔ | ✔ | ✖️ | ✔ | ✔ | ✖️ | ✔ | BeamNG GmbH |
| [Waymax](https://github.com/waymo-research/waymax) | 2023 | JAX | ✔ | ✔ | ✔ | ✖️ | ✔ | ✔ | ✖️ | ✖️ | Waymo |
| [UNISim](https://github.com/Sense-X/UNISim) | 2023 | None | ✖️ | ✔ | ✔ | ✔ | ✖️ | ✖️ | ✔ | ✖️ | Waabi |
| [TBSim](https://github.com/NVlabs/traffic-behavior-simulation) | 2023 | None | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✖️ | ✖️ | NVIDIA |
| [Nvidia DriveWorks](https://developer.nvidia.com/drive/driveworks) | 2024 | Nvidia GPU | ✖️ | ✔ | ✔ | ✔ | ✖️ | ✔ | ✔ | ✖️ | NVIDIA |

</details>

## Contributing
We welcome contributions from the community! If you have research papers, tools, or resources to add, please create a pull request or open an issue.

## License
This repository is released under the [Apache 2.0 license](https://github.com/TUM-AVS/FM-AD-Survey/blob/main/LICENSE). 

``` 
