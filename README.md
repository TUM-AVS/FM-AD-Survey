# Foundation Models in Autonomous Driving: A Dual Survey on Scenario Generation and Scenario Analysis :car:
<div align="center">
<a href="https://example.com/paper-to-be-published"><img src="https://img.shields.io/badge/Paper-PDF-red.svg" alt="Paper Badge"/></a>
<a href="https://github.com/TUM-AVS/FM-AV-Survey-Scenario-Generation-Analysis/stargazers"><img src="https://img.shields.io/github/stars/TUM-AVS/FM-AV-Survey-Scenario-Generation-Analysis" alt="Stars Badge"/></a>
<a href="https://github.com/TUM-AVS/FM-AV-Survey-Scenario-Generation-Analysis/network/members"><img src="https://img.shields.io/github/forks/TUM-AVS/FM-AV-Survey-Scenario-Generation-Analysis" alt="Forks Badge"/></a>
<a href="https://github.com/TUM-AVS/FM-AV-Survey-Scenario-Generation-Analysis/pulls"><img src="https://img.shields.io/github/issues-pr/TUM-AVS/FM-AV-Survey-Scenario-Generation-Analysis" alt="Pull Requests Badge"/></a>
<a href="https://github.com/TUM-AVS/FM-AV-Survey-Scenario-Generation-Analysis/issues"><img src="https://img.shields.io/github/issues/TUM-AVS/FM-AV-Survey-Scenario-Generation-Analysis" alt="Issues Badge"/></a>
<a href="https://github.com/TUM-AVS/FM-AV-Survey-Scenario-Generation-Analysis/blob/main/LICENSE"><img src="https://img.shields.io/github/license/TUM-AVS/FM-AV-Survey-Scenario-Generation-Analysis" alt="License Badge"/></a>
</div>

This repository will collect research, implementations, and resources related to **Foundation Models for Scenario Generation and Analysis** in autonomous driving. The repository will be maintained by [TUM-AVS](https://www.mos.ed.tum.de/avs/startseite/) (Professorship of Autonomous Vehicle Systems at Technical University of Munich) and will be continuously updated to track the latest work in the community.

**Keywords:**
- **Foundation Model Types:** Foundation Models, Large Language Models (LLMs), Vision-Language Models (VLMs), Multimodal Large Language Models (MLLMs), Diffusion Models (DMs), World Models (WMs), Generative Models (GMs)
- **Scenario Generation & Analysis:** Scenario Generation, Scenario Simulation, Traffic Simulation, Scenario Testing, Scenario Understanding, Driving Scene Generation, Scene Reasoning, Risk Assessment, Safety-Critical Scenarios, Accident Prediction
- **Application Context:** Autonomous Driving, Self-Driving Vehicles, AV Simulation, Driving Video Generation, Traffic Datasets, Closed-Loop Simulation, Safety Assurance

<p align="center">
<img src="Assets/00_concept_c.png" width="100%" height="auto"/>
</p>

## :fire: Updates
- [May.2024] Repository initialized

## 🤝 &nbsp; Citation
Please visit [Foundation Models in Autonomous Driving: A Dual Survey on Scenario Generation and Scenario Analysis](https://example.com/paper-to-be-published) for more details and comprehensive information (coming soon). If you find our paper and repo helpful, please consider citing it as follows:

```BibTeX
@article{Foundation-Models-AV-Dual-Survey,
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

## 🌟 Diffusion Models for Autonomous Driving

| Paper | Date | Venue | Code | Application |
|:------|:-----|:------|:-----|:------------|
| [Guided Conditional Diffusion for Controllable Traffic Simulation](https://aiasd.github.io/ctg.github.io/) | 2022-10 | ICRA 2023 | [GitHub](https://github.com/NVlabs/CTG) | Scenario Generation |
| [Generating Driving Scenes with Diffusion](https://arxiv.org/abs/2305.18452) | 2023-05 | arXiv | - | Scenario Generation |
| [DiffScene: Guided Diffusion Models for Safety-Critical Scenario Generation](https://openreview.net/forum?id=hclEbdHida) | 2023-06 | AdvML-Frontiers 2023 | - | Scenario Generation |
| [BEVControl: Accurately Controlling Street-view Elements with Multi-perspective Consistency via BEV Sketch Layout](https://arxiv.org/abs/2308.01661) | 2023-09 | arXiv | - | Scenario Generation |
| [DriveSceneGen: Generating Diverse and Realistic Driving Scenarios From Scratch](https://arxiv.org/abs/2309.14685) | 2023-09 | IEEE Robotics and Automation Letters 2024 | - | Scenario Generation |
| [MagicDrive: Street View Generation with Diverse 3D Geometry Control](https://arxiv.org/abs/2310.02601) | 2023-10 | ICLR 2024 | [GitHub](https://github.com/cure-lab/MagicDrive) | Scenario Generation |
| [DrivingDiffusion: Layout-Guided multi-view driving scene video generation with latent diffusion model](https://arxiv.org/abs/2310.07771) | 2023-10 | ECCV 2024 | - | Scenario Generation |
| [Language-guided traffic simulation via scene-level diffusion](https://research.nvidia.com/labs/avg/publication/zhong.rempe.etal.corl23/) | 2023-11 | CoRL 2023 | - | Scenario Generation |
| [Scenario Diffusion: Controllable Driving Scenario Generation With Diffusion](https://neurips.cc/virtual/2023/poster/72611) | 2023-11 | NeurIPS 2023 | - | Scenario Generation |
| [Panacea: Panoramic and Controllable Video Generation for Autonomous Driving](https://arxiv.org/abs/2311.16813) | 2023-11 | CVPR 2024 | [GitHub](https://github.com/wenyuqing/panacea) | Scenario Generation |
| [SAFE-SIM: Safety-Critical Closed-Loop Traffic Simulation with Diffusion-Controllable Adversaries](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/03157-supp.pdf) | 2023-12 | ECCV 2024 | [GitHub](https://github.com/jxmmy7777/safe-sim) | Scenario Generation |
| [Text2Street: Controllable Text-to-image Generation for Street Views](https://arxiv.org/abs/2402.04504) | 2024-02 | ICPR 2024 | - | Scenario Generation |
| [GEODIFFUSION: Text-Prompted Geometric Control for Object Detection Data Generation](https://arxiv.org/pdf/2306.04607) | 2024-02 | ICLR 2024 | [GitHub](https://kaichen1998.github.io/projects/geodiffusion/) | Scenario Generation |
| [GenDDS: Generating Diverse Driving Video Scenarios with Prompt-to-Video Generative Model](https://arxiv.org/abs/2408.15868) | 2024-04 | ITSC 2024 | - | Scenario Generation |
| [Versatile Behavior Diffusion for Generalized Traffic Agent Simulation](https://arxiv.org/abs/2404.02524) | 2024-04 | RSS 2024 | [GitHub](https://github.com/SafeRoboticsLab/VBD) | Scenario Generation |
| [SceneControl: Diffusion for Controllable Traffic Scene Generation](https://waabi.ai/scenecontrol/) | 2024-05 | ICRA 2024 | - | Scenario Generation |
| [DrivingGen: Efficient Safety-Critical Driving Video Generation with Latent Diffusion Models](https://ieeexplore.ieee.org/document/10688119) | 2024-07 | ICME 2024 | - | Scenario Generation |
| [SLEDGE: Synthesizing Driving Environments with Generative Models and Rule-Based Traffic](https://arxiv.org/abs/2403.17933) | 2024-07 | ECCV 2024 | [GitHub](https://github.com/autonomousvision/sledge) | Scenario Generation |
| [Data-driven Diffusion Models for Enhancing Safety in Autonomous Vehicle Traffic Simulations](https://arxiv.org/abs/2410.04809) | 2024-10 | arXiv | - | Scenario Generation |
| [AdvDiffuser: Generating Adversarial Safety-Critical Driving Scenarios via Guided Diffusion](https://ieeexplore.ieee.org/abstract/document/10802408) | 2024-10 | IROS 2023 | - | Scenario Generation |
| [DiffRoad: Realistic and Diverse Road Scenario Generation for Autonomous Vehicle Testing](https://arxiv.org/abs/2411.09451) | 2024-11 | arXiv | - | Scenario Generation |
| [SceneDiffuser: Efficient and Controllable Driving Simulation Initialization and Rollout](https://arxiv.org/pdf/2412.12129) | 2024-12 | NeurIPS 2024 | [GitHub](https://scenediffuser.github.io/) | Scenario Generation |
| [Direct Preference Optimization-Enhanced Multi-Guided Diffusion Model for Traffic Scenario Generation](https://arxiv.org/abs/2502.12178) | 2025-02 | arXiv | - | Scenario Generation |
| [Causal Composition Diffusion Model for Closed-loop Traffic Generation](https://arxiv.org/abs/2412.17920) | 2025-02 | arXiv | - | Scenario Generation |
| [AVD2: Accident Video Diffusion for Accident Video Description](https://arxiv.org/pdf/2502.14801) | 2025-03 | ICRA 2025 | [GitHub](https://github.com/An-Answer-tree/AVD2) | Scenario Generation, Scenario Analysis |
| [DualDiff+: Dual-Branch Diffusion for High-Fidelity Video Generation with Reward Guidance](https://arxiv.org/abs/2503.03689) | 2025-03 | arXiv | - | Scenario Generation |
| [DriveGen: Towards Infinite Diverse Traffic Scenarios with Large Models](https://arxiv.org/abs/2503.05808) | 2025-03 | arXiv | - | Scenario Generation |
| [Scenario Dreamer: Vectorized Latent Diffusion for Generating Driving Simulation Environments](https://arxiv.org/abs/2503.22496) | 2025-03 | arXiv | - | Scenario Generation |
| [DiVE: Efficient Multi-View Driving Scenes Generation Based on Video Diffusion Transformer](https://arxiv.org/abs/2504.DiVE) | 2025-04 | arXiv | - | Scenario Generation |
| [DualDiff: Dual-branch Diffusion Model for Autonomous Driving with Semantic Fusion](https://www.arxiv.org/abs/2505.01857) | 2025-05 | arXiv | - | Scenario Generation |
| [Dual-Conditioned Temporal Diffusion Modeling for Driving Scene Generation](https://zzzura-secure.duckdns.org/dctdm) | 2025-05 | ICAR 2025 | [GitHub](https://github.com/PeteBai/DcTDM) | Scenario Generation |
| [LD-Scene: LLM-Guided Diffusion for Controllable Generation of Adversarial Safety-Critical Driving Scenarios](https://arxiv.org/abs/2505.LD-Scene) | 2025-05 | arXiv | - | Scenario Generation |

## 🌟 Large Language Models for Autonomous Driving

### Scenario Generation (LLM)

| Paper | Date | Venue | Code | 
|:------|:-----|:------|:-----|
| [TARGET: Automated Scenario Generation from Traffic Rules for Testing Autonomous Vehicles](https://arxiv.org/abs/2305.06018) | 2023-05 | arXiv | - |
| [Language Conditioned Traffic Generation](https://arxiv.org/abs/2307.07947) | 2023-07 | CoRL 2023 | [GitHub](https://github.com/Ariostgx/lctgen/) |
| [A Generative AI-driven Application: Use of Large Language Models for Traffic Scenario Generation](https://ieeexplore.ieee.org/document/10415934) | 2023-11 | ELECO 2023 | - |
| [ChatGPT-Based Scenario Engineer: A New Framework on Scenario Generation for Trajectory Prediction](https://ieeexplore.ieee.org/document/10423819) | 2024-02 | IEEE Transactions on Intelligent Vehicles | - |
| [Enhancing Autonomous Vehicle Training with Language Model Integration and Critical Scenario Generation](https://arxiv.org/abs/2404.08570) | 2024-04 | arXiv | [GitHub](https://github.com/zachtian/CRITICAL) |
| [LLMScenario: Large Language Model Driven Scenario Generation](https://ieeexplore.ieee.org/document/10529537) | 2024-05 | IEEE Transactions on Systems, Man, and Cybernetics: Systems | - |
| [Automatic Generation Method for Autonomous Driving Simulation Scenarios Based on Large Language Model](https://link.springer.com/chapter/10.1007/978-981-96-3977-9_10) | 2024-05 | AIAT 2024 | - |
| [Driving with LLMs: Fusing Object-Level Vector Modality for Explainable Autonomous Driving](https://ieeexplore.ieee.org/abstract/document/10611018) | 2024-05 | ICRA 2024 | [GitHub](https://github.com/wayveai/Driving-with-LLMs) |
| [ChatScene: Knowledge-Enabled Safety-Critical Scenario Generation for Autonomous Vehicles](https://arxiv.org/abs/2405.14062) | 2024-05 | ECCV 2024 | - |
| [Editable scene simulation for autonomous driving via collaborative llm-agents](https://ieeexplore.ieee.org/document/10656629) | 2024-06 | CVPR 2024 | [GitHub](https://github.com/yifanlu0227/ChatSim) |
| [Chat2Scenario: Scenario Extraction From Dataset Through Utilization of Large Language Model](https://ieeexplore.ieee.org/document/10588843) | 2024-06 | IV 2024 | [GitHub](https://github.com/ftgTUGraz/Chat2Scenario) |
| [LLMDrive: Closed-Loop End-to-End Driving with Large Language Models](https://arxiv.org/abs/2312.07488) | 2024-07 | CVPR 2024 | [GitHub](https://github.com/opendilab/LMDrive) |
| [SoVAR: Building Generalizable Scenarios from Accident Reports for Autonomous Driving Testing](https://arxiv.org/abs/2409.08081) | 2024-09 | ASE 2024 | - |
| [DiLu: A Knowledge-Driven Approach to Autonomous Driving with Large Language Models](https://arxiv.org/abs/2309.16292) | 2024-09 | ICLR 2024 | [GitHub](https://pjlab-adg.github.io/DiLu_Page/) |
| [LeGEND: A Top-Down Approach to Scenario Generation of Autonomous Driving Systems Assisted by Large Language Models](https://arxiv.org/abs/2409.10066) | 2024-09 | ASE 2024 | [GitHub](https://github.com/MayDGT/LeGEND) |
| [Traffic Scene Generation from Natural Language Description for Autonomous Vehicles with Large Language Model](https://arxiv.org/abs/2409.09575) | 2024-09 | arXiv | [GitHub](https://github.com/basiclab/TTSG) |
| [Promptable Closed-loop Traffic Simulation](https://arxiv.org/abs/2409.05863) | 2024-09 | CoRL 2024 | [GitHub](https://ariostgx.github.io/ProSim/) |
| [Agent Driver: A Conversational LLM Framework for Human-Vehicle Interaction](https://ieeexplore.ieee.org/document/10640507) | 2024-10 | IV 2024 | - |
| [Feasibility and Acceptability of Language Model Agent for Negotiation in Traffic](https://arxiv.org/abs/2410.05008) | 2024-10 | arXiv | - |
| [LLM-Drive: Closed-loop End-to-end Driving with Large Language Models](https://github.com/henrique111222333/LLMDrive) | 2024-11 | - | [GitHub](https://github.com/henrique111222333/LLMDrive) |
| [LLM-Driven Testing for Autonomous Driving Scenarios](https://ieeexplore.ieee.org/document/10852505) | 2024-11 | FLLM 2024 | - |
| [ChatSUMO: Large Language Model for Automating Traffic Scenario Generation in Simulation of Urban MObility](https://ieeexplore.ieee.org/document/10770822) | 2024-11 | IEEE Transactions on Intelligent Vehicles | - |
| [Drive-by-LLM: Leveraging Large Language Models for Safe and Efficient Autonomous Driving](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=YOUR_USER_ID&citation_for_view=YOUR_USER_ID:citation_id) | 2024-12 | - | - |
| [Generating Traffic Scenarios via In-Context Learning to Learn Better Motion Planner](https://arxiv.org/abs/2412.18086) | 2024-12 | AAAI 2025 Oral | [GitHub](https://ezharjan.github.io/AutoSceneGen/) |
| [LLM-attacker: Enhancing Closed-loop Adversarial Scenario Generation for Autonomous Driving with Large Language Models](https://arxiv.org/abs/2501.15850) | 2025-01 | arXiv | - |
| [Risk-Aware Driving Scenario Analysis with Large Language Models](https://arxiv.org/abs/2502.02145) | 2025-02 | arXiv | [GitHub](https://github.com/TUM-AVS/From-Words-to-Collisions) |
| [Text2Scenario: Text-Driven Scenario Generation for Autonomous Driving Test](https://arxiv.org/abs/2503.02911) | 2025-03 | arXiv | [GitHub](https://caixxuan.github.io/Text2Scenario.GitHub.io/) |
| [A Comprehensive LLM-powered Framework for Driving Intelligence Evaluation](https://arxiv.org/abs/2503.05164) | 2025-03 | arXiv | - |
| [Seeking to Collide: Online Safety-Critical Scenario Generation for Autonomous Driving with Retrieval Augmented Large Language Models](https://arxiv.org/abs/2505.00972) | 2025-05 | arXiv | - |

### Scenario Analysis (LLM)

| Paper | Date | Venue | Code | 
|:------|:-----|:------|:-----|
| [Semantic Anomaly Detection with Large Language Models](https://openaccess.thecvf.com/content/ICCV2023W/AICCV/papers/Divekar_Towards_Understanding_Context_and_Sub-Context_for_Semantic_Anomaly_Detection_in_ICCVW_2023_paper.pdf) | 2023-09 | ICCV Workshop 2023 | - |
| [A Closer Look at the Self-Verification Abilities of Large Language Models in Logical Reasoning](https://arxiv.org/abs/2311.07954) | 2023-11 | arXiv | - |
| [DriveGPT4: Interpretable End-to-end Autonomous Driving via Large Language Model](https://ieeexplore.ieee.org/document/10575589) | 2024-01 | IEEE Robotics and Automation Letters 2024 | [GitHub](https://tonyxuqaq.github.io/projects/DriveGPT4/) |
| [Understanding the Driving Scene Description with Large Language Models](https://ieeexplore.ieee.org/document/10646063) | 2024-02 | IV 2024 | - |
| [Drive Like a Human: Rethinking Autonomous Driving with Large Language Models](https://arxiv.org/abs/2402.01697) | 2024-02 | WACV 2024 | [GitHub](https://github.com/PJLab-ADG/DriveLikeAHuman) |
| [DrivingGaussian: Composite Gaussian Splatting for Surrounding Dynamic Autonomous Driving Scenes](https://arxiv.org/abs/2312.07920) | 2024-03 | CVPR 2024 | [GitHub](https://github.com/VDIGPKU/DrivingGaussian) |
| [DrivingDojo: Democratizing Autonomous Driving Simulation at Scale](https://arxiv.org/abs/2409.13349) | 2024-09 | NeurIPS 2024 | [GitHub](https://driving-dojo.github.io/) |
| [Exploring Risk-Aware Driving Scenarios with LLM for Autonomous Vehicle Testing](https://ieeexplore.ieee.org/document/10588829) | 2024-09 | IV 2024 | - |
| [SafeDiffuser: Safe Planning with Diffusion Probabilistic Models](https://openreview.net/forum?id=jw8P8rZaDZ) | 2024-10 | ICLR 2024 | [GitHub](https://github.com/pjlab-adg/SafeDiffuser) |
| [MotionLLM: Multimodal Motion-Language Reasoning](https://arxiv.org/abs/2405.20013) | 2024-10 | NeurIPS 2024 | [GitHub](https://github.com/zgzxy001/MotionLLM) |
| [TUMTraffic-VideoQA: A Benchmark for Unified Spatio-Temporal Video Understanding in Traffic Scenes](https://arxiv.org/abs/2501.01846) | 2025-01 | ICML 2025 | [GitHub](https://github.com/Lumos-Bot/TUMTrafic-VideoQA) |
| [ULS-AV: Safety Assessment of Ultra Large-Scale Autonomous Vehicle System with Vision Language Model](https://ieeexplore.ieee.org/document/10742698) | 2025-01 | arXiv | [GitHub](https://github.com/LingKang28/ULS-AV) |
| [Language as Context: Decoding Road Scene Language with Multi-Modal Foundation Models for Autonomous Driving](https://ieeexplore.ieee.org/document/10845653) | 2025-01 | CORL 2024 | - |
| [LangSurf: Language-Embedded Surface Gaussians for 3D Scene Understanding](https://arxiv.org/abs/2411.10740) | 2024-12 | NeurIPS 2024 | [GitHub](https://langsurf.github.io/)

## 🌟 Multimodal Large Language Models for Autonomous Driving

### Scenario Generation (MLLM)

| Paper | Date | Venue | Code |
|:------|:-----|:------|:-----|
| [VehicleGPT: A Mobile Large Language Model Platform for Intelligent Vehicles](https://ieeexplore.ieee.org/document/10640503) | 2023-08 | IV 2024 | - |
| [GPT-Driver: Learning to Drive with GPT](https://arxiv.org/abs/2310.01415) | 2023-10 | arXiv | - |
| [LMDrive: Closed-Loop End-to-End Driving with Large Language Models](https://arxiv.org/abs/2312.07488) | 2023-12 | CVPR 2024 | [GitHub](https://github.com/opendilab/LMDrive) |
| [DriveMLM: Aligning Multi-Modal Large Language Models with Behavioral Planning States for Autonomous Driving](https://arxiv.org/abs/2312.09245) | 2023-12 | arXiv | [GitHub](https://github.com/OpenGVLab/DriveMLM) |
| [On the Road with GPT-4V(ision): Early Explorations of Visual-Language Model on Autonomous Driving](https://arxiv.org/abs/2311.05332) | 2023-12 | CVPR Workshop 2024 | [GitHub](https://github.com/PJLab-ADG/GPT4V-AD-Exploration) |
| [DriveLM: Driving with Graph Visual Question Answering](https://arxiv.org/abs/2312.14150) | 2023-12 | ECCV 2024 | [GitHub](https://github.com/OpenDriveLab/DriveLM) |
| [DriveVLM: The Convergence of Autonomous Driving and Large Vision-Language Models](https://arxiv.org/abs/2402.12289) | 2024-02 | arXiv | - |
| [Can We Use Large Language Models for Real-Time Safety-Critical Decision Making in Autonomous Driving?](https://arxiv.org/abs/2402.02848) | 2024-02 | arXiv | - |
| [SurroundScript: Generating Autonomous Driving World Models from Natural Language](https://ieeexplore.ieee.org/document/10606667) | 2024-04 | ICRA 2024 | - |
| [DMT: Dynamic Mutual Training for Semi-Supervised Learning](https://openaccess.thecvf.com/content/CVPR2024/papers/Feng_DMT_Dynamic_Mutual_Training_for_Semi-Supervised_Learning_CVPR_2024_paper.pdf) | 2024-04 | CVPR 2024 | [GitHub](https://github.com/JackWangPair/DMT) |
| [DriveCoT: Agent Framework for Autonomous Driving](https://arxiv.org/abs/2404.13333) | 2024-04 | arXiv | - |
| [Generating CAVs Cooperative Merging Strategies for Mixed Traffic Environment Using LLMs](https://ieeexplore.ieee.org/document/10575629) | 2024-05 | IV 2024 | - |
| [DriveAdapter: Breaking the Coupling Barrier of Perception and Planning in End-to-End Autonomous Driving](https://arxiv.org/abs/2308.00398) | 2024-05 | ICLR 2024 | [GitHub](https://github.com/OpenDriveLab/DriveAdapter) |
| [GPT-4o as an End-to-End Autonomous Driving System](https://arxiv.org/abs/2405.01692) | 2024-05 | arXiv | - |
| [BEVWorld: A Multimodal World Model for Autonomous Driving via Unified BEV Latent Space](https://arxiv.org/abs/2407.05679) | 2024-07 | arXiv | [GitHub](https://github.com/zhangyp15/BEVWorld) |
| [RoadBEV: Road Surface Reconstruction in Bird's Eye View](https://arxiv.org/abs/2404.06605) | 2024-07 | ECCV 2024 | [GitHub](https://github.com/ztsrxh/RoadBEV) |
| [VLMAgent: A Multimodal Large Language Model Framework for Autonomous Driving Scenarios](https://ieeexplore.ieee.org/document/10681593) | 2024-08 | arXiv | - |
| [Real-Time Traffic Safety Assessment Using Foundation Models](https://ieeexplore.ieee.org/document/10588877) | 2024-08 | IV 2024 | - |
| [PlanAgent: A Multi-modal Large Language Agent for Closed-loop Vehicle Motion Planning](https://arxiv.org/abs/2406.01587) | 2024-08 | arXiv | [GitHub](https://github.com/PlanAgent/PlanAgent) |
| [Language-Guided Hybrid Motion Planning](https://ieeexplore.ieee.org/document/10588847) | 2024-08 | IV 2024 | - |
| [VideoLLaMB: Long Video Understanding with Recurrent Memory Bridges](https://arxiv.org/abs/2409.01071) | 2024-09 | arXiv | [GitHub](https://videollamb.github.io/) |
| [DataSP: Adapting Vision Foundation Models for Automated Measurement in Automotive Service](https://ieeexplore.ieee.org/document/10640543) | 2024-09 | IV 2024 | - |
| [Cognitive Kernel: An Open-source Agent System towards Generalist Autopilots](https://arxiv.org/abs/2409.10277) | 2024-09 | arXiv | [GitHub](https://github.com/tsinghua-fib-lab/cognitive-kernel) |
| [OmniDrive: A Holistic LLM-Agent Framework for Autonomous Driving with 3D Perception, Reasoning and Planning](https://arxiv.org/abs/2405.01533) | 2024-10 | arXiv | [GitHub](https://github.com/NVlabs/OmniDrive) |
| [FusionLLM: A Decentralized LLM Training System on Geo-distributed GPUs with Adaptive Compression](https://arxiv.org/abs/2410.12707) | 2024-10 | arXiv | [GitHub](https://github.com/OpenLMLab/FusionLLM) |
| [GPT-4V for Video-based Urban Safety Research](https://arxiv.org/abs/2410.17744) | 2024-10 | arXiv | - |
| [OpenLKA: An Open Dataset of Lane Keeping Assist from Recent Car Models under Real-world Driving Conditions](https://arxiv.org/abs/2505.07230) | 2025-05 | arXiv | [GitHub](https://github.com/alxkm/OpenLKA)

### Scenario Analysis (MLLM)

| Paper | Date | Venue | Code |
|:------|:-----|:------|:-----|
| [Automated Evaluation of Large Vision-Language Models on Self-driving Corner Cases](https://arxiv.org/abs/2404.10595) | 2024-04 | WACV2025 | [GitHub](https://github.com/Dongmin-Cho/Self-Driving-Corner-Cases) |
| [A Vision-Language Foundation Model for Autonomous Driving](https://arxiv.org/abs/2405.17968) | 2024-05 | arXiv | - |
| [MotionLLM: Multimodal Motion-Language Reasoning](https://arxiv.org/abs/2405.17013) | 2024-05 | NeurIPS 2024 | [GitHub](https://github.com/IDEA-Research/MotionLLM) |
| [Multi-modal Large Language Model Driven Scenario Testing for Autonomous Vehicles](https://arxiv.org/abs/2408.01699) | 2024-08 | arXiv | - |
| [Cognitive Kernel: An Open-source Agent System towards Generalist Autopilots](https://arxiv.org/abs/2409.10277) | 2024-09 | arXiv | [GitHub](https://github.com/tsinghua-fib-lab/cognitive-kernel) |
| [LangSurf: Language-Embedded Surface Gaussians for 3D Scene Understanding](https://arxiv.org/abs/2411.17635) | 2024-11 | NeurIPS 2024 | [GitHub](https://github.com/LangSurf/LangSurf) |

## 🌟 Vision-Language Models for Autonomous Driving

### Scenario Generation (VLM)

| Paper | Date | Venue | Code | 
|:------|:-----|:------|:-----|
| [FuseDream: Training-Free Text-to-Image Generation with Improved CLIP+GAN Space Optimization](https://arxiv.org/abs/2112.01573) | 2021-12 | ICLR 2022 | [GitHub](https://github.com/gnobitab/FuseDream) |
| [Learning to Prompt for Open-Vocabulary Object Detection with Vision-Language Model](https://arxiv.org/abs/2203.14940) | 2022-06 | CVPR 2022 | [GitHub](https://github.com/dyabel/detpro) |
| [Masked Autoencoders for Self-Supervised Learning on Automotive Point Clouds](https://arxiv.org/abs/2207.00531) | 2022-07 | arXiv | [GitHub](https://github.com/GeoX-Lab/G-MAE) |
| [Flamingo: a Visual Language Model for Few-Shot Learning](https://arxiv.org/abs/2204.14198) | 2022-10 | NeurIPS 2022 | - |
| [Visual Instruction Tuning](https://arxiv.org/abs/2304.08485) | 2023-04 | NeurIPS 2023 | [GitHub](https://github.com/haotian-liu/LLaVA) |
| [Vision Foundation Model Embedding-Based Semantic Anomaly Detection](https://arxiv.org/abs/2505.05431) | 2025-05 | ICRA 2025 Workshop | [GitHub](https://github.com/rasmuszero/vfm-anomaly) |
| [Self-Evaluation Guided Beam Search for Reasoning](https://arxiv.org/abs/2305.00633) | 2023-05 | NeurIPS 2023 | [GitHub](https://github.com/YuxiXie/SelfEval-Guided-Decoding) |
| [NuInstruct: A Multi-Modal Dataset for Teaching Object Localization in Autonomous Driving](https://arxiv.org/abs/2401.17564) | 2024-01 | CVPR 2024 | [GitHub](https://github.com/CarVision/nu-instruct) |
| [DriveLM: Driving with Graph Visual Question Answering](https://arxiv.org/abs/2312.14150) | 2024-03 | ECCV 2024 | [GitHub](https://github.com/OpenDriveLab/DriveLM) |
| [RAG-Driver: Generalisable Driving Explanations with Retrieval-Augmented In-Context Learning in Multi-Modal Large Language Model](https://arxiv.org/abs/2402.10828) | 2024-04 | arXiv | - |
| [DrivingCoT: Chain-of-Thought Reasoning for Autonomous Driving with Large Language Models](https://arxiv.org/abs/2403.16996) | 2024-04 | arXiv | - |

### Scenario Analysis (VLM)

| Paper | Date | Venue | Code |
|:------|:-----|:------|:-----|
| [CLIP-AD: A Language-Guided Staged Dual-Path Model for Zero-shot Anomaly Detection](https://arxiv.org/abs/2311.00626) | 2023-11 | arXiv | - |
| [TransDETR: Temporally Correlated Transformers for Multi-frame Detection in Videos](https://arxiv.org/abs/2401.02921) | 2024-01 | arXiv | - |
| [Vision Language Models Can Parse Floor Plan Maps](https://arxiv.org/abs/2407.12482) | 2024-07 | IROS 2024 | - |
| [Multi-modal Large Language Model Driven Scenario Testing for Autonomous Vehicles](https://arxiv.org/abs/2408.01699) | 2024-08 | arXiv | - |
| [VisionZip: Longer is Better But Not Necessary in Vision Language Models](https://arxiv.org/abs/2404.06865) | 2024-09 | EMNLP 2024 | [GitHub](https://github.com/dvlab-research/VisionZip) |
| [LangSurf: Language-Embedded Surface Gaussians for 3D Scene Understanding](https://arxiv.org/abs/2411.17635) | 2024-11 | NeurIPS 2024 | [GitHub](https://github.com/LangSurf/LangSurf) |

## 🌟 World Models for Autonomous Driving

### Scenario Generation (WM)

| Paper | Date | Venue | Code |
|:------|:-----|:------|:-----|
| [Learning Interactive Driving Policies via Data-driven Simulation](https://arxiv.org/abs/2204.10755) | 2022-04 | ICRA 2022 | [GitHub](https://github.com/autonomousvision/tuplan_garage) |
| [Learning from All Vehicles](https://arxiv.org/abs/2203.11934) | 2022-06 | CVPR 2022 | [GitHub](https://github.com/dotchen/LAV) |
| [ST-P3: End-to-end Vision-based Autonomous Driving via Spatial-Temporal Feature Learning](https://arxiv.org/abs/2207.07601) | 2022-07 | ECCV 2022 | [GitHub](https://github.com/OpenPerceptionX/ST-P3) |
| [MUVO: A Multimodal World Model for Autonomous Driving with Geometric Representations](https://arxiv.org/abs/2311.11762) | 2023-11 | ICRA 2024 | [GitHub](https://github.com/robot-learning-freiburg/MUVO) |
| [ADriver-I: A General World Model for Autonomous Driving](https://arxiv.org/abs/2311.13549) | 2023-11 | arXiv | - |
| [GenAD: Generalized Predictive Model for Autonomous Driving](https://arxiv.org/abs/2403.09630) | 2024-03 | CVPR 2024 | [GitHub](https://github.com/OpenDriveLab/GenAD) |
| [DriveDreamer: Towards Real-world-driven World Models for Autonomous Driving](https://arxiv.org/abs/2309.09777) | 2024-05 | ECCV 2024 | [GitHub](https://github.com/JeffWang987/DriveDreamer) |
| [OccWorld: Learning a 3D Occupancy World Model for Autonomous Driving](https://arxiv.org/abs/2311.16038) | 2024-05 | ECCV 2024 | [GitHub](https://github.com/wzzheng/OccWorld) |
| [WoVoGen: World Volume-aware Diffusion for Controllable Multi-camera Driving Scene Generation](https://arxiv.org/abs/2312.02934) | 2024-07 | ECCV 2024 | [GitHub](https://github.com/fudan-zvg/WoVoGen) |
| [OccSora: 4D Occupancy Generation Models as World Simulators for Autonomous Driving](https://arxiv.org/abs/2405.20337) | 2024-05 | arXiv | [GitHub](https://github.com/wzzheng/OccSora) |
| [Driving in the Occupancy World: Vision-Centric 4D Occupancy Forecasting and Planning via World Models for Autonomous Driving](https://arxiv.org/abs/2408.14197) | 2024-08 | AAAI2025 | [GitHub](https://github.com/wzzheng/OccWorld) |
| [DriveDreamer-2: LLM-Enhanced World Models for Diverse Driving Video Generation](https://arxiv.org/abs/2403.06845) | 2024-09 | arXiv | [GitHub](https://github.com/f-gloria/DriveDreamer2) |
| [Vista: A Generalizable Driving World Model with High Fidelity and Versatile Controllability](https://arxiv.org/abs/2405.17398) | 2024-10 | NeurIPS 2024 | [GitHub](https://github.com/OpenDriveLab/Vista) |
| [DrivingWorld: 4D World Model for Autonomous Driving](https://arxiv.org/abs/2410.07681) | 2024-10 | arXiv | - |
| [Genie: Generative Interactive Environments](https://arxiv.org/abs/2402.15391) | 2024-10 | ICML 2024 | - |
| [MILE: A Multi-Level Framework for Scalable Graph Representation Learning](https://arxiv.org/abs/2404.18908) | 2024-10 | arXiv | [GitHub](https://github.com/Graph-COM/MILE) |
| [DRIVEGPT: Scaling Autoregressive Multi-Modal Foundation Models for Autonomous Driving](https://arxiv.org/abs/2411.17327) | 2024-11 | arXiv | [GitHub](https://github.com/OpenGVLab/DRIVEGPT) |
| [Gensim: A General Social Simulation Platform with Large Language Model based Agents](https://arxiv.org/abs/2410.04360) | 2024-11 | arXiv | [GitHub](https://github.com/TsinghuaC3I/Gensim) |
| [DriveDreamer4D: World Models with Efficient 4D Scene Representation for Autonomous Driving](https://arxiv.org/abs/2409.17156) | 2024-12 | arXiv | [GitHub](https://github.com/GigaAI-research/DriveDreamer4D) |

### Scenario Analysis (WM)

| Paper | Date | Venue | Code |
|:------|:-----|:------|:-----|
| [Learning Interactive Driving Policies via Data-driven Simulation](https://arxiv.org/abs/2204.10755) | 2022-04 | ICRA 2022 | [GitHub](https://github.com/autonomousvision/tuplan_garage) |
| [Learning from All Vehicles](https://arxiv.org/abs/2203.11934) | 2022-06 | CVPR 2022 | [GitHub](https://github.com/dotchen/LAV) |
| [ST-P3: End-to-end Vision-based Autonomous Driving via Spatial-Temporal Feature Learning](https://arxiv.org/abs/2207.07601) | 2022-07 | ECCV 2022 | [GitHub](https://github.com/OpenPerceptionX/ST-P3) |
| [Drive as You Speak: Enabling Human-Like Interaction with Large Language Models in Autonomous Vehicles](https://arxiv.org/abs/2309.10228) | 2023-09 | WACV 2024 | - |
| [Generative Modeling of Autonomous Driving Trajectories](https://arxiv.org/abs/2309.09866) | 2023-09 | ICLR 2024 | [GitHub](https://github.com/autonomousvision/carla_garage) |
| [CoWorld: Decoupled Generation for Autonomous Driving using Pre-trained Foundation Models](https://arxiv.org/abs/2309.17080) | 2023-09 | arXiv | - |
| [DrivingDojo: Synthesizing Realistic and Diverse Driving Scenarios for Safety-Critical Learning](https://arxiv.org/abs/2402.16329) | 2024-02 | ICRA 2024 | [GitHub](https://github.com/autonomousvision/driving_dojo) |
| [DIVA: A Disentangled World Model for Autonomous Driving](https://arxiv.org/abs/2302.12635) | 2024-05 | ICCV 2023 | - |
| [CAT: Closed-loop Adversarial Training for Safe End-to-End Driving](https://arxiv.org/abs/2310.12432) | 2024-05 | RSS 2024 | [GitHub](https://github.com/georgeliu233/CAT) |
| [WorldModel: A Foundation Model for Autonomous Driving](https://arxiv.org/abs/2403.04878) | 2024-08 | arXiv | - |
| [OccSora: 4D Occupancy Generation Models as World Simulators for Autonomous Driving](https://arxiv.org/abs/2405.20337) | 2024-09 | arXiv | [GitHub](https://github.com/wzzheng/OccSora) |

## 🌟 Datasets Comparison

| Dataset | Year | Img | View | Real | Lidar | Radar | Traj | 3D | 2D | Lane | Weather | Time | Region | Company |
|:--------|:-----|:----|:-----|:-----|:------|:------|:-----|:---|:---|:-----|:--------|:-----|:-------|:--------|
| CamVid | 2009 | RGB | FPV | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ | D | U | - |
| KITTI | 2013 | RGB/S | FPV | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | D | U/R/H | - |
| Cyclists | 2016 | RGB | FPV | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | D | U | - |
| Cityscapes | 2016 | RGB/S | FPV | ✅ | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ | ❌ | D | U | - |
| SYNTHIA | 2016 | RGB | FPV | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | D/N | U | - |
| Campus | 2016 | RGB | BEV | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | D | C | - |
| RobotCar | 2016 | RGB | FPV | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | D/N | U | - |
| Mapillary | 2017 | RGB | FPV | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ | D/N | U | - |
| P.F.B. | 2017 | RGB | FPV | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ | D/N | U | - |
| BDD100K | 2018 | RGB | FPV | ✅ | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | D | U/H | - |
| HighD | 2018 | RGB | BEV | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ | D | H | - |
| Udacity | 2018 | RGB | FPV | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | D | U | - |
| KAIST | 2018 | RGB/S | FPV | ✅ | ✅ | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ | D/N | U | - |
| Argoverse | 2019 | RGB/S | FPV | ✅ | ✅ | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ | D/N | U | - |
| TRAF | 2019 | RGB | FPV | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ | D | U | - |
| ApolloScape | 2019 | RGB/S | FPV | ✅ | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | D | U | - |
| ACFR | 2019 | RGB | BEV | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | D | RA | - |
| H3D | 2019 | RGB | FPV | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ | D | U | - |
| INTERACTION | 2019 | RGB | BEV | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | D | I/RA | - |
| Comma2k19 | 2019 | RGB | FPV | ✅ | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ | D/N | U/S/R/H | - |
| InD | 2020 | RGB | BEV | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | D | I | - |
| RounD | 2020 | RGB | BEV | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | D | RA | - |
| nuScenes | 2020 | RGB | FPV | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ | D/N | U | - |
| Lyft Level 5 | 2020 | RGB | FPV | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ | D/N | U/S | - |
| Waymo Open | 2020 | RGB | FPV | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | D/N | U | - |
| A*3D | 2020 | RGB | FPV | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | D/N | U | - |
| RobotCar Radar | 2020 | RGB | FPV | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | D/N | U | - |
| Toronto3D | 2020 | RGB | BEV | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ | ✅ | ❌ | D/N | U | University of Waterloo |
| A2D2 | 2020 | RGB | FPV | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | D | U/H/S/R | Audi |
| WADS | 2020 | RGB | FPV | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ | D/N | U/S/R | Michigan Technological University |
| Argoverse 2 | 2021 | RGB/S | FPV | ✅ | ✅ | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | D/N | U | - |
| PandaSet | 2021 | RGB | FPV | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | D/N | U | - |
| ONCE | 2021 | RGB | FPV | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | D/N | U | - |
| Leddar PixSet | 2021 | RGB | FPV | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ | ✅ | D/N | U/S/R | Leddar |
| ZOD | 2022 | RGB | FPV | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | D/N | U/R/S/H | Zenseact |
| IDD-3D | 2022 | RGB | FPV | ✅ | ✅ | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | - | R | INAI |
| CODA | 2022 | RGB | FPV | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | D/N | U/S/R | Huawei |
| SHIFT | 2022 | RGB | FPV | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | D/N | U/S/R/H | ETH Zürich |
| DeepAccident | 2023 | RGB/S | FPV/BEV | ❌ | ✅ | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | D/N | U/S/R/H | HKU, Huawei, CARLA |
| Dual_Radar | 2023 | RGB | FPV | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | D/N | U | Tsinghua University |
| V2V4Real | 2023 | RGB | FPV | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ | ✅ | ❌ | - | U/H/S | UCLA Mobility Lab |
| SCaRL | 2024 | RGB/S | FPV/BEV | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | D/N | U/S/R/H | Fraunhofer CARLA |
| MARS | 2024 | RGB | FPV | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | D/N | U/S/H | NYU, MAY Mobility |
| Scenes101 | 2024 | RGB | FPV | ✅ | ❌ | ❌ | ✅ | ❌ | ❌ | ✅ | ✅ | D/N | U/S/R/H | Wayve |
| TruckScenes | 2025 | RGB | FPV | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | D/N | H/U | MAN |

*Notes: View: FPV=First-Person, BEV=Bird's-Eye; Time: D=Day, N=Night; Region: U=Urban, R=Rural, H=Highway, S=Suburban, C=Campus, I=Intersection, RA=Road Area; Img: RGB/S=RGB+Stereo*


## 🌟 Simulators

| Simulator | Year | Back-end | Open Source | Realistic Perception | Custom Scenario | Real World Map | Human Design Map | Python API | C++ API | ROS API | Company |
|:----------|:-----|:---------|:------------|:---------------------|:----------------|:---------------|:-----------------|:-----------|:--------|:--------|:--------|
| TORCS | 2000 | None | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | - |
| Webots | 2004 | ODE | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ | - |
| CarRacing | 2017 | None | ✅ | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | - |
| CARLA | 2017 | UE4 | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ | - |
| SimMobilityST | 2017 | None | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | - |
| GTA-V | 2017 | RAGE | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | - |
| highway-env | 2018 | None | ✅ | ❌ | ✅ | ❌ | ✅ | ✅ | ❌ | ❌ | - |
| Deepdrive | 2018 | UE4 | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ | - |
| esmini | 2018 | Unity | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | - |
| AutonoViSim | 2018 | PhysX | ❌ | ✅ | ✅ | ❌ | ❌ | ✅ | ❌ | ❌ | - |
| AirSim | 2018 | UE4 | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ | - |
| SUMO | 2018 | None | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ | ✅ | ❌ | - |
| Apollo | 2018 | Unity | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | - |
| Sim4CV | 2018 | UE4 | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ | ❌ | - |
| MATLAB | 2018 | MATLAB | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Mathworks |
| Scenic | 2019 | None | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | Toyota Research Institute, UC Berkeley |
| SUMMIT | 2020 | UE4 | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ | - |
| MultiCarRacing | 2020 | None | ✅ | ❌ | ✅ | ❌ | ✅ | ✅ | ❌ | ❌ | - |
| SMARTS | 2020 | None | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | - |
| LGSVL | 2020 | Unity | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | - |
| CausalCity | 2020 | UE4 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | - |
| Vista | 2020 | None | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ❌ | ❌ | MIT |
| MetaDrive | 2021 | Panda3D | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | - |
| L2R | 2021 | UE4 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | - |
| AutoDRIVE | 2021 | Unity | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | - |
| Nuplan | 2021 | None | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | Motional |
| AWSIM | 2021 | Unity | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ | Autoware |
| InterSim | 2022 | None | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ❌ | ❌ | Tsinghua |
| Nocturne | 2022 | None | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | Facebook |
| BeamNG.tech | 2022 | Soft-body physics | ❌ | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ | ✅ | BeamNG GmbH |
| Waymax | 2023 | JAX | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ | ❌ | Waymo |
| UNISim | 2023 | None | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ | ❌ | Waabi |
| TBSim | 2023 | None | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | NVIDIA |
| Nvidia DriveWorks | 2024 | Nvidia GPU | ❌ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ | NVIDIA |

## Contributing
We welcome contributions from the community! If you have research papers, tools, or resources to add, please create a pull request or open an issue.

## License
This repository is released under the [Apache 2.0 license](https://github.com/TUM-AVS/FM-AV-Survey-Scenario-Generation-Analysis/blob/main/LICENSE). 

``` 