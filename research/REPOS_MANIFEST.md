# repos/ manifest

Trimmed 2026-10-02 to free disk (root disk was 100% full). 22 folders kept; the rest deleted.
Every entry records its exact origin and commit, so any deleted repo can be restored exactly as the research notes cited it (`path:line` references).

## Kept

| MB before trim | Folder | Why kept | Origin / restore |
|---|---|---|---|
| 30 | `agent-as-policy` | blueprint: AGP YAM bridge, settle/calibration | `git clone https://github.com/agent-as-policy-2026/agent-as-policy agent-as-policy && git -C agent-as-policy checkout c6875d0457358dd54dd23f61355d21149d63ca64` |
| 122 | `cap-x` | blueprint: CaP-Gym, SAM3/grasp/IK APIs | `git clone https://github.com/capgym/cap-x cap-x && git -C cap-x checkout 53e9966d7a8e2fa7494676772bccc35280f5c0ed` |
| 36 | `dimensionalOS_dimos` | blueprint: agentic robot runtime | `git clone https://github.com/dimensionalOS/dimos dimensionalOS_dimos && git -C dimensionalOS_dimos checkout cdfaef03eb3e21f74b9fd61b46bd12d24fd06252` |
| 107 | `EmbodiedSWE` | your source | `git clone https://github.com/EmbodiedSWE/EmbodiedSWE EmbodiedSWE && git -C EmbodiedSWE checkout 433d0ceedc2c5acf58551f4b3502b2cd1c7e8f1b` |
| 5 | `google-deepmind_gemini-robotics-sdk` | blueprint: ER agent + success detector | `git clone https://github.com/google-deepmind/gemini-robotics-sdk google-deepmind_gemini-robotics-sdk && git -C google-deepmind_gemini-robotics-sdk checkout 41cbd35caa681b0c1ca336d93a095f632829cd50` |
| 168 | `GPT-as-Policy` | your source | `git clone https://github.com/anonymous-report-421/GPT-as-Policy GPT-as-Policy && git -C GPT-as-Policy checkout 8f3d362b077d8efb77e2a7274d5b2c20e2243846` |
| 261 | `GPT-Policy-cheng` | your source | `git clone https://github.com/cheng-haha/GPT-Policy GPT-Policy-cheng && git -C GPT-Policy-cheng checkout ab970d88bc5570d80a7b4f3e8d3ad97ebe65a007` |
| 28 | `huggingface_lerobot` | action-head training | `git clone https://github.com/huggingface/lerobot huggingface_lerobot && git -C huggingface_lerobot checkout e0d50211ef236143ae867228662b7dfaba554f02` |
| 32 | `innate-os-pr817` | your source (innate-os at PR #817 head) | `git clone https://github.com/innate-inc/innate-os.git innate-os-pr817 && git -C innate-os-pr817 checkout ef8fd0df33767c1343e0138e4eed391753d12662` |
| 12 | `inspect-robots` | eval harness (Robocurve) | `git clone https://github.com/robocurve/inspect-robots.git inspect-robots && git -C inspect-robots checkout 095172fb8452ac1d47da49834df4d3faa562e4ee` |
| 4 | `inspect-robots-yam` | eval harness YAM driver (Robocurve) | `git clone https://github.com/robocurve/inspect-robots-yam.git inspect-robots-yam && git -C inspect-robots-yam checkout 178c93037afcf593618e0a1db6ae1f1b0cfeaa0f` |
| 481 | `llm-robotics-playground` | your source (agent .venv removed) | `git clone https://github.com/dimentary/llm-robotics-playground llm-robotics-playground && git -C llm-robotics-playground checkout 5ffdf5803636efa99f1184bb8481ce7cb85908b6` |
| 5 | `metal-arm-harness` | your source | `git clone https://github.com/makermods-robotics/metal-arm-harness.git metal-arm-harness && git -C metal-arm-harness checkout 1f3c6df34d5ad5ad13a20bc0cd3cdea42498c424` |
| 27 | `NVIDIA_Isaac-GR00T` | action-head / VLA | `git clone https://github.com/NVIDIA/Isaac-GR00T NVIDIA_Isaac-GR00T && git -C NVIDIA_Isaac-GR00T checkout 51d4c89f72fda44cbf77285c6a8114b52676b8a1` |
| 304 | `Pigey` | blueprint: Opus 4.7 + pi0.5 + TAMP | `git clone https://github.com/lianegalanti/Pigey Pigey && git -C Pigey checkout 002cbda2797500b5cc18625f29dab206fc549922` |
| 45 | `piper-astra-jev` | your source | `git clone https://github.com/RobotKitAI/piper-astra-jev piper-astra-jev && git -C piper-astra-jev checkout 10d671e01d467475084033e7af9f6595886ab7b3` |
| 47 | `quackd` | your source | `git clone https://github.com/rokbenko/quackd.git quackd && git -C quackd checkout 25635c6869c4740f3ff07e2640077d398a24c62c` |
| 6 | `RLinf_RPent` | blueprint: Claude Agent SDK planner + frozen pi0.5 | `git clone https://github.com/RLinf/RPent RLinf_RPent && git -C RLinf_RPent checkout 72686b8c7606853afce7786bee4a3d95498a1c0d` |
| 3 | `RoboDojo` | benchmark | `git clone https://github.com/RoboDojo-Benchmark/RoboDojo RoboDojo && git -C RoboDojo checkout 726e9aabfaa642203722eb126f5eaf0f37f3e1ad` |
| 59 | `showlab_Show-Harness` | blueprint: discrete action units + small-model FT | `git clone https://github.com/showlab/Show-Harness showlab_Show-Harness && git -C showlab_Show-Harness checkout 403f0685fdf4c0c8eb67c6d9c3137da96a58fdfe` |
| 339 | `so101-painting` | your source (HF bucket download; agent .venv removed) | HF bucket: https://huggingface.co/buckets/mishig/so101-painting |
| 86 | `strands-labs_robots` | blueprint: agent tools over LeRobot/GR00T | `git clone https://github.com/strands-labs/robots strands-labs_robots && git -C strands-labs_robots checkout 3d2c59a89a194d942e5907b3def5f1386d236255` |

## Deleted (62)

Restore with the command shown (run inside `research/repos/`).

| MB | Folder | Restore |
|---|---|---|
| 2 | `aditya-ramabadran_drivingbench_harness_v1` | `git clone https://github.com/aditya-ramabadran/drivingbench_harness_v1 aditya-ramabadran_drivingbench_harness_v1 && git -C aditya-ramabadran_drivingbench_harness_v1 checkout 0efc0a64e84ae02640d124d226d6adee2f05595d` |
| 42 | `agenticros_agenticros` | `git clone https://github.com/agenticros/agenticros agenticros_agenticros && git -C agenticros_agenticros checkout 18a7913650d74d130c32a6ee246a7fe7f16cabe6` |
| 15 | `air-embodied-brain_Zetta-Embodiment` | `git clone https://github.com/air-embodied-brain/Zetta-Embodiment air-embodied-brain_Zetta-Embodiment && git -C air-embodied-brain_Zetta-Embodiment checkout 1fee179644d52c32fa5a7728751cf0853a29b9c0` |
| 212 | `amap-cvlab_ABot-Claw` | `git clone https://github.com/amap-cvlab/ABot-Claw amap-cvlab_ABot-Claw && git -C amap-cvlab_ABot-Claw checkout 39c9ffc94a84f2e6baac431a9bff9771111f8539` |
| 2 | `astra-paints` | `git clone https://github.com/riyer8/astra-paints.git astra-paints && git -C astra-paints checkout e0a0f591d4f3e0c6998d49dff8cd195e4366183d` |
| 243 | `astra-robodojo-rollouts` | HF dataset: `git clone https://huggingface.co/datasets/YuMoool/astra-robodojo-rollouts` |
| 23 | `automatika-robotics_embodied-agents` | `git clone https://github.com/automatika-robotics/embodied-agents automatika-robotics_embodied-agents && git -C automatika-robotics_embodied-agents checkout 29898459a7fc80ed1a5f6cbbfdc0f9459b68321f` |
| 11 | `Awesome-Astra-Embodied-AI` | `git clone https://github.com/zjwzcx/Awesome-Astra-Embodied-AI Awesome-Astra-Embodied-AI && git -C Awesome-Astra-Embodied-AI checkout ba9fd3b305dcd53cdfcb3960207555849bfbb5a8` |
| 67 | `awesome-jev` | `git clone https://github.com/Frank-ZY-Dou/awesome-jev awesome-jev && git -C awesome-jev checkout 0f2a8aa1eb984195d3f34890e2d471e45539826d` |
| 1 | `Awesome-Robot-Use-Agent` | `git clone https://github.com/visitworld123/Awesome-Robot-Use-Agent Awesome-Robot-Use-Agent && git -C Awesome-Robot-Use-Agent checkout 14d08d55151a4d57971b0eebdd16e6036872164e` |
| 919 | `clapboardbench` | `git clone https://github.com/robocurve/clapboardbench.git clapboardbench && git -C clapboardbench checkout 2f68726a8347dad847a4c6532fe7e842d1d95d5c` |
| 16 | `CodeActionBench` | `git clone https://github.com/lyhkk/CodeActionBench CodeActionBench && git -C CodeActionBench checkout f5cf933e71676bb6e3c2d52d31800523569aa16c` |
| 350 | `dexagent1.github.io` | `git clone https://github.com/dexagent1/dexagent1.github.io.git dexagent1.github.io && git -C dexagent1.github.io checkout d6ec746a12cb4c28d92dff3e64cfe085b5042ed0` |
| 212 | `DexAgent123.github.io` | `git clone https://github.com/DexAgent123/DexAgent123.github.io.git DexAgent123.github.io && git -C DexAgent123.github.io checkout e07adf711b04673ab418a54e13f525371d4c2e63` |
| 4 | `EIT-HAI_Thea` | `git clone https://github.com/EIT-HAI/Thea EIT-HAI_Thea && git -C EIT-HAI_Thea checkout b9c47830155194e53be801eefe8aea6a26a97366` |
| 167 | `EMERGE-Policy_EMERGE-Policy` | `git clone https://github.com/EMERGE-Policy/EMERGE-Policy EMERGE-Policy_EMERGE-Policy && git -C EMERGE-Policy_EMERGE-Policy checkout e4c7134ad2aeb6dded049106ada1c28985b66e69` |
| 180 | `FBddcz_embodied-jev` | `git clone https://github.com/FBddcz/embodied-jev FBddcz_embodied-jev && git -C FBddcz_embodied-jev checkout f08de2e4e20d6cd69fea9c57ac1062c3ef510f1e` |
| 29 | `FlagOpen_RoboOS` | `git clone https://github.com/FlagOpen/RoboOS FlagOpen_RoboOS && git -C FlagOpen_RoboOS checkout 35d6fe76a61d267f6d4728f3af67e8794382ba26` |
| 3 | `genrobo-GRID-playground` | `git clone https://github.com/GenRobo/GRID-playground genrobo-GRID-playground && git -C genrobo-GRID-playground checkout d6bff42ed9899744659e2860b74e8ee7b50045a5` |
| 24 | `genrobo-isaac-sim-mcp` | `git clone https://github.com/GenRobo/isaac-sim-mcp genrobo-isaac-sim-mcp && git -C genrobo-isaac-sim-mcp checkout 0bb9c0712080a0fc02d97338e4568d10d58c7294` |
| 1 | `Grigorij-Dudnik_RoboCrew` | `git clone https://github.com/Grigorij-Dudnik/RoboCrew Grigorij-Dudnik_RoboCrew && git -C Grigorij-Dudnik_RoboCrew checkout 356adc610ed8b3a728b2f9e0922bb2e56d91ee0e` |
| 208 | `hesd10_astra-robot-sim2real` | `git clone https://github.com/hesd10/astra-robot-sim2real hesd10_astra-robot-sim2real && git -C hesd10_astra-robot-sim2real checkout e324643f03d05656f49bc4fc575759fd46cb9fe0` |
| 56 | `HorizonRobotics_HoloAgent` | `git clone https://github.com/HorizonRobotics/HoloAgent HorizonRobotics_HoloAgent && git -C HorizonRobotics_HoloAgent checkout ef14d3152ca6246d8ae64920694c6c74581d246c` |
| 92 | `in-context-robot-learning` | `git clone https://github.com/cheng-haha/in-context-robot-learning in-context-robot-learning && git -C in-context-robot-learning checkout bc2b4fb2b9b480ea13c1ba94299d786541ed38b6` |
| 58 | `innate-os` | `git clone https://github.com/innate-inc/innate-os.git innate-os && git -C innate-os checkout 1ed590a17a8db0521b0ea6feb0c82fda672acc9c` |
| 54 | `Jev-as-Policy` | `git clone https://github.com/YuanKJing/Jev-as-Policy Jev-as-Policy && git -C Jev-as-Policy checkout cac97e79845bf56a5b2ef0e6d4928238bf5caee7` |
| 33 | `kairunwen-Awesome-Robot-Use-Agent` | `git clone https://github.com/kairunwen/Awesome-Robot-Use-Agent kairunwen-Awesome-Robot-Use-Agent && git -C kairunwen-Awesome-Robot-Use-Agent checkout 0bc3c05bb5910ba110efc04d906aa79911595f8f` |
| 2 | `llm-token-speed` | `git clone https://github.com/robocurve/llm-token-speed llm-token-speed && git -C llm-token-speed checkout 32015a9bb3086d6ee8975c620a26c77377f1ce28` |
| 1 | `lpigeon_ros-skill` | `git clone https://github.com/lpigeon/ros-skill lpigeon_ros-skill && git -C lpigeon_ros-skill checkout deda49c8a2db0503addb227fb61ef919a3dfadee` |
| 170 | `lpigeon_unitree-go2-mcp-server` | `git clone https://github.com/lpigeon/unitree-go2-mcp-server lpigeon_unitree-go2-mcp-server && git -C lpigeon_unitree-go2-mcp-server checkout e7a6737dc9fd1e107b7b0fbbaf4a66a749c0e6a7` |
| 106 | `manda-real2sim-frontier-assets` | `git clone https://github.com/Manda-Robotics/real2sim-frontier-assets manda-real2sim-frontier-assets && git -C manda-real2sim-frontier-assets checkout 9823603af57844f48783d5618ad4bd88a4c4f2b4` |
| 42 | `manda-RoboLab-Verified` | `git clone https://github.com/Manda-Robotics/RoboLab-Verified manda-RoboLab-Verified && git -C manda-RoboLab-Verified checkout 790c1a5a1d42e5d7b6e294e31bed38f85d4e4671` |
| 25 | `manda-robot-episode-labeler` | `git clone https://github.com/Manda-Robotics/robot-episode-labeler manda-robot-episode-labeler && git -C manda-robot-episode-labeler checkout 31f0b1951d5eef2dd9763793ae7df09de36a0681` |
| 2 | `Mosi-AI_RoboICL` | `git clone https://github.com/Mosi-AI/RoboICL Mosi-AI_RoboICL && git -C Mosi-AI_RoboICL checkout 2898c2b9dd8f42587b24cb555f21777b2e98272b` |
| 1 | `nasa-jpl_rosa` | `git clone https://github.com/nasa-jpl/rosa nasa-jpl_rosa && git -C nasa-jpl_rosa checkout e7e53754bed673e2ce36d1fa311e04e51bfbacad` |
| 155 | `NVlabs_ASPIRE` | `git clone https://github.com/NVlabs/ASPIRE NVlabs_ASPIRE && git -C NVlabs_ASPIRE checkout f4c8939aab0af9b97690c561bd80e282940f7886` |
| 94 | `NVlabs_ENPIRE` | `git clone https://github.com/NVlabs/ENPIRE NVlabs_ENPIRE && git -C NVlabs_ENPIRE checkout 99ee90acf65b5b18957c8382ad580db999528be3` |
| 24 | `omni-mcp_isaac-sim-mcp` | `git clone https://github.com/omni-mcp/isaac-sim-mcp omni-mcp_isaac-sim-mcp && git -C omni-mcp_isaac-sim-mcp checkout 6653138244ebe1f1b6b098528061a4659aeec839` |
| 1917 | `omnilink-tech_omnisim` | `git clone https://github.com/omnilink-tech/omnisim omnilink-tech_omnisim && git -C omnilink-tech_omnisim checkout 2f1f70effc56b7174281acc2c4f573cfe46254bc` |
| 61 | `OpenMind_OM1` | `git clone https://github.com/OpenMind/OM1 OpenMind_OM1 && git -C OpenMind_OM1 checkout 3589af1214a6aa6b0058f30d4ae347ed31ebf265` |
| 18 | `OpenMOSS_OpenETA` | `git clone https://github.com/OpenMOSS/OpenETA OpenMOSS_OpenETA && git -C OpenMOSS_OpenETA checkout 7d4a0a1522ba8ebbd362bde880bad81d2a98f15e` |
| 22 | `openretriever_retriever` | `git clone https://github.com/openretriever/retriever openretriever_retriever && git -C openretriever_retriever checkout b0d766329c045b33eacf5e978e0279f277f7d558` |
| 59 | `openroboto-ai_jev-robot-control` | `git clone https://github.com/openroboto-ai/jev-robot-control openroboto-ai_jev-robot-control && git -C openroboto-ai_jev-robot-control checkout 7a4ed8b72c3c17d7aa790678ed9660df67c10dd3` |
| 63 | `phospho-app_phosphobot` | `git clone https://github.com/phospho-app/phosphobot phospho-app_phosphobot && git -C phospho-app_phosphobot checkout a5de051197c879e3c685d1362f649ce54ee47a3c` |
| 15 | `PhyAgentOS_PhyAgentOS-core` | `git clone https://github.com/PhyAgentOS/PhyAgentOS-core PhyAgentOS_PhyAgentOS-core && git -C PhyAgentOS_PhyAgentOS-core checkout 466e38b1dccdd0f9251ca48cb104a7079343db72` |
| 78 | `piper-astra-jev-recordings` | GitHub release assets: https://github.com/RobotKitAI/piper-astra-jev/releases/tag/recordings |
| 2 | `PlaiPin_rosclaw` | `git clone https://github.com/PlaiPin/rosclaw PlaiPin_rosclaw && git -C PlaiPin_rosclaw checkout 98e7874d1dda17edaf33a5a1eb91309b9a6116c9` |
| 3 | `pollen-robotics_reachy_mini_conversation_app` | `git clone https://github.com/pollen-robotics/reachy_mini_conversation_app pollen-robotics_reachy_mini_conversation_app && git -C pollen-robotics_reachy_mini_conversation_app checkout 5eb39ed1c96790390363cca85d4088ff27a0fdff` |
| 1249 | `public-website` | `git clone https://github.com/anonymous-report-421/public-website public-website && git -C public-website checkout 6ead090465d3b0fb19e90eb70569689c19f73d00` |
| 12 | `RoboClaw-Robotics_RoboClaw` | `git clone https://github.com/RoboClaw-Robotics/RoboClaw RoboClaw-Robotics_RoboClaw && git -C RoboClaw-Robotics_RoboClaw checkout 5238184185efee5c36017f10a79c2e0de4830690` |
| 2 | `roboharm` | `git clone https://github.com/robocurve/roboharm.git roboharm && git -C roboharm checkout 5952f0035037604265d5cf38fa24938d4acf7133` |
| 43 | `RoboProbe` | `git clone https://github.com/RoboProbe/RoboProbe RoboProbe && git -C RoboProbe checkout 9ffacc54f372beef6479d61711bf4945e31ae4ce` |
| 64 | `RobotecAI_rai` | `git clone https://github.com/RobotecAI/rai RobotecAI_rai && git -C RobotecAI_rai checkout 6802d4073e8caa2ab72c5509fa1eeeb659663f64` |
| 180 | `robotmcp_ros-mcp-server` | `git clone https://github.com/robotmcp/ros-mcp-server robotmcp_ros-mcp-server && git -C robotmcp_ros-mcp-server checkout 476591ac058f58cae810cc775e50b51f8d3d757c` |
| 203 | `ros-claw_rosclaw` | `git clone https://github.com/ros-claw/rosclaw ros-claw_rosclaw && git -C ros-claw_rosclaw checkout 0e0d8e453c4af6eb7d89e2425c9ca0da2acc52ba` |
| 1 | `so101-block-sorting` | HF bucket: https://huggingface.co/buckets/mishig/so101-block-sorting |
| 21 | `stationerybench` | `git clone https://github.com/robocurve/stationerybench.git stationerybench && git -C stationerybench checkout 43d5dd380eeb9f555fe691a896a076cf43d3b18b` |
| 10 | `STEERIX-home_robo-jev` | `git clone https://github.com/STEERIX-home/robo-jev STEERIX-home_robo-jev && git -C STEERIX-home_robo-jev checkout b326bedf1e96b42fca75ab9002a42e9ac9f94a14` |
| 108 | `wise-vision_ros2_mcp` | `git clone https://github.com/wise-vision/ros2_mcp wise-vision_ros2_mcp && git -C wise-vision_ros2_mcp checkout 56ec53f41bf08f641bdcd5516866ce2bae8cf09e` |
| 6 | `xiao-yang25_robot-harness` | `git clone https://github.com/xiao-yang25/robot-harness xiao-yang25_robot-harness && git -C xiao-yang25_robot-harness checkout eb414db8e01adff8143f4bb6af5db5581a4732b7` |
| 175 | `XPolicyLab` | `git clone https://github.com/XPolicyLab/XPolicyLab XPolicyLab && git -C XPolicyLab checkout 408b99d959a7b2207f5f785528fefcc019d7b131` |
| 7 | `ZJU4EmbodiedAI_EmbodiedSkills` | `git clone https://github.com/ZJU4EmbodiedAI/EmbodiedSkills ZJU4EmbodiedAI_EmbodiedSkills && git -C ZJU4EmbodiedAI_EmbodiedSkills checkout 296795f6b60bc599fa24d87ae1764acf1c82aa9b` |
