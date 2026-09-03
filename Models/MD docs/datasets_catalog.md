# Indoor Campus Navigation Assistant — Dataset Catalog

**Project:** Indoor campus navigation assistant for blind/low-vision students.
**Pipeline components:**
- (a) YAMNet-style audio classifier (ambient campus sounds)
- (b) YOLO-nano object detector (indoor obstacles)
- (c) Depth / free-space segmentation
- (d) LLM prompted to verbalize structured route data
- (e) Synthetic / simulation data (Unity already in the stack)

For every dataset: **name, scale, classes, license, URL, recommended use.**

---

## 1. Audio Datasets (YAMNet fine-tuning)

### 1.1 AudioSet (Google) — primary pre-training / target classes
- **Scale:** 2,084,320 human-labeled 10-second YouTube clips; ontology of **632 audio event classes**; Balanced subset = 22,176 segments; Evaluation set 20,367 segments.
- **License:** Dataset under **CC BY 4.0**; ontology CC BY-SA 4.0. **Caveat:** clips are YouTube-derived — many are not freely-redistributable; only **embeddings / features** or **list of YouTube IDs** are downloadable, not the audio itself.
- **Direct URL:** http://research.google.com/audioset/ and https://research.google.com/audioset/download.html
- **Hackathon usability:** Use as a *target label list* and for *downstream fine-tuning* on classes that overlap with campus sounds (Speech, Footsteps, Door, Alarm, Siren, Bell, Keyboard, Click, Knock, Crowd, Motor noise, Traffic, etc.). Use the swagshaw/audioset-downloader (https://github.com/swagshaw/audioset-downloader) to pull individual segments.
- **Best in pipeline:** YAMNet backbone already trained on AudioSet ontology (521 classes overlap). Use as a **starting point** — fine-tune the head for the 15-30 campus-specific sounds you care about.
- **2024-2025 relevant update:** "A Refined AudioSet with Multi-Stage LLM Label Reannotation" (https://arxiv.org/html/2508.15429v1) and "AudioSet-Tools" (https://link.springer.com/article/10.1186/s13636-025-00436-z) provide cleaner labels for the same ontology — recommended for hackathon label-noise reduction.

### 1.2 ESC-50 — small benchmark for environmental sounds
- **Scale:** **2,000** clips / **50** balanced classes (40 clips/class), 5 sec each.
- **License:** **CC BY-NC 3.0** (non-commercial). ⚠️ **Hackathon/demo OK** but not for commercial shipping.
- **Direct URL:** https://github.com/karolpiczak/esc-50
- **Best in pipeline:** Quick sanity-check fine-tune / cross-eval baseline. Excellent for "doors creaking, footsteps, clock tick" type ambient classes.
- **Subclass ESC-10** (10 classes subset): https://dcase-repo.github.io/dcase_datalist/datasets/sounds/esc_10.html — even smaller, used as a debug set.

### 1.3 UrbanSound8K — urban ambient + crowd scenes
- **Scale:** **8,732** labeled sound excerpts (≤4 s) / **10** classes: air_conditioner, car_horn, children_playing, dog_bark, drilling, engine_idling, gun_shot, jackhammer, siren, street_music.
- **License:** **CC BY-NC 4.0** (Attribution-NonCommercial). ⚠️ Demo OK, no commercial.
- **Direct URL:** https://urbansounddataset.weebly.com/ and https://www.kaggle.com/datasets/chrisfilo/urbansound8k
- **Best in pipeline:** Train detector for `children_playing`, `street_music`, `siren` — directly useful in crowded campus hallways and outdoor transitions.

### 1.4 FSD50K (Freesound)
- **Scale:** **51,197** clips / **200** sound classes drawn from AudioSet ontology (144 leaf, 56 higher-level); **108.3 hours** of multi-labeled audio.
- **License:** Per-clip **Creative Commons** (mix of CC0, CC-BY, CC-BY-NC). ⚠️ Most clips OK for demo, but **must check each clip's license** if redistributing.
- **Direct URL:** https://fsannotator.upf.edu/fsd/release/FSD50K/ and https://zenodo.org/records/4060432
- **Best in pipeline:** **Best mid-size training set** for fine-tuning YAMNet on the 521 → ~50 campus-relevant classes. Larger and more diverse than ESC-50 / UrbanSound8K. ~108 hours is fine for a hackathon GPU.

### 1.5 DCASE Challenge datasets (2018-2025)
**Annual competition; each year has new task data.** All DCASE data is free for non-commercial research use; specific licenses are listed per task.

- **TAU Urban Acoustic Scenes 2018/2019** — 10 scene classes (Airport, Indoor shopping mall, Metro station, Pedestrian street, Public square, Street with traffic, Tram, Bus, Metro, Park). 12 European cities. **~64 hours**. License: research-only, attribution required.
  - 2018 Mobile: https://dcase-repo.github.io/dcase_datalist/datasets/scenes/tau_asc_2018_dev.html
  - 2019 Dev: https://dcase-repo.github.io/dcase_datalist/datasets/scenes/tau_asc_2019_dev.html
  - 2019 Eval: https://zenodo.org/records/3063822
- **DCASE 2020-2024 Task 4 — Sound Event Detection in Domestic Environments (DESED)** — 10-sec clips in domestic environments (the most relevant to indoor campus rooms: offices, kitchens, hallways).
  - URL: https://dcase.community/challenge2024/task-sound-event-detection-with-heterogeneous-training-dataset-and-potentially-missing-labels
  - URL: https://dcase.community/challenge2023/task-sound-event-detection-with-weak-labels-and-synthetic-soundscapes
  - URL: https://project.inria.fr/desed/dcase-challenge/dcase-2020-task-4/
  - License: research use only (INRIA). Synthetic soundscapes are CC0 / public.
  - **Best in pipeline:** Background + foreground event annotations for things like *door slam, keyboard typing, phone ring, speech* — the exact soundscape of an office/classroom.
- **DCASE 2025 Challenge** — current year's tasks (SELD, SELD under domain shift, etc.). URL: https://dcase.community/challenge2025/index — use for *fresh* evaluation if your classifier is strong.

### 1.6 Emergency / Alarm / Siren / Fire Sounds
- **Large-scale audio dataset for emergency vehicle sirens and road noises (Nature Scientific Data 2022)** — 1,800 audio files, 2 classes (siren vs. road noise). URL: https://www.nature.com/articles/s41597-022-01727-2
- **Emergency Vehicle Siren Sound Classification (Kaggle)** — 3-second WAV files, emergency vehicles vs. traffic. URL: https://www.kaggle.com/code/aryashah2k/emergency-vehicle-siren-sound-classification
- **Kaggle "Large-scale audio dataset for emergency vehicle sirens and road noises"** — 600 audio files, 400 emergency + 200 non-emergency. URL: https://github.com/Bravonoid/emergency-sound-detection
- **Best in pipeline:** Train a high-priority *alert-channel* class. YAMNet already knows "siren" / "alarm" / "smoke detector" from AudioSet, but biasing it to *fire alarms specific to US/EU campus buildings* (long sustained beeps) is hackathon-valuable. License varies — most are research-only.

### 1.7 Smaller / domain-specific corpora
- **VGG-Sound** — 200k 10-sec clips, 309 classes. URL: https://www.robots.ox.ac.uk/~vgg/data/vggsound/ — research-only, complements AudioSet for fine-tuning.
- **MACS (Multi-Annotator Captioned Soundscapes)** — TAU, for caption-style description. URL: https://dcase-repo.github.io/dcase_datalist/datasets/list.html
- **AudioCaption (Listen and Tell)** — SJTU, caption-style. Same DCASE list.

### 1.8 Speech / TTS (component (d) LLM → speech output)
- **Mozilla Common Voice** — 13k+ hours of speech, 100+ languages. **CC0** (public domain). URL: https://commonvoice.mozilla.org/en/terms — use to demo the *output* TTS channel is multilingual/accessible.
- Not needed for training if you use a stock TTS API; mention only.

---

## 2. Object Detection Datasets (YOLO-nano fine-tuning)

### 2.1 COCO 2017 — universal backbone
- **Scale:** 118k training / 5k val images, **80** object classes.
- **License:** Flickr-derived, **CC BY 4.0** for annotations; image rights vary.
- **Direct URL:** https://cocodataset.org/index.htm
- **Relevant classes for indoor nav:** *person, chair, dining table, laptop, backpack, handbag, suitcase, bottle, cup, keyboard, mouse, remote, cell phone, book, clock, fire hydrant, stop sign, traffic light*. **MISSING: door, stair, escalator, elevator, signage.**
- **Best in pipeline:** Pre-train YOLO-nano from COCO weights, then fine-tune on door/stair/escalator-specific datasets below.

### 2.2 Open Images V7 — broadest open-vocabulary detection
- **Scale:** **1,743,042** train / **41,620** val images, **601** classes with bounding boxes.
- **License:** **CC BY 4.0** for annotations; image rights follow original Flickr source.
- **Direct URL:** https://storage.googleapis.com/openimages/web/factsfigures_v7.html and https://docs.ultralytics.com/datasets/detect/open-images-v7
- **Relevant classes for indoor nav:** *Person, Door, Stairs, Chair, Table, Backpack, Suitcase, Sign, Clock, Cabinetry, Whiteboard.* Some indoor-relevant classes are present but annotation density is lower than COCO.
- **Best in pipeline:** Excellent **backbone warm-up**, but for *door/stair/escalator/elevator* use specialized datasets below.

### 2.3 Critical indoor-hazard datasets

| Dataset | Scale | Classes | License | URL |
|---|---|---|---|---|
| **DoorDet (2025)** | multi-class, fine-grained door | cabinet, refrigerator, door, window, etc. | Research | https://arxiv.org/html/2508.07714v1 |
| **Indoor Objects Detection (Roboflow)** | 1,349 images / 7,331 boxes / 10 classes | chair, table, door, cabinetDoor, etc. | Public Roboflow | https://universe.roboflow.com/project-tgiyj/indoor-objects-detection-5j3uk |
| **Door / Stair / Escalator (Roboflow)** | small | door, stair, escalator | Public Roboflow | https://universe.roboflow.com/devil-assassin-zrncn/escalator-stairs-lsrlf |
| **Stair dataset (Mendeley)** | 2,670 train / 424 val | stairs | Research | https://data.mendeley.com/datasets/3jjdm6rn96/3 |
| **Stairs Image Dataset (Kaggle)** | 3,000+ stair images | stair | Public Kaggle | https://www.kaggle.com/datasets/dataclusterlabs/stairs-image-dataset |
| **Indoor-objects YOLO dataset (Kaggle)** | YOLOv5 format, indoor | door, chair, table, person, etc. | Public | https://www.kaggle.com/datasets/thepbordin/indoor-object-detection |
| **Real-time 2D–3D door dataset (Intel Realsense)** | RGB-D door images | door | Public | https://pmc.ncbi.nlm.nih.gov/articles/PMC8082488/ |
| **Indoor Object Detection (Synthetic — Mendeley)** | synthetic indoor objects | multi-class | Public | https://data.mendeley.com/datasets/nnph98d3kc |
| **MCIndoor20000** | 20,000 images, fully labeled indoor | multi-class indoor | Public | (search: MCIndoor20000) |

### 2.4 Blind-navigation specific datasets
- **"A Dataset for Crucial Object Recognition in Blind and Low-Vision Individuals" (arXiv 2024)** — 21 navigational videos from public platforms, with crucial objects for BLV: crosswalk, pole, traffic light, door, etc. URL: https://arxiv.org/html/2407.16777v2
- **"A dataset for the recognition of obstacles on blind sidewalk" (Universal Access in the Information Society 2023)** — outdoor obstacles (Tang et al.). URL: https://dl.acm.org/doi/abs/10.1007/s10209-021-00837-9
- **YOLO-OD dataset (Sensors 2024)** — for blind & visually impaired obstacle detection. URL: https://www.mdpi.com/1424-8220/24/23/7621
- **HAODVIP-ADL (2025)** — Activities of Daily Living indoor detection for VIP. URL: https://www.researchgate.net/publication/390040054_A_hybrid_object_detection_approach_for_visually_impaired_persons
- **Nav-YOLO (ISPRS IJGI 2025)** — lightweight detector for visually impaired navigation. URL: https://www.mdpi.com/2220-9964/14/9/364
- **"A Novel Dataset for Intelligent Indoor Object Detection Systems"** — labeled indoor object dataset. URL: https://journals.bilpubgroup.com/index.php/aia/article/view/925

### 2.5 Crowded indoor / crowd detection (hallway problem)
- **CrowdHuman** — 470k human instances, train/val; rich annotations (head, body, ignore). URL: https://www.crowdhuman.org/ and https://github.com/Asthestarsfalll/CrowdHuman. License: research use, attribution.
- **Mall Dataset (CUHK)** — 60,000+ pedestrians from a public shopping-mall webcam, used for crowd counting/profiling. URL: https://personal.ie.cuhk.edu.hk/~ccloy/downloads_mall_dataset.html
- **ATC Shopping Center Tracking (ATR)** — pedestrian tracking + 3D range sensors in a shopping center. URL: https://dil.atr.jp/crest2010_HRI/ATC_dataset/
- **Best in pipeline:** **CrowdHuman is the single best dataset for "how do I move through a crowded hallway of students?"** — paired with YOLO-nano person class from COCO.

### 2.6 Hallway / corridor specific
- **"indoor-navigation" Roboflow dataset (Akhash)** — 1,115 images of hallway-window-door-elevator. URL: https://universe.roboflow.com/akhash/indoor-navigation-xs4of
- **"An effective obstacle detection system using deep learning for visually impaired" (Biomedical Signal Processing and Control 2024)** — https://www.sciencedirect.com/science/article/pii/S2090447923002769

---

## 3. Depth / Segmentation Datasets (component c)

### 3.1 NYU Depth V2
- **Scale:** 1,449 densely labeled aligned RGB+Depth pairs + 464 video sequences; 407,024 raw frames; 26 indoor scene classes.
- **Sensor:** Microsoft Kinect v1.
- **License:** Research use. URL: https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html
- **Best in pipeline:** Canonical *indoor monocular depth* fine-tune. ~5 GB raw, perfect for a hackathon.

### 3.2 SUN RGB-D
- **Scale:** **10,335** RGB-D images, dense 2D/3D annotations, **~700** object categories.
- **License:** Research use. URL: https://rgbd.cs.princeton.edu/
- **Best in pipeline:** Pairs nicely with NYU Depth V2 for both depth *and* semantic segmentation.

### 3.3 ScanNet
- **Scale:** 2.5M views across **1,500+** RGB-D scans, 3D camera poses, surface reconstructions, instance-level semantic labels.
- **License:** Terms-of-use agreement (research). URL: http://www.scan-net.org/ and http://www.scan-net.org/ScanNet/
- **Best in pipeline:** Heavyweight 3D scene-understanding; not necessary for YOLO-nano fine-tune but useful for embodied-AI / SLAM if the team goes that direction.

### 3.4 ScanNet++ (ICCV 2023 Oral)
- **Scale:** Higher-fidelity RGB-D + 3D meshes, captures + ground truth. License: research. URL: https://scannetpp.mlsg.cit.tum.de/scannetpp/
- **Best in pipeline:** If you need higher-fidelity indoor meshes than ScanNet.

### 3.5 Matterport3D
- **Scale:** **10,800 panoramic views from 194,400 RGB-D images** across **90** building-scale scenes.
- **License:** **CC BY-NC-SA 3.0 US** (non-commercial). URL: https://niessner.github.io/Matterport/
- **Best in pipeline:** Standard embodied-AI indoor corpus; non-commercial is fine for hackathon demo.

### 3.6 Habitat-Matterport 3D (HM3D)
- **Scale:** **1,000** large-scale 3D environments for embodied AI.
- **License:** **CC BY-NC-SA 3.0 US** for Matterport3D-based models. URL: https://github.com/facebookresearch/habitat-matterport3d-dataset
- **Best in pipeline:** Plug-and-play with Habitat-Sim for synthetic-data generation.

### 3.7 Hypersim (Apple)
- **Scale:** **77,400** photorealistic synthetic images across **461** indoor scenes, per-pixel labels + ground truth depth.
- **License:** Research use, attribution. URL: https://github.com/apple/ml-hypersim and https://machinelearning.apple.com/research/hypersim
- **Best in pipeline:** **Best synthetic indoor depth/semantic dataset** — closes the sim-to-real gap for monocular depth fine-tuning without needing to render Unity scenes yourself.

### 3.8 ARKitScenes (Apple)
- **Scale:** **5,047** captures of **1,661** unique scenes using Apple LiDAR.
- **License:** **CC BY 4.0**. URL: https://github.com/apple/ARKitScenes and https://machinelearning.apple.com/research/arkitscenes
- **Best in pipeline:** Real depth with consumer-hardware depth range (0.5-6 m, median max ~2.4 m) — exactly the "what's in front of me while walking" range. The single most relevant depth dataset for a wearable navigation device.

### 3.9 ADE20K
- **Scale:** **>27,000** images from SUN/Places, **3,000+** object categories, full scene parsing. Train 20,210 / val 2,000.
- **License:** Research use. URL: https://ade20k.csail.mit.edu/ and https://github.com/CSAILVision/ADE20K
- **Best in pipeline:** Universal indoor/outdoor semantic-segmentation pre-training; classes include walls, floors, doors, stairs, windows, signs.

### 3.10 Free-space / traversability
- **Ikhyeon-Cho/urban-traversability-dataset** — self-supervised traversability dataset. URL: https://github.com/Ikhyeon-Cho/urban-traversability-dataset
- **THUD++ (arXiv 2024)** — 6,000+ frames / 60 min of dynamic indoor scenes, navigation emulator. URL: https://arxiv.org/html/2412.08096v1
- **"Learning self-supervised traversability with navigation experiences of mobile robots"** — risk-aware self-training. (See arXiv 2024)
- **OpenLORIS-Object** — RGB-D lifelong robotic vision dataset, 5 environments, real-world robustness. URL: https://lifelong-robotic-vision.github.io/dataset/object.html

### 3.11 KITTI (outdoor depth reference)
- **Scale:** 93k+ depth maps + LiDAR + stereo + GPS.
- **License:** CC BY-NC-SA 3.0. URL: https://www.cvlibs.net/datasets/kitti/
- **Best in pipeline:** Use for **outdoor-to-indoor transition zone** depth (campus entrances, plaza steps) and for depth-completion benchmarks.

---

## 4. Indoor Localization / Mapping (component d, position input)

### 4.1 Visual localization benchmarks
- **7-Scenes (Microsoft)** — 7 indoor environments, 500-1000 frame sequences each, Kinect RGB-D, ground truth from KinectFusion. URL: https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/
- **InLoc (CVPR 2018 / TPAMI 2021)** — large-scale indoor 6DoF localization: panoramic 3D scans registered to floor plans + mobile-phone query photos. URL: https://github.com/HajimeTaira/InLoc_demo
- **Cambridge Landmarks (outdoor)** — 5 landmark buildings in Cambridge UK. URL: https://github.com/vislearn/dsacstar/blob/master/README.md
- **Extended Cambridge Landmarks (ECL, CVPR 2025)** — extended outdoor re-localization. URL: https://ronferens.github.io/extcambridgelandmarks/
- **Aachen Day-Night / Visual Localization Benchmark** — outdoor but useful for night-time transfer. URL: https://www.visuallocalization.net/datasets/

### 4.2 Indoor mapping standards / crowdsourced
- **OGC IndoorGML 1.0/1.1** — open XML schema for indoor spatial / navigation. URL: https://www.ogc.org/standards/indoorgml/
- **OSM Indoor Mapping** — community standard for indoor tagging. URL: https://wiki.openstreetmap.org/wiki/Indoor_Mapping
- **"A Data Model for Using OpenStreetMap to Integrate Indoor and Outdoor Route Planning" (Sensors 2018)** — pedestrian route planning using OSM. URL: https://www.mdpi.com/1424-8220/18/7/2100
- **MazeMap** — commercial indoor-navigation for campuses; reference for campus topology, but proprietary. URL: https://www.mazemap.com/post/indoor-navigation-campuses
- **Mappedin** — proprietary campus map provider; reference only.

### 4.3 Indoor WiFi / IMU localization (alt modality for SLAM input)
- **Indoor Location & Navigation (Kaggle competition)** — WiFi + geomagnetic + iBeacon dense fingerprints, ground-truth waypoints. URL: https://www.kaggle.com/competitions/indoor-location-navigation/data
- **ATC Shopping Center Tracking (ATR)** — pedestrian tracking + 3D range sensors in a shopping center. URL: https://dil.atr.jp/crest2010_HRI/ATC_dataset/

---

## 5. LLM Direction-Generation / Navigation Instruction Data (component d)

These are vision-language navigation (VLN) corpora. For a hackathon, you usually **prompt** an LLM rather than train on these — but they shape the **style and content** of your prompt templates.

### 5.1 Room-to-Room (R2R)
- **Scale:** 21,567 natural-language navigation instructions over 90 buildings (Matterport3D).
- **License:** Research; built on Matterport3D (CC BY-NC-SA 3.0). URL: https://bringmeaspoon.org/
- **Style:** "Walk past the kitchen island, turn right at the couch, enter the door on your left." → use as few-shot examples.

### 5.2 Room-across-Room (RxR) — Google
- **Scale:** **126,000** instructions in English, Hindi, Telugu; **~10M words**; **16,500** unique paths; word-by-word spatial grounding.
- **License:** Research; built on Matterport3D. URL: https://github.com/google-research-datasets/RxR and https://ai.google.com/research/rxr/
- **Style:** Richer, more varied English instructions. Ideal few-shot corpus for LLM prompt.

### 5.3 Touchdown (Cornell)
- **Scale:** **9,326** natural-language navigation + spatial reasoning examples in Google Street View.
- **License:** Research. URL: https://github.com/lil-lab/touchdown and https://sites.google.com/view/streetlearn/touchdown
- **Style:** Outdoor-style ("turn left at the green dumpster") — useful for *campus-quad-to-building* transitions.

### 5.4 Map2Seq (Heidelberg)
- **Scale:** **7,672** natural-language landmark navigation instructions paired with OpenStreetMap route paths.
- **License:** Research. URL: https://www.cl.uni-heidelberg.de/statnlpgroup/map2seq/
- **Style:** Pure landmark-based, structured turn-by-turn; **closest match** to the LLM-prompted route-to-instruction task you'll build. Recommended for prompt-template design.

### 5.5 Vision-language navigation prompting research
- **"Semantic Map-based Generation of Navigation Instructions" (arXiv 2024)** — instruction-generation benchmark. URL: https://arxiv.org/html/2403.19603
- **"Controllable Navigation Instruction Generation with Chain of Thought"** — CoT with Landmarks. URL: https://dl.acm.org/doi/10.1007/978-3-031-73397-0_3
- **MapGPT (ACL 2024)** — Map-guided prompting with adaptive path planning for embodied navigation. URL: https://aclanthology.org/2024.acl-long.529.pdf
- **FLAME (AAAI 2025)** — Multimodal LLM for urban VLN. URL: https://ojs.aaai.org/index.php/AAAI/article/download/32974/35129
- **NavRAG** — Retrieval-Augmented LLM for embodied navigation. (Search arXiv)

### 5.6 Accessibility Q&A (alt input modality)
- **VizWiz-VQA** — 31,000 visual questions from blind users. **CC BY 4.0**. URL: https://vizwiz.org/tasks-and-datasets/vqa/
- **VizWiz-LF** — 4,200 long-form answers to 600 VQs. URL: https://openreview.net/forum?id=z7FvXbyyrM
- **Best use:** Reuse the **"what's in front of me?"** style of blind-user questions to design your LLM prompt + verbalization UX.

---

## 6. Simulation / Synthetic Data (Unity + others)

### 6.1 Unity Perception Package
- **Scale:** Generate as much as you need; toolkit includes bounding box, instance & semantic segmentation, depth labels with full domain randomization.
- **License:** Apache-2.0 (Unity Perception), free. URL: https://github.com/Unity-Technologies/com.unity.perception and https://docs.unity3d.com/Packages/com.unity.perception@1.0/manual/index.html
- **Best in pipeline:** **Build your own campus hallway simulator** to generate door, stair, person-bag-detection training images. Full control over class set, lighting, occlusions.

### 6.2 PeopleSansPeople (Unity)
- **Scale:** Parametric 3D human assets + scene templates; generates 2D + 3D bounding box, instance + keypoint labels.
- **License:** Apache-2.0, free. URL: https://unity-technologies.github.io/PeopleSansPeople/
- **Best in pipeline:** Generate dense crowd images for crowded-hallway detector fine-tuning — addresses the core hallway problem.

### 6.3 Habitat-Sim (Meta)
- **Scale:** High-performance physics-enabled 3D simulator, supports HM3D, HSSD, Matterport3D, Replica, ScanNet. Thousands of FPS.
- **License:** MIT-style. URL: https://github.com/facebookresearch/habitat-sim and https://aihabitat.org/docs/concepts/
- **Best in pipeline:** If you decide to do embodied navigation / RL, use Habitat-Sim with HM3D.

### 6.4 InternScenes (arXiv 2025)
- **Scale:** Large-scale simulatable indoor scene dataset. URL: https://arxiv.org/html/2509.10813v1
- **Best in pipeline:** Alternative to HM3D if you need a recent 2025 dataset.

### 6.5 CARLA (driving sim — outdoor)
- **Scale:** Driving sim with pedestrians + dynamic agents. License: MIT.
- URL: https://github.com/anita-hu/simulanes (Simulanes — CARLA lane dataset) and https://huggingface.co/datasets/immanuelpeter/carla-autopilot-multimodal-dataset
- **Best in pipeline:** Use only for the *outdoor-to-indoor transition* (campus street → building entrance).

### 6.6 Hypersim (see §3.7)
- Already covered; consider this both a depth dataset and a synthetic source.

---

## 7. Master Summary Table

| Dataset | Component | Scale | License | URL |
|---|---|---|---|---|
| AudioSet | (a) Audio | 2.08M clips / 632 classes | CC BY 4.0 (clips YT-derived) | http://research.google.com/audioset/ |
| ESC-50 | (a) Audio | 2,000 / 50 classes | CC BY-NC 3.0 | https://github.com/karolpiczak/esc-50 |
| UrbanSound8K | (a) Audio | 8,732 / 10 classes | CC BY-NC 4.0 | https://urbansounddataset.weebly.com/ |
| FSD50K | (a) Audio | 51,197 / 200 classes (108 h) | Per-clip CC | https://fsannotator.upf.edu/fsd/release/FSD50K/ |
| TAU Urban Acoustic Scenes 2019 | (a) Audio | ~64 h / 10 scenes | Research | https://dcase-repo.github.io/dcase_datalist/datasets/scenes/tau_asc_2019_dev.html |
| DESED / DCASE Task 4 (2020-2024) | (a) Audio | domestic SED | Research (INRIA) | https://dcase.community/challenge2024/task-sound-event-detection-with-heterogeneous-training-dataset-and-potentially-missing-labels |
| DCASE 2025 Challenge | (a) Audio | SELD + new | Research | https://dcase.community/challenge2025/index |
| VGG-Sound | (a) Audio | 200k / 309 classes | Research | https://www.robots.ox.ac.uk/~vgg/data/vggsound/ |
| Emergency Vehicle Siren dataset (Nature 2022) | (a) Audio | 1,800 / 2 classes | Research | https://www.nature.com/articles/s41597-022-01727-2 |
| Mozilla Common Voice | (a/d) Audio / TTS | 13k+ hours | CC0 | https://commonvoice.mozilla.org/en/terms |
| COCO 2017 | (b) Detection | 118k / 80 classes | CC BY 4.0 (ann) | https://cocodataset.org/index.htm |
| Open Images V7 | (b) Detection | 1.74M / 601 classes | CC BY 4.0 (ann) | https://storage.googleapis.com/openimages/web/factsfigures_v7.html |
| DoorDet 2025 | (b) Door | multi-class door | Research | https://arxiv.org/html/2508.07714v1 |
| Indoor Objects Detection (Roboflow) | (b) Indoor | 1,349 imgs / 10 classes | Public | https://universe.roboflow.com/project-tgiyj/indoor-objects-detection-5j3uk |
| Stair dataset (Mendeley) | (b) Stairs | 2,670 train + 424 val | Research | https://data.mendeley.com/datasets/3jjdm6rn96/3 |
| escalator-stairs (Roboflow) | (b) Stairs/Escalator | small | Public | https://universe.roboflow.com/devil-assassin-zrncn/escalator-stairs-lsrlf |
| Stairs Image Dataset (Kaggle) | (b) Stairs | 3,000+ | Public Kaggle | https://www.kaggle.com/datasets/dataclusterlabs/stairs-image-dataset |
| Indoor Objects YOLO (Kaggle) | (b) Indoor | YOLOv5 format | Public | https://www.kaggle.com/datasets/thepbordin/indoor-object-detection |
| Indoor-navigation (Roboflow Akhash) | (b) Hallway | 1,115 imgs | Public | https://universe.roboflow.com/akhash/indoor-navigation-xs4of |
| Crucial Object Recognition (BLV, arXiv 2024) | (b) BLV | 21 videos | Research | https://arxiv.org/html/2407.16777v2 |
| YOLO-OD (Sensors 2024) | (b) BLV | public dataset | Research | https://www.mdpi.com/1424-8220/24/23/7621 |
| HAODVIP-ADL 2025 | (b) BLV ADL | indoor ADL | Research | https://www.researchgate.net/publication/390040054_A_hybrid_object_detection_approach_for_visually_impaired_persons |
| Nav-YOLO (IJGI 2025) | (b) BLV | lightweight model + data | Research | https://www.mdpi.com/2220-9964/14/9/364 |
| CrowdHuman | (b) Crowd | 470k persons | Research | https://www.crowdhuman.org/ |
| Mall Dataset (CUHK) | (b) Crowd | 60k+ pedestrians | Research | https://personal.ie.cuhk.edu.hk/~ccloy/downloads_mall_dataset.html |
| ATC Shopping Center (ATR) | (b/d) Crowd/SLAM | tracking + 3D | Research | https://dil.atr.jp/crest2010_HRI/ATC_dataset/ |
| NYU Depth V2 | (c) Depth | 1,449 dense + 464 seqs | Research | https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html |
| SUN RGB-D | (c) Depth + SemSeg | 10,335 RGB-D | Research | https://rgbd.cs.princeton.edu/ |
| ScanNet | (c) 3D Indoor | 2.5M views / 1,500 scans | Research | http://www.scan-net.org/ |
| ScanNet++ | (c) 3D Indoor | high-fidelity | Research | https://scannetpp.mlsg.cit.tum.de/scannetpp/ |
| Matterport3D | (c/d) Indoor | 90 buildings / 194k RGB-D | CC BY-NC-SA 3.0 | https://niessner.github.io/Matterport/ |
| HM3D | (c/d) Indoor | 1,000 scenes | CC BY-NC-SA 3.0 | https://github.com/facebookresearch/habitat-matterport3d-dataset |
| Hypersim | (c) Synthetic | 77,400 / 461 scenes | Research | https://github.com/apple/ml-hypersim |
| ARKitScenes | (c) Depth | 5,047 / 1,661 scenes | CC BY 4.0 | https://github.com/apple/ARKitScenes |
| ADE20K | (c) SemSeg | >27k imgs / 3k classes | Research | https://ade20k.csail.mit.edu/ |
| THUD++ | (c) Dynamic indoor | 6,000+ frames | Research | https://arxiv.org/html/2412.08096v1 |
| OpenLORIS-Object | (c) Lifelong | RGB-D, 5 envs | Research | https://lifelong-robotic-vision.github.io/dataset/object.html |
| KITTI | (c) Outdoor Depth | 93k+ depth | CC BY-NC-SA 3.0 | https://www.cvlibs.net/datasets/kitti/ |
| 7-Scenes | (d) Indoor Loc | 7 envs | Research | https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/ |
| InLoc | (d) Indoor Loc | panoramas + queries | Research | https://github.com/HajimeTaira/InLoc_demo |
| Cambridge Landmarks | (d) Outdoor Loc | 5 buildings | Research | https://github.com/vislearn/dsacstar/blob/master/README.md |
| OGC IndoorGML | (d) Standard | XML schema | Open standard | https://www.ogc.org/standards/indoorgml/ |
| OSM Indoor Mapping | (d) Crowdsource | wiki schema | ODbL | https://wiki.openstreetmap.org/wiki/Indoor_Mapping |
| Indoor Location & Navigation (Kaggle) | (d) WiFi/SLAM | fingerprints | Research | https://www.kaggle.com/competitions/indoor-location-navigation/data |
| Room-to-Room (R2R) | (d/e) VLN | 21,567 instr / 90 bldgs | Research (CC BY-NC-SA via MP3D) | https://bringmeaspoon.org/ |
| Room-across-Room (RxR) | (d/e) VLN | 126k / 3 langs / 16.5k paths | Research (CC BY-NC-SA via MP3D) | https://github.com/google-research-datasets/RxR |
| Touchdown | (d) VLN outdoor | 9,326 instr | Research | https://github.com/lil-lab/touchdown |
| Map2Seq | (d) VLN landmark | 7,672 instr + OSM | Research | https://www.cl.uni-heidelberg.de/statnlpgroup/map2seq/ |
| VizWiz-VQA | (d) BLV VQA | 31,000 VQs | CC BY 4.0 | https://vizwiz.org/tasks-and-datasets/vqa/ |
| MapGPT (ACL 2024) | (d) LLM prompting | paper+data | Research | https://aclanthology.org/2024.acl-long.529.pdf |
| CoT-Landmarks (2024) | (d) LLM CoT | paper+data | Research | https://dl.acm.org/doi/10.1007/978-3-031-73397-0_3 |
| Unity Perception | (e) Synthetic | unlimited | Apache-2.0 | https://github.com/Unity-Technologies/com.unity.perception |
| PeopleSansPeople | (e) Synthetic humans | unlimited | Apache-2.0 | https://unity-technologies.github.io/PeopleSansPeople/ |
| Habitat-Sim | (e) Sim | unlimited | MIT-style | https://github.com/facebookresearch/habitat-sim |
| CARLA (driving) | (e) Sim | unlimited | MIT | https://github.com/carla-simulator/carla |
| InternScenes (2025) | (e) Sim dataset | large | Research | https://arxiv.org/html/2509.10813v1 |

---

## 8. Recommended Minimal Fine-Tuning Plan (Hackathon Demo)

**Goal:** In ~24-48 hours, get a *credible* YAMNet and YOLO-nano demo on campus indoor audio + obstacles + crowd + door/stair.

### Must-have (Tier 1) — small, fast, high signal

1. **YAMNet fine-tune audio head** on:
   - **UrbanSound8K** (8,732 clips, 10 classes) — covers `siren`, `children_playing`, `street_music` → 1 hour to fine-tune.
   - **ESC-50** (2,000 / 50 classes) — proves "footsteps / door creak / clock / breathing" detection. <30 min fine-tune.
   - **Emergency Vehicle Siren dataset** (Kaggle / Nature 2022) — 600-1,800 clips, 2-3 classes — high-impact *alert* class.

   → Result: YAMNet with custom head for ~15-20 campus sounds: footsteps, door slam, speech, alarm, phone ring, keyboard, crowd babble, traffic, elevator ding.

2. **YOLO-nano fine-tune on detection** with:
   - **COCO pre-trained weights** (start) → fine-tune on the Roboflow **Indoor Objects Detection** set (1,349 imgs, 10 classes: door, chair, table, cabinetDoor, etc.).
   - **Mendeley Stair dataset** (2,670 imgs) for stair detection.
   - **CrowdHuman** (subset, ~50k person crops) for crowded-hallway person detection.
   - **Roboflow Akhash indoor-navigation** (1,115 imgs) for hallway+door+elevator sample diversity.

   → Result: YOLO-nano detecting *person, door, chair, table, stairs, handbag, suitcase* in real campus frames at >15 FPS.

### Should-have (Tier 2) — depth / segmentation

3. **NYU Depth V2** (5 GB) — fine-tune a tiny monocular depth head (MiDaS-small or DPT) for *free-space estimation* (where's the floor / where's the obstacle in front of me).
4. **Hypersim** (synthetic) for sim-to-real depth pre-training if you have spare GPU hours.
5. **ADE20K** classes subset for wall/floor/ceiling/door/stair semantic segmentation pre-training.

### Should-have (Tier 2) — LLM spoken directions

6. **R2R / RxR** (just the JSON instruction text, ~50-100 sampled) — use as few-shot examples in your LLM prompt template to *generate* "turn right at the end of the hallway, then enter the third door on your left."
7. **Map2Seq** (7,672 OSM landmark instructions) — same use, but the landmark-based style is closest to the campus use case.
8. **VizWiz-VQA** (subset) — to design *what blind users actually ask* and inform the prompt / UX ("what's in front of me?" → answer with a description).

### Should-have (Tier 2) — synthetic data via Unity

9. **Unity Perception** + **PeopleSansPeople** to render 1k-5k synthetic hallway images with controlled door/stair/elevator placement and crowded students. This is your *unfair advantage* since the team already uses Unity.
10. (Optional) **Habitat-Sim + HM3D** if the team wants to test the LLM direction-generation in a sim before going live.

### Nice-to-have (Tier 3) — only if time permits

11. **OpenStreetMap Indoor Mapping** of the actual campus (1-2 hours of JOSM editing) → feeds IndoorGML-compliant graph into the LLM prompt.
12. **7-Scenes / InLoc** for image-retrieval-based *where am I?* localization.
13. **Indoor Location & Navigation (Kaggle)** for WiFi fingerprint backup positioning.
14. **OGC IndoorGML** schema → export the campus graph to a standard format.

### Minimal final pipeline
```
YAMNet (fine-tuned on UrbanSound8K + ESC-50 + Siren)
      → ambient sound tag (e.g., "crowd of students ahead", "elevator ding", "fire alarm")
YOLO-nano (fine-tuned on Indoor-Roboflow + Stairs + CrowdHuman + Unity synthetic)
      → bounding boxes for person, door, chair, stairs, handbag
NYU-Depth fine-tune (or MiDaS-zero-shot)
      → depth map → free-space mask in front of user
LLM (R2R/RxR/Map2Seq few-shot prompted)
      → structured route (graph) → natural-language turn-by-turn ("Walk forward 5 m, then turn right at the library sign")
TTS (Common Voice-style or stock)
      → spoken output for the blind user
```

**Total dataset downloads (Tier 1 + Tier 2):** ~50 GB, < 1 day of training on a single A100. Demonstrably hackathon-friendly.
