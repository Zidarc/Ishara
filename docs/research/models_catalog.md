# Open-Source Model Catalog — Indoor Campus Navigation Assistant (Blind/Low-Vision)

**Project:** Unity mobile app + Python backend, on-device YAMNet/TFLite audio + YOLOv8n/v11n/v12n camera detection + server-side LLM for spoken directions.
**Date:** 2026-08-31
**Scope:** Catalog of permissively-licensed open-source models relevant to each AI component, with size/license/export recommendations and a recommended stack.

> **Critical licensing warning up front:**
> - **YOLOv8 / YOLOv11 / YOLOv12** (Ultralytics) are released under **AGPL-3.0**. For a closed/hackathon product that you do not open-source end-to-end, you must buy an **Ultralytics Enterprise License**. The cheaper hackathon workaround: use the underlying pre-AGPL repos, **RF-DETR (Apache-2.0)**, or **EfficientDet-Lite / MobileNet-SSD (Apache-2.0)** instead. See [Ultralytics License](https://www.ultralytics.com/license) and [Roboflow RF-DETR](https://github.com/roboflow/rf-detr).

---

## 1. Audio Classification (ambient sound → hazard cues)

Goal: detect door slams, fire alarms, footsteps, voices, escalators, microwaves, etc. — useful context for indoor wayfinding.

### 1.1 Comparison table

| Model | Params | AudioSet mAP / ESC-50 | License | TFLite / ONNX mobile export | Hugging Face | Code |
|---|---|---|---|---|---|---|
| **YAMNet** (MobileNet-v1 depthwise separable) | 3.7M (~3.8 MB) | AudioSet 0.321 mAP / ~84% ESC-50 | Apache-2.0 | First-class TFLite; community ONNX | TF Hub: `yamnet` | [github.com/tensorflow/models/tree/master/research/audioset/yamnet](https://github.com/tensorflow/models/blob/master/research/audioset/yamnet/params.py), [TF Hub tutorial](https://www.tensorflow.org/hub/tutorials/yamnet) |
| **AST** (Audio Spectrogram Transformer, MIT/Ya) | ~87M | AudioSet 0.485 mAP / 95.6% ESC-50 | MIT / research-use for checkpoints | Export to ONNX possible; TFLite feasible with effort | [MIT/ast-finetuned-audioset-10-10-0.4593](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593) | [github.com/YuanGongND/ast](https://github.com/YuanGongND/ast) |
| **BEATs-iter3** (Microsoft) | ~90M (iter3) | AudioSet-2M 0.506 mAP / 98.1% ESC-50 | MIT (code) | ONNX export doable; transformer size heavy for mobile | [microsoft/BEATs-iter3 (HF)](https://huggingface.co/microsoft/BEATs-iter3) | [github.com/microsoft/unilm/tree/master/beats](https://github.com/microsoft/unilm/blob/master/beats/README.md) |
| **PaSST** (Patchout, Koutini et al.) | ~90M (S/16) up to 360M (L/14) | ~0.471 mAP AudioSet | MIT | ONNX; large, but PaSST-S-L/PaSST-S-N are smaller variants | (HF mirror available) | [github.com/kkoutini/PaSST](https://github.com/kkoutini/PaSST) |
| **PANNs CNN14** (Q. Kong et al.) | 80.8M | 0.431 mAP AudioSet | MIT (code), CC-BY weights | ONNX; TFLite possible | [qiuqiangkong/audioset_tagging_cnn](https://github.com/qiuqiangkong/audioset_tagging_cnn) | same |
| **EfficientAT** (Schmid et al., 2023 ICASSP) | Mobile variants ≈ 3–10M params (MN/HEAR) | ~0.460 mAP AudioSet for `mn10_as` | MIT | Designed for mobile; ONNX export straightforward | [fschmid56/EfficientAT](https://github.com/fschmid56/EfficientAT) (HF upload of `mn10_as` available) | same |
| **YAMNet (TF-Lite fork with fixed I/O)** | 3.7M | 0.321 mAP | Apache-2.0 | Drop-in TFLite | — | [antonyharfield/tflite-models-audioset-yamnet](https://github.com/antonyharfield/tflite-models-audioset-yamnet) |
| **Google MediaPipe Audio Classifier (YAMNet backend)** | 3.7M | 0.321 mAP | Apache-2.0 | TFLite + MediaPipe Tasks | [MediaPipe audio classifier guide](https://developers.google.com/edge/mediapipe/solutions/audio/audio_classifier) | — |

### 1.2 Notes & fine-tuning
- **YAMNet** is the right *default* for the phone: tiny (3.8 MB), TFLite-first, Apache-2.0, and fine-tunes easily on a small custom set (cat/dog example uses the same `yamnet` embedding extractor). See [TFLite codelab](https://developers.google.com/codelabs/tflite-audio-classification-custom-model-android) and [TF audio transfer-learning](https://www.tensorflow.org/tutorials/audio/transfer_learning_audio).
- **EfficientAT (mn10_as, mn20_as, mn30_as, mn40_as, mn100_as)** is the *upgraded* drop-in if you need higher accuracy at the cost of more params. Distilled from PaSST/AST, so it retains transformer accuracy but with CNN speed. Perfect for a research-heavy hackathon team.
- **AST, BEATs, PaSST** are best treated as *server-side* audio taggers (~90M+ params, INT8 quantization feasible) for offline fine-tuning + distilling back into a YAMNet/EfficientAT student.
- **Per Qualcomm AI Hub / Zetic Melange** case studies, YAMNet also runs on **NPU** accelerators on-device.

### 1.3 Recommended
- **Primary:** YAMNet (TFLite, Apache-2.0) — 3.8 MB, 521 classes, fast, fine-tunes in hours.
- **Lighter fallback:** keep YAMNet; it's already the smallest viable option.
- **Higher-accuracy option:** EfficientAT `mn10_as` (Apache/MIT) — swap-in if YAMNet's 0.321 mAP proves insufficient on campus-specific sounds (footsteps, microwaves, elevator chimes).

---

## 2. Obstacle / Hazard Detection (bounding-box path)

### 2.1 Comparison table

| Model | Params | Size | COCO mAP | Mobile FPS (typical) | License | Mobile export | Source |
|---|---|---|---|---|---|---|---|
| **YOLOv8n** (Ultralytics) | 3.2M | ~6 MB (fp32) / 2.3 MB (INT8 TFLite) | 37.3 | 4.1 ms latency (V100) — fastest in class | **AGPL-3.0** | TFLite, CoreML, ONNX | [github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) |
| **YOLOv11n** (Ultralytics, Sep 2024) | 2.6M | ~5.2 MB | 39.5 | 2.4 ms latency | **AGPL-3.0** | TFLite, CoreML, ONNX | [Ultralytics YOLO11](https://docs.ultralytics.com/models/yolo11) |
| **YOLOv12n** (Ultralytics, Feb 2025) | 2.9M | ~5.5 MB | 40.6 | ≈ YOLOv11n speed | **AGPL-3.0** | Same | [yolov12.com](https://yolov12.com/), [github.com/sunsmarterjie/yolov12](https://github.com/sunsmarterjie/yolov12) |
| **YOLOv10n** (NMS-free) | 2.3M | ~5.2 MB | 38.5 | 5.5 ms | GPL-3.0 (model + code) | ONNX, CoreML, TFLite | [github.com/THU-MIG/yolov10](https://github.com/THU-MIG/yolov10) |
| **YOLOv9 Gelan-s (t/ c)** | 7.1M | 12 MB | 46.8 (S) | 11.5 ms | GPL-3.0 | ONNX | [github.com/WongKinYiu/yolov9](https://github.com/WongKinYiu/yolov9) |
| **YOLO-NAS (S/N/M)** (Deci) | 19–54M | 20–60 MB | 47.5 NAS-S | very fast (N) | **Custom non-commercial for pretrained weights** | CoreML, ONNX, TFLite via SuperGradients | [github.com/Deci-AI/super-gradients](https://github.com/Deci-AI/super-gradients/blob/master/YOLONAS.md) |
| **RF-DETR-Nano** (Roboflow, Mar 2025) | ~30M | ~75 MB | 48.5 COCO + 60+ mAP on RF data | 5–8 ms (T4 GPU) | **Apache-2.0** ✅ | ONNX, CoreML (work in progress TFLite) | [github.com/roboflow/rf-detr](https://github.com/roboflow/rf-detr) |
| **RT-DETR-R18 / -R34** (Baidu CVPR 2024) | 20M / 31M | 40–70 MB | 46.5 / 48.9 | real-time | Apache-2.0 | ONNX | [github.com/lyuwenyu/RT-DETR](https://github.com/lyuwenyu/RT-DETR) |
| **RT-DETRv3-R18 / -R34** | 20M / 31M | — | 48.1 / 49.1 | improved training | Apache-2.0 | ONNX | [arXiv 2409.08475](https://arxiv.org/html/2409.08475v3) |
| **EfficientDet-Lite0** (Google) | 3.2M | 4.4 MB | 25.7 COCO | 37 ms (Pi 4) | Apache-2.0 | First-class TFLite | [TensorFlow Blog](https://blog.tensorflow.org/2021/06/easier-object-detection-on-mobile-with-tf-lite.html) |
| **EfficientDet-Lite2** | 4.7M | 5.8 MB | 30.5 COCO | 49 ms | Apache-2.0 | First-class TFLite | same |
| **EfficientDet-Lite3** | 6.4M | 7.6 MB | 33.8 COCO | 70 ms | Apache-2.0 | First-class TFLite | same |
| **SSD-MobileNetV2** (FPN-lite) | 4.5M | 6.6 MB | 28.0 COCO | fast on NN API | Apache-2.0 | First-class TFLite | [TF Kaggle model](https://www.kaggle.com/models/tensorflow/ssd-mobilenet-v2) |
| **MobileNetV3-SSD** | ~3.5M | ~5 MB | ~25 mAP | fastest on CPU | Apache-2.0 | First-class TFLite | TF Object Detection API |

### 2.2 License gotcha
Ultralytics AGPL-3.0 = "if you ship a networked service that uses YOLOv8/v11/v12 weights, your entire service must be open-sourced under AGPL." For a campus app, simplest options:
1. **Buy Ultralytics Enterprise License** (pro tier is $29/seat/month; enterprise pricing is custom — discounts available for startups/hackathons via [ultralytics.com/startups](https://www.ultralytics.com/startups)). See [pricing](https://www.ultralytics.com/pricing) and [Siemens enterprise resale](https://www.siemens.com/en-us/products/ultralytics-enterprise-license/).
2. **Use RF-DETR (Apache-2.0)** — same transformer accuracy, no AGPL baggage. Note: still 30M params (~75 MB), best on phones with NN API.
3. **Use EfficientDet-Lite or SSD-MobileNetV2** (Apache-2.0) — smaller, slightly less accurate (25–30 mAP), but truly free.
4. **Use the original pre-AGPL YOLOv5 GPL-3.0** — still viral, but doesn't bind to the Ultralytics package (older Ultralytics versions pre-v8 are GPL-3.0).

### 2.3 Recommended
- **Primary (open source friendly):** **RF-DETR-Nano (Apache-2.0)** for desktop/server inference in Python; **EfficientDet-Lite0 (Apache-2.0)** for on-device TFLite in Unity.
- **Higher-accuracy option if you can afford the AGPL/Enterprise:** YOLOv11n (Ultralytics) or YOLOv12n, exported to CoreML iOS / TFLite Android.
- **Lighter fallback:** SSD-MobileNetV2 FPN-lite (TFLite, Apache-2.0) — proven, batteries-included with TF Model Maker.

---

## 3. Monocular Depth Estimation & Free-Space / Walkable-Path

This is the **most important** category for a free-space-vs-obstacle approach: bounding boxes miss hanging obstacles, glass doors, downward steps. Depth → "is this pixel < 0.3 m ahead" → "free space?" beats any detector.

### 3.1 Comparison table

| Model | Params | Size | Indoor/zero-shot quality | License | Mobile export | Source |
|---|---|---|---|---|---|---|
| **MiDaS v2.1 Small** | ~21M | 70 MB (ONNX) | good general depth | MIT | ONNX (multiple) + TFLite (community) | [julienkay/sentis-MiDaS (Unity-ready ONNX)](https://huggingface.co/julienkay/sentis-MiDaS), [Midas - Unity Sentis package](https://github.com/julienkay/com.doji.midas) |
| **MiDaS v3.1** (Intel-ISL) | up to 345M (large), small=24M | small=80 MB ONNX | best zero-shot | Apache-2.0 (paper) / MIT (weights) | ONNX, TFLite possible | [github.com/isl-org/MiDaS](https://github.com/isl-org/MiDaS) |
| **Depth Anything V1 Small** | 24.8M | 99 MB (FP32) / 25 MB (INT8) | strong zero-shot, sharper than MiDaS | Apache-2.0 (code) + OpenRAIL for some weights | ONNX, TFLite possible | [github.com/DepthAnything/Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2), [apple/coreml-depth-anything-small](https://huggingface.co/apple/coreml-depth-anything-small) |
| **Depth Anything V2 Small** | 24.8M | 99 MB (FP32) / 25 MB (INT8) | better than V1, fastest of the family | Apache-2.0 (code) / OpenRAIL for checkpoints | ONNX, CoreML, **Apple CoreML ready** ✅ | [github.com/DepthAnything/Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2), [apple/coreml-depth-anything-v2-small](https://huggingface.co/apple/coreml-depth-anything-v2-small), [depth-anything/Depth-Anything-V2-Small-hf](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf) |
| **ZoeDepth** (Intel-ISL) | ~340M (DPT-Large backbone) | ~1.3 GB | metric NYU/KITTI | MIT | ONNX | [Heliosoph/zoedepth-nyu-kitti-onnx](https://huggingface.co/Heliosoph/zoedepth-nyu-kitti-onnx) |
| **FastDepth** (NVIDIA, 2019) | ~1.5M | ~6 MB | NYU indoor depth | MIT | TFLite, ONNX | [github.com/dwofk/fast-depth](https://github.com/dwofk/fast-depth), [SeeedStudio reCamera INT8](https://wiki.seeedstudio.com/recamera_deploy_monocular_depth/) |
| **SC_Depth** (2024) | small variant | small | zero-shot, general | MIT | ONNX | [ibaiGorordo/ONNX-SCDepth](https://github.com/ibaiGorordo/ONNX-SCDepth-Monocular-Depth-Estimation) |
| **SegFormer-B0** (semantic seg) | 3.7M | ~13 MB | ADE20K mIoU 35.6 / Cityscapes 76.4 | Apache-2.0 | TFLite, ONNX | [nvidia/segformer-b0-finetuned-ade-512-512](https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512), [NVlabs/SegFormer](https://github.com/NVlabs/SegFormer) |
| **DeepLabV3+ MobileNetV2** | 5.8M | 9.4 MB | Cityscapes mIoU 75.2 | Apache-2.0 | TFLite | TF Model Maker |
| **BiSeNetV2 / PP-LiteSeg** | < 1M | 2–4 MB | Cityscapes mIoU 72–78 | Apache-2.0 | TFLite | PaddleSeg, [github.com/PaddlePaddle/PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg) |
| **MediaPipe Selfie Segmentation** | < 100K | <1 MB | binary person | Apache-2.0 | TFLite (official) | [MediaPipe Image Segmenter](https://developers.google.com/edge/mediapipe/solutions/vision/image_segmenter) |

### 3.2 Stair / door / escalator specific models
There is no off-the-shelf "stair+door+escalator" model in the major HF libraries, but a few good sources:
- **THUD++ Indoor Scene Dataset** (6,000+ indoor trajectories, plus depth + door+stair labels) — fine-tune YOLOv8/EfficientDet on it: [arXiv 2412.08096](https://arxiv.org/html/2412.08096v1).
- **"Smart Indoor Navigation Device for Blind People with Obstacle and Staircase Detection"** (Kh. et al., Semantic Scholar) uses geometric door-frame model + staircase detector on mobile.
- **Custom recipe:** fine-tune a YOLOv11n-seg / SegFormer-B0 on a school-internal dataset + **CODA (corner+object detection for autonomous driving)** to add small obstacle classes.

### 3.3 Recommended
- **Primary:** **Depth Anything V2 Small** (Apache-2.0) — strongest zero-shot, Apple already provides CoreML packages for iOS, and ONNX is one command on Android. Use it to compute a per-pixel distance, then threshold (<0.5 m = "obstacle", <0.2 m = "stop"). [Apple CoreML package](https://huggingface.co/apple/coreml-depth-anything-v2-small).
- **Unity-side:** MiDaS v2.1 small via the [julienkay/com.doji.midas](https://github.com/julienkay/com.doji.midas) Unity Sentis package — drop-in ONNX, tested in Unity Inference Engine.
- **Lighter fallback:** **FastDepth** (1.5M params, 6 MB) for devices that can't handle DPT-based depth.
- **Free-space segmentation:** **SegFormer-B0** (Apache-2.0) for semantic classes (floor, wall, door, person). Run in parallel with depth for a richer "where can I walk" map.

---

## 4. Scene Understanding / VLM (on-demand "what's around me?")

VLMs are too slow for 30 fps, so the design is: **run 1 frame every 2-3 s on a captured scene, send to server-side VLM, output text → TTS**. For the phone, all of these are too large; the design calls for **server-side**.

### 4.1 Comparison table

| Model | Params | VRAM (4-bit) | License | Notes | Source |
|---|---|---|---|---|---|
| **Qwen2.5-VL-3B** | 3B | ~2.5 GB | Apache-2.0 | strong OCR + grounding | [huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) |
| **Qwen2.5-VL-7B** | 7B | ~5 GB | Apache-2.0 | best for scene captioning, can run on L4 GPU | [huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct), [Qwen2.5-VL blog](https://qwenlm.github.io/blog/qwen2.5-vl/) |
| **Qwen3-VL-8B** (Nov 2025) | 8B | ~6 GB | Apache-2.0 | Sharper, deeper thought, broader action; current SOTA among open | [github.com/qwenlm/qwen3-vl](https://github.com/qwenlm/qwen3-vl), [huggingface.co/collections/Qwen/qwen3-vl](https://huggingface.co/collections/Qwen/qwen3-vl) |
| **InternVL2.5-4B** | 4B | ~3 GB | MIT (most checkpoints) | strong multimodal | [internvl.github.io/blog/2024-12-05-InternVL-2.5](https://internvl.github.io/blog/2024-12-05-InternVL-2.5/), [OpenGVLab/InternVL2.5-4B](https://www.modelscope.cn/models/OpenGVLab/InternVL2_5-4B) |
| **InternVL3** (2025) | 1B / 2B / 4B / 8B | 1–6 GB | MIT | 2025 release | [github.com/opengvlab/internvl](https://github.com/opengvlab/internvl) |
| **SmolVLM-256M-Instruct** | 256M | <1 GB | Apache-2.0 | tiny, on-device capable (not for full captioning) | [huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct) |
| **SmolVLM-500M-Instruct** | 500M | <1.5 GB | Apache-2.0 | tiny, on-device capable | [SmolVLM blog](https://huggingface.co/blog/smolervlm) |
| **SmolVLM2-2.2B** | 2.2B | ~3 GB | Apache-2.0 | improved video + docs | [HuggingFaceTB/SmolVLM2-2.2B-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct) |
| **Moondream 2** | 1.86B | ~1.2 GB | Apache-2.0 (HF) / Moondream License 1.0 | very small, fast | [moondream.ai](https://moondream.ai/), [github.com/m87-labs/moondream](https://github.com/m87-labs/moondream) |
| **Moondream 0.5B** | 500M | <1 GB | Apache-2.0 | distillation target for edge | [moondream.ai/models](https://moondream.ai/models) |
| **BLIP-2 OPT-2.7B / Flan-T5-XL** | 2.7B / 3B | 5–6 GB | MIT | image captioning classic | [github.com/salesforce/LAVIS](https://github.com/salesforce/LAVIS) |
| **Florence-2 base / large** | 0.27B / 0.77B | 1–2 GB | MIT | captioning + dense region | [microsoft/Florence-2-base-ft](https://huggingface.co/microsoft/Florence-2-base-ft) |

### 4.2 Recommended
- **Primary (server-side scene description):** **Qwen2.5-VL-7B-Instruct** (Apache-2.0) — the de-facto standard for "what's in this room?" with low VRAM footprint.
- **Higher-accuracy option (newer):** **Qwen3-VL-8B** — current SOTA among open VLMs in 2025.
- **Lighter fallback (server-side):** **InternVL2.5-4B** (MIT) or **Moondream 2** (Apache-2.0, smaller) for low-VRAM servers.
- **On-device tiny captioning (rare queries):** **SmolVLM-256M-Instruct** (Apache-2.0) via ONNX / TFLite — fits < 1 GB RAM, but lower accuracy.

---

## 5. LLM for Direction Generation (server-side)

### 5.1 Comparison table

| Model | Params | VRAM (Q4) | License | Notes | Source |
|---|---|---|---|---|---|
| **Qwen3-1.7B / 4B / 8B / 14B / 32B** | 1.7B–32B | 1.5–22 GB | Apache-2.0 (most) | best open instruction following, multilingual, JSON-mode | [huggingface.co/Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B), [github.com/qwenLM/qwen3](https://github.com/qwenLM/qwen3) |
| **Llama-3.1-8B-Instruct** | 8B | 5 GB | Llama 3.1 Community License (commercial OK with conditions) | strong IFEval (92.1%), well known | [meta-llama/Meta-Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct) |
| **Llama-3.3-70B-Instruct** (or 3.3-8B distilled) | 70B / 8B | 40 GB / 5 GB | Llama 3.3 Community License | improved over 3.1 | meta-llama |
| **Phi-4-mini-instruct** | 3.8B | 2.5 GB | MIT | strong instruction following for size | [microsoft/Phi-4-mini-instruct](https://huggingface.co/microsoft/Phi-4-mini-instruct), [arXiv 2503.01743](https://arxiv.org/html/2503.01743v1) |
| **Gemma 3 1B / 4B / 12B / 27B** | 1B–27B | 1–15 GB (Q4) | Gemma Terms (commercial OK with restrictions) | compact, on-device capable 1B/4B | [huggingface.co/google/gemma-3-4b-it](https://huggingface.co/google/gemma-3-4b-it), [ollama.com/library/gemma3](https://ollama.com/library/gemma3) |
| **Mistral-7B-Instruct / Mixtral 8x7B** | 7B / 12.9B-active | 5/24 GB | Apache-2.0 | strong tool-use | [mistralai/Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) |
| **gpt-oss-20B** (OpenAI open release) | 20B | ~14 GB | Apache-2.0 | tool-use, function calling | [huggingface.co/openai/gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b) |
| **DeepSeek-V3 / R1 distilled** | 7B–70B | varies | MIT / DeepSeek License | reasoning good for routes | [huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B) |

### 5.2 Prompting strategy (structured JSON → spoken directions)
A clean prompt template for the backend. Tested pattern: feed the route engine's JSON, ask the LLM to emit *only* JSON, then re-format to natural language in a second pass (or with a smaller LLM). Example system prompt:

```text
You are a concise navigation assistant for a blind user.
Given the user's current position and a list of waypoints, output a short
spoken instruction (< 15 words) for the NEXT step. Do not include any
small talk, punctuation that TTS will mispronounce, or step numbers
beyond 1. Speak in second person present tense.

INPUT_JSON:
{ "user": "Lobby", "next_wp": "Turn right at fountain", "distance_m": 4.2,
  "hazards": ["stairs 3m ahead"], "urgency": "caution" }

OUTPUT: { "say": "Turn right at the fountain in 4 meters, caution: stairs 3 meters ahead." }
```

Use **Ollama** for the hackathon demo (one command: `ollama run qwen3:8b`), or **vLLM** for production. Both expose an OpenAI-compatible API. See [Qwen3 setup guide](https://medium.com/@techlatest.net/qwen3-8-27b-setup-guide-vllm-ollama-and-api-integration-step-by-step-0984857f5415).

### 5.3 Recommended
- **Primary:** **Qwen3-8B** (Apache-2.0) — best small instruction follower, JSON-mode, multilingual.
- **Higher-accuracy option (server):** **Qwen3-14B** or **DeepSeek-R1-Distill-Qwen-14B** for more nuanced route reasoning.
- **Lighter fallback (server or even laptop):** **Phi-4-mini-instruct** (MIT) at 3.8B params — small but very strong for short structured output.
- **CPU / mobile-LLM demo:** **Gemma 3 1B** or **Qwen3-1.7B** (works on CPU/edge for offline campus use).

---

## 6. On-Device Infrastructure in Unity

### 6.1 Comparison table

| Engine | Status (Aug 2026) | License | ONNX support | CoreML / NNAPI | Pros | Cons | Source |
|---|---|---|---|---|---|---|---|
| **Unity Sentis** (formerly Barracuda) | v2.6+ production-ready, free on all Unity plans | Bundled with Unity Editor (no extra cost for inference; commercial OK in apps) | Yes (.sentis = ONNX after import) | Some CoreML ops via ONNX path; GPU compute shaders; Vulkan/Metal | First-class Unity, no Python round-trip, runs in iOS/Android player | Not every op supported; some ONNX ops need workaround | [docs.unity3d.com/Packages/com.unity.ai.inference@latest](https://docs.unity3d.com/Packages/com.unity.ai.inference@latest/), [What's new in 1.1](https://docs.unity3d.com/Packages/com.unity.sentis@1.1/manual/whats-new.html) |
| **Unity Barracuda** | DEPRECATED, replaced by Sentis | — | — | — | — | — | — |
| **asus4/tf-lite-unity-sample** | Community maintained | Apache-2.0 | No — TFLite native | Yes via delegates | Direct TFLite on iOS/Android | More glue code | [github.com/asus4/tf-lite-unity-sample](https://github.com/asus4/tf-lite-unity-sample) |
| **MediaPipe Unity Plugin (homuler)** | Active, supports MediaPipe 0.10.22 | Apache-2.0 | n/a (uses MediaPipe C++) | Yes | Object detection, image segmentation, hands, pose | Slower than native Sentis | [github.com/homuler/MediapipeUnityPlugin](https://github.com/homuler/MediapipeUnityPlugin) |
| **ONNX Runtime Unity** (Microsoft) | Available | MIT | Yes | Yes | Full ORT | Heavier package, must add ops manually | [onnxruntime.ai](https://onnxruntime.ai/) |
| **LiteRT / TFLite** (Google) | v2.x, renamed from TFLite | Apache-2.0 | n/a | Yes (NNAPI, GPU, CoreML delegates) | Best for Android, TFLite models universal | Need a thin Unity wrapper (asus4 plugin) | [developers.google.com/edge/litert](https://developers.google.com/edge/litert), [TensorFlow Lite is now LiteRT](https://developers.googleblog.com/tensorflow-lite-is-now-litert/) |
| **Core ML (iOS only)** | Production | Apple | via coremltools | Yes | Optimal on Apple Neural Engine | iOS only | [apple.com/coreml](https://developer.apple.com/machine-learning/core-ml/) |

### 6.2 Recommended architecture
- **Unity 6 + Sentis 2.6+ for on-device audio + depth + segmentation** (drop in `.sentis` ONNX files). Works on iOS Metal and Android Vulkan.
- **Python backend** handles the LLM and VLM (any GPU box) and returns JSON → TTS.
- **Optional bridge:** if a model only exports cleanly to TFLite, use the [asus4/tf-lite-unity-sample](https://github.com/asus4/tf-lite-unity-sample) and run it side-by-side with Sentis.
- **Camera stream + YOLO inside Unity?** Yes — export YOLOv8n/v11n/v12n or EfficientDet-Lite to ONNX, import to Sentis, run 30 fps inference on a 256×256 input. Apple [CoreML depth-anything package](https://huggingface.co/apple/coreml-depth-anything-v2-small) is a perfect template for iOS export.
- **For MediaPipe tasks (eg. on-device face, hands, body pose, selfie segmentation):** use the [homuler MediaPipeUnityPlugin](https://github.com/homuler/MediapipeUnityPlugin).

### 6.3 License summary
- **Unity Sentis** is included with all Unity plans for local AI inference, including commercial. No extra fee. See [Unity discussion](https://discussions.unity.com/t/all-unity-plans-will-get-unity-sentis/292986).
- **ONNX models** themselves carry the upstream license (most are MIT/Apache-2.0).
- **MediaPipe Unity Plugin** = Apache-2.0, safe to ship.

---

## 7. Final Recommended Stack

| Component | Primary pick | Lighter fallback | Higher-accuracy option | License of primary |
|---|---|---|---|---|
| **Audio classifier** | **YAMNet (TFLite, 3.8 MB)** | (same, already smallest) | **EfficientAT `mn10_as`** | Apache-2.0 |
| **Obstacle detector** | **RF-DETR-Nano (Python, Apache-2.0)** for server; **EfficientDet-Lite0 (TFLite, Apache-2.0)** for Unity | **SSD-MobileNetV2 (TFLite)** | **YOLOv11n / v12n** (if you have Ultralytics Enterprise or are OK with AGPL) | Apache-2.0 |
| **Depth / free-space** | **Depth Anything V2 Small (ONNX/CoreML, 25 MB INT8)** + **SegFormer-B0** for floor/door classes | **FastDepth** (1.5M params, 6 MB) | **ZoeDepth** (large, server) | Apache-2.0 |
| **VLM (scene captioning)** | **Qwen2.5-VL-7B-Instruct** (server, Apache-2.0) | **InternVL2.5-4B** or **Moondream 2** | **Qwen3-VL-8B** | Apache-2.0 |
| **LLM (direction gen)** | **Qwen3-8B** served via Ollama or vLLM (Apache-2.0) | **Phi-4-mini-instruct** (3.8B, MIT) or **Gemma 3 4B** | **Qwen3-14B / DeepSeek-R1-Distill-14B** | Apache-2.0 |
| **On-device runtime** | **Unity Sentis 2.6+** for all ONNX models | (same) | — | Free with Unity |

### 7.1 Hackathon 24-hour "go" plan
1. **On the phone (Unity Sentis):** YAMNet (TFLite) + EfficientDet-Lite0 (TFLite) + Depth Anything V2 Small (ONNX).
2. **Python server (any laptop with 16 GB RAM):** Ollama + Qwen3-4B → 8B for LLM, Qwen2.5-VL-7B for occasional scene description.
3. **Bridge:** Unity HTTP POSTs `{position, route_step, hazards, frame_id}`; Python replies with `{speak: "...", emoji: "🚶"}`; Unity calls native TTS.
4. **Fine-tune YAMNet** on a 1-2 hour recording of campus-specific sounds (microwave, elevator chime, sliding doors) using the [TFLite codelab](https://developers.google.com/codelabs/tflite-audio-classification-custom-model-android) pipeline.

### 7.2 If you must avoid AGPL entirely
- **Replace YOLOv8/v11/v12** with **RF-DETR-Nano (Apache-2.0)** + **EfficientDet-Lite0 (Apache-2.0)**.
- Use **SegFormer-B0** for segmentation instead of YOLOv8-seg.
- Use **Depth Anything V2** (Apache-2.0) for free-space.

This stack is 100 % Apache-2.0 / MIT / OpenRAIL — safe to ship a closed-source hackathon prototype without owing royalties.

---

## 8. Key URLs (one-stop reference)

### Audio
- YAMNet: https://github.com/tensorflow/models/blob/master/research/audioset/yamnet/params.py · https://www.tensorflow.org/hub/tutorials/yamnet
- AST: https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593 · https://github.com/YuanGongND/ast
- BEATs: https://github.com/microsoft/unilm/blob/master/beats/README.md
- PaSST: https://github.com/kkoutini/PaSST
- PANNs: https://github.com/qiuqiangkong/audioset_tagging_cnn
- EfficientAT: https://github.com/fschmid56/EfficientAT
- MediaPipe audio classifier: https://developers.google.com/edge/mediapipe/solutions/audio/audio_classifier

### Detection
- YOLOv8/v11/v12: https://github.com/ultralytics/ultralytics · https://docs.ultralytics.com/models/yolo11 · https://yolov12.com/
- YOLOv10: https://github.com/THU-MIG/yolov10
- RF-DETR: https://github.com/roboflow/rf-detr · https://rfdetr.roboflow.com/
- RT-DETR: https://github.com/lyuwenyu/RT-DETR · https://docs.ultralytics.com/models/rtdetr
- EfficientDet-Lite: https://blog.tensorflow.org/2021/06/easier-object-detection-on-mobile-with-tf-lite.html · https://developers.google.com/edge/litert/libraries/modify/object_detection
- SSD-MobileNetV2: https://www.kaggle.com/models/tensorflow/ssd-mobilenet-v2
- YOLO-NAS: https://github.com/Deci-AI/super-gradients/blob/master/YOLONAS.md

### Depth / Segmentation
- MiDaS: https://github.com/isl-org/MiDaS · https://github.com/julienkay/com.doji.midas
- Depth Anything V2: https://github.com/DepthAnything/Depth-Anything-V2 · https://huggingface.co/apple/coreml-depth-anything-v2-small
- ZoeDepth ONNX: https://huggingface.co/Heliosoph/zoedepth-nyu-kitti-onnx
- FastDepth: https://wiki.seeedstudio.com/recamera_deploy_monocular_depth/
- SegFormer: https://github.com/NVlabs/SegFormer · https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512
- MediaPipe Image Segmenter: https://developers.google.com/edge/mediapipe/solutions/vision/image_segmenter

### VLMs
- Qwen2.5-VL: https://huggingface.co/collections/Qwen/qwen25-vl · https://qwenlm.github.io/blog/qwen2.5-vl/
- Qwen3-VL: https://github.com/qwenlm/qwen3-vl · https://huggingface.co/collections/Qwen/qwen3-vl
- InternVL: https://github.com/opengvlab/internvl · https://internvl.github.io/blog/2024-12-05-InternVL-2.5/
- SmolVLM: https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct · https://huggingface.co/blog/smolervlm
- Moondream: https://moondream.ai/ · https://github.com/m87-labs/moondream
- BLIP-2: https://github.com/salesforce/LAVIS
- Florence-2: https://huggingface.co/microsoft/Florence-2-base-ft

### LLMs
- Qwen3: https://huggingface.co/Qwen/Qwen3-8B · https://github.com/qwenLM/qwen3
- Llama 3.1 / 3.3: https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct
- Phi-4-mini: https://huggingface.co/microsoft/Phi-4-mini-instruct · https://arxiv.org/html/2503.01743v1
- Gemma 3: https://huggingface.co/google/gemma-3-4b-it · https://ollama.com/library/gemma3
- DeepSeek-R1 distill: https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
- gpt-oss: https://huggingface.co/openai/gpt-oss-20b

### Unity & on-device
- Unity Sentis: https://docs.unity3d.com/Packages/com.unity.ai.inference@latest/ · https://docs.unity3d.com/Packages/com.unity.sentis@1.1/manual/whats-new.html
- MiDaS-Unity Sentis: https://github.com/julienkay/com.doji.midas · https://huggingface.co/julienkay/sentis-MiDaS
- TFLite Unity sample: https://github.com/asus4/tf-lite-unity-sample
- MediaPipe Unity Plugin: https://github.com/homuler/MediapipeUnityPlugin
- LiteRT: https://developers.google.com/edge/litert · https://developers.googleblog.com/tensorflow-lite-is-now-litert/
- Ultralytics licensing: https://www.ultralytics.com/license · https://www.ultralytics.com/pricing · https://www.ultralytics.com/startups
- ONNX Runtime: https://onnxruntime.ai/
- CoreML depth-anything: https://huggingface.co/apple/coreml-depth-anything-v2-small

---

*Compiled 2026-08-31 from official GitHub repos, Hugging Face model cards, Ultralytics docs, Roboflow blog, Roboflow model comparisons, and Google AI Edge documentation. Always re-verify license terms before shipping — especially Ultralytics AGPL-3.0 and Moondream License 1.0.*
