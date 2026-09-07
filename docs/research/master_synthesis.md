# Indoor Campus Navigation for Blind/Low-Vision Students — Master Research Synthesis

**Compiled:** 2026-08-31
**Purpose:** Complete solution-space research for a hackathon project using marker-based AR (Vuforia) + Unity frontend + Python backend + on-device AI (audio classification, obstacle detection) + LLM-generated spoken directions.

**Companion documents (full detail):**
- `positioning_solutions.md` — 12 indoor positioning approaches, 18 deployed systems, 150+ sources
- `models_catalog.md` — ~30 open-source models with licenses and sizes
- `datasets_catalog.md` — 60+ datasets with licenses and a minimal fine-tuning plan

---

## Executive Summary

Your current architecture (Vuforia Image Targets + spatial audio/haptics + on-device YAMNet/YOLO + server-side LLM) is validated by the research as a defensible, demo-optimal choice — it matches the #1 recommended approach for hackathon-scale indoor blind navigation. Three findings materially change your build plan:

1. **YOLOv8/v11/v12 is AGPL-3.0** — shipping it in a closed app legally requires an Ultralytics Enterprise license. A 100% Apache-2.0 alternative stack exists (RF-DETR-Nano, EfficientDet-Lite0) at comparable accuracy.
2. **Unity Sentis runs ONNX models natively inside Unity** — YAMNet and your detector can run in-process on the phone with no Python round-trip for the per-frame path, simplifying your architecture.
3. **Monocular depth (Depth Anything V2 Small) arguably matters more than bounding boxes** for "where can I walk" — boxes miss glass doors, hanging obstacles, and downward steps; depth thresholds don't.

The research also flags one genuine design tension you should be ready to defend: **markers require the user to hold the camera up**, while the strongest evidence-backed blind-navigation systems (NavCog3, Manduchi's UCSC apps) optimize for phone-in-pocket operation. Your answer for a hackathon: camera-up is acceptable for a demo, and the marker system is swappable for a BLE+PDR hybrid in phase 2.

---

## Part 1: The Complete Positioning Solution Space

All twelve viable approaches, condensed. Full profiles with accuracy figures, costs, citations, and failure modes are in `positioning_solutions.md`.

| # | Approach | Accuracy | Infra cost | Phone-native? | Adapts to change? | Open source? |
|---|---|---|---|---|---|---|
| 1 | Fiducial markers (ArUco/AprilTag/Vuforia) | 1–10 cm @ 1 m | ~$0 (print) | Yes | No — re-print | ArUco GPLv3; AprilTag BSD; Vuforia proprietary |
| 2 | BLE beacons | 1–3 m RSSI; 0.5–1.5 m AoA | $500–3k/floor | Yes | Re-survey | Formats open; SDKs closed |
| 3 | UWB | 10–30 cm | $10k+ campus | iPhone 11+/some Android | With anchors | Mostly proprietary |
| 4 | Wi-Fi RTT / RSSI | 1–2 m RTT; 3–5 m RSSI | $ if reusing APs | Android-only (RTT) | Re-survey | Open |
| 5 | ARCore/ARKit VIO + VPS | 1–3 cm relative; ~5 m global | $0 (VIO) | Yes | Yes (VPS) | ARCore Apache-2.0 |
| 6 | SLAM (ORB-SLAM3, semantic) | 0.05–0.3 m relative | $0 | Flagships only | Re-map needed | ORB-SLAM3 GPLv3 |
| 7 | Phone LiDAR | <1 mm static; 5 m range | $0 (Pro devices only) | iPhone/iPad Pro only | Re-scan | Apple proprietary |
| 8 | Magnetic field | 2–4 m | $0 (survey) | Yes | Re-survey | Mostly closed (IndoorAtlas) |
| 9 | Pedestrian dead reckoning | ~2–5% of distance drift | $0 | Yes | n/a (relative) | Open |
| 10 | RFID/NFC/Talking Signs/NaviLens | Point (NFC) to 12 m (NaviLens) | $ tags | Yes (NFC/NaviLens) | Re-tag | NaviLens SDK open |
| 11 | Hybrid sensor fusion | 0.5–2 m | Varies | Yes | Most robust | Many open |
| 12 | Image retrieval / VPS | 1–5 m | $ to build image DB | Yes | Re-index | Research open |

### What the deployed systems actually use

- **NavCog3 (CMU):** BLE fingerprinting + PDR particle filter → ~1–2 m at mall/airport scale. The closest research ancestor to your project.
- **Microsoft Soundscape** (open-sourced 2023, MIT license): 3D spatial audio, GPS + iBeacon. No indoor positioning of its own — the audio-cue design patterns are directly reusable.
- **NaviLens:** purpose-built color codes for blind users — 12 m read range, 160° FOV, reads while moving. Deployed in NY MTA, Cal Poly. Free SDK + tag generator.
- **Wayfindr:** open standard (ITU-recognized) for BLE audio wayfinding — its guidance-on-trigger patterns are worth reading before designing your audio UX.
- **Aira / Be My Eyes:** remote human agent model — validates demand, not a technical competitor.
- **Manduchi lab (UCSC):** phone-in-pocket inertial wayfinding — the key counterpoint to camera-up designs.

### Verdict on your Vuforia choice

Marker-based AR + VIO bridging is the **#1 hackathon recommendation** from the research: printable, free, real-time 6-DoF pose, visually compelling, robust in lit corridors. Validated by published systems (ceiling-marker campus navigation, SINS_AR). Two honest caveats for judges: (a) ArUco/AprilTag are the open-source alternatives if Vuforia licensing ever matters — the pipeline is the same; (b) NaviLens-style tags are strictly better *for blind users specifically* (12 m, 160° FOV, motion-tolerant) if you ever want a second marker tier.

---

## Part 2: The Model Stack

### ⚠️ The one decision that matters most: detector licensing

**YOLOv8n/v11n/v12n are AGPL-3.0.** If you ship them in a closed app or networked service, the AGPL obligates you to open-source the whole service — or buy Ultralytics Enterprise. For a hackathon this is survivable (nobody enforces AGPL at a demo), but say it knowingly or avoid it.

**100% permissive recommended stack** (all Apache-2.0/MIT):

| Component | Primary | Size | Fallback | Upgrade |
|---|---|---|---|---|
| Audio classifier | **YAMNet** (TFLite) | 3.8 MB | — (already smallest) | EfficientAT mn10_as |
| Obstacle detector | **RF-DETR-Nano** (server) / **EfficientDet-Lite0** (on-device) | 75 MB / 4.4 MB | SSD-MobileNetV2 (6.6 MB) | YOLOv11n (if licensed) |
| Depth / free-space | **Depth Anything V2 Small** (ONNX/CoreML, INT8 ~25 MB) + **SegFormer-B0** | 25 MB | FastDepth (6 MB) | ZoeDepth (server) |
| VLM (scene Q&A, server) | **Qwen2.5-VL-7B-Instruct** | ~5 GB VRAM | InternVL2.5-4B / Moondream 2 | Qwen3-VL-8B |
| LLM (directions, server) | **Qwen3-8B** via Ollama/vLLM | ~5 GB VRAM | Phi-4-mini (3.8B, MIT) | Qwen3-14B |
| On-device runtime | **Unity Sentis 2.6+** | — | asus4/tf-lite-unity-sample | — |

### Architecture simplification you should seriously consider

Unity Sentis (free on all Unity plans, Barracuda's successor) imports ONNX directly. That means **YAMNet + EfficientDet-Lite0 + Depth Anything V2 Small can all run inside the Unity app itself**. Your Python backend then only handles what genuinely needs server compute: the LLM/VLM for directions and scene queries. This kills the per-frame network round-trip entirely — better latency, better demo reliability, works offline for the perception layer.

### Why depth beats bounding boxes for your core promise

Your core gap statement is "continuous position awareness relative to a destination" — but the obstacle channel is about "can I walk forward." A bounding box tells you a person is ahead; a depth map tells you the nearest obstruction in your walking corridor is 0.4 m away, whether it's a person, a glass door, or an unmarked step-down. **Depth Anything V2 Small is zero-shot** — no training needed, 25 MB quantized, runs in Sentis. The detector (for "person ahead, moving toward you" semantics) becomes the complement, not the primary.

---

## Part 3: Datasets & Fine-Tuning Plan

Full catalog (60+ datasets, licenses, URLs) in `datasets_catalog.md`. The minimal plan:

### Tier 1 — Must-have (~small, fast, high signal)

**YAMNet audio head** — fine-tune on:
- **UrbanSound8K** (8,732 clips / 10 classes; CC BY-NC — demo OK) → sirens, crowds
- **ESC-50** (2,000 / 50 classes; CC BY-NC) → footsteps, doors, clocks
- Emergency siren corpus (Kaggle/Nature 2022) → high-priority alert class
- Result: custom head for ~15–20 campus sounds (footsteps, door slam, alarm, elevator ding, crowd babble). Hours, not days, to train.

**Detector** — COCO-pretrained → fine-tune on:
- **Roboflow Indoor Objects Detection** (1,349 imgs / 10 classes incl. door)
- **Mendeley Stair dataset** (2,670 imgs)
- **CrowdHuman** subset (crowded-hallway person detection — your core scenario)
- **Roboflow indoor-navigation** (1,115 hallway/door/elevator imgs)

### Tier 2 — Should-have

- **NYU Depth V2** (indoor, 5 GB) — only if fine-tuning depth; Depth Anything V2 zero-shot may suffice
- **R2R / RxR / Map2Seq** instruction corpora — not for training; mine as few-shot examples for your LLM direction prompt. Map2Seq's landmark-based style is the closest match to your task.
- **VizWiz-VQA** — what blind users actually ask; informs your UX copy
- **Unity Perception + PeopleSansPeople** — your unfair advantage: render synthetic crowded-hallway training images with domain randomization, since you already live in Unity

### Tier 3 — Nice-to-have

OSM indoor mapping of your actual demo building (feeds a route graph to the LLM), 7-Scenes/InLoc for image-retrieval localization experiments, Kaggle Indoor Location for Wi-Fi fingerprint backup.

**Total: ~50 GB of data, under a day of training on one A100** — demonstrably hackathon-feasible.

---

## Part 4: How It All Fits — Integrated Architecture

```
┌─────────────────────── UNITY APP (phone) ───────────────────────┐
│                                                                  │
│  Vuforia Image Targets ──→ 6-DoF pose at markers                 │
│  (phase 2: BLE+PDR hybrid for phone-in-pocket)                   │
│           │                                                      │
│  Unity Sentis (all on-device, no network):                       │
│   • YAMNet (audio → campus sound events)                         │
│   • Depth Anything V2 Small (free-space mask, obstacle range)    │
│   • EfficientDet-Lite0 (person/door/stair semantics)             │
│           │                                                      │
│  Geometric layer (no AI): bearing→haptic pattern,                │
│  proximity→tone pitch, event→voice trigger                       │
│                                                                  │
│  Low-vision mode: high-contrast outlines, landmark labels,       │
│  background dimming over the Vuforia camera feed                 │
└──────────────┬───────────────────────────────────────────────────┘
               │ WebSocket / UnityWebRequest — small JSON only
               │ {marker_id, pose, sound_events, obstacles, route_step}
┌──────────────▼──────────── PYTHON BACKEND ───────────────────────┐
│  Route graph (OSM indoor / IndoorGML) + position fusion          │
│  LLM (Qwen3-8B via Ollama, or Claude/GPT API):                   │
│    structured JSON → <15-word spoken instruction                  │
│  VLM (Qwen2.5-VL-7B, on-demand): "what's around me?" queries     │
└──────────────┬───────────────────────────────────────────────────┘
               │ {say: "Turn right at the fountain in 4 meters",
               │  urgency: "caution"}
┌──────────────▼─────── OUTPUT CHANNELS ───────────────────────────┐
│  Voice: event-triggered only (waypoint, hazard, off-route)       │
│  Haptics: continuous, geometric (bearing + proximity)            │
│  Spatial audio: beacon toward next waypoint                      │
└───────────────────────────────────────────────────────────────────┘
```

Privacy posture (wiretap/FERPA concern): audio classification and camera inference happen **on-device**; only derived labels and marker IDs cross the network. No raw audio or imagery leaves the phone. This is a genuinely strong answer to the legal questions you researched earlier.

---

## Part 5: Known Risks & Honest Caveats

1. **Camera-up requirement.** Marker tracking needs the phone held up and pointed — atypical for blind users who often pocket the phone or use a cane hand. Mitigations: demo framing (a navigation session, not all-day wear), audio-first UX so the screen presence is incidental, and the phase-2 BLE+PDR story for phone-in-pocket.
2. **Marker occlusion in crowds.** Your exact target scenario (crowded hallway) is the marker system's weakest condition — bodies block the camera view. Mitigation: dense marker placement at decision points + VIO bridging + PDR so brief losses don't kill tracking.
3. **Vuforia licensing** for anything beyond the demo (free Basic tier is limited; Classic is $499/app). ArUco/AprilTag are the open escape hatch.
4. **LLM latency and hallucination.** Directions must come from your route graph, not model imagination — the LLM verbalizes a computed instruction, it does not decide the route. Keep the <15-word constraint and validate output against the graph.
5. **Trust/adoption.** You identified this earlier and the research agrees it's the harder problem. Nothing in this stack fixes it — only user testing with actual blind/low-vision students can. If you can get even 2–3 real users to try the demo, that's worth more to judges than any model choice.

---

## Recommended Next Actions

1. **Prototype the perception trio in Sentis** (YAMNet + Depth Anything V2 + EfficientDet-Lite0) — all three are downloadable today and run inference in Unity without training.
2. **Fine-tune YAMNet** on UrbanSound8K + ESC-50 + your own 1–2 h of campus recordings (elevator chimes, your building's specific alarm).
3. **Build the route graph** for your demo building (OSM indoor or simple JSON waypoints) and wire the LLM direction prompt with Map2Seq-style few-shot examples.
4. **Order/decide on detector licensing** — RF-DETR-Nano/EfficientDet-Lite0 (Apache) vs YOLO (AGPL) — before you write training code.
5. **Dry-run the privacy story** — on-device inference, derived signals only — since you've already identified it as a legal differentiator.
