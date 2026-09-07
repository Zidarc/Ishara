# Indoor Positioning & Navigation Technologies for Blind/Low-Vision Users
*A comprehensive map of every viable approach for a hackathon campus navigation assistant.*

Compiled August 2026. Each approach is assessed on: (1) how it works, (2) accuracy / range, (3) hardware + installation cost, (4) consumer smartphone compatibility, (5) research & deployed product evidence with URLs, (6) documented failure modes, and (7) open-source / license status.

---

## 1. Fiducial Marker Systems (ArUco, AprilTag, Vuforia Image Targets, STag)

**One-line summary:** A printed square pattern (with known size) is detected by a phone camera; pose (x,y,z,roll,pitch,yaw) is computed from corner geometry, giving absolute position when a marker is in view.

**How it works:** Each marker encodes a unique ID in a binary grid inside a black border. The four outer corners are detected to sub-pixel accuracy; given the marker's printed physical size, a pinhole-camera model yields 6-DoF pose relative to the camera. Continuous positioning is achieved by chaining poses via visual-inertial odometry (VIO) between markers, or by mapping every "fiducial the user can see" to a known map coordinate.

**Accuracy / Range:** Sub-degree rotation and 1–10 cm translation at <1 m, degrading with distance. AprilTag and ArUco work at 5–10 m in good light; very large AprilTags can be detected at 25+ m ([laserscanning-europe.com](https://www.laserscanning-europe.com/en/what-apriltags-what-apriltag-size-should-use)). Comparative study FMAC found ArUco has 100% detection and best pose accuracy between 0.5–0.9 m; STag detection drops to ~84% at distance ([arxiv.org/html/2601.07723v1](https://arxiv.org/html/2601.07723v1)). Detectability breaks down around 2.5 m on small markers with low-resolution webcams ([dl.acm.org/doi/10.1145/3807246.3807278](https://dl.acm.org/doi/10.1145/3807246.3807278)).

**Hardware + Install Cost:** Printers, paper, and tape. Essentially zero marginal cost per marker; one-time design labor. Vuforia additionally needs a database/Cloud account; U.S. pricing is a one-time $499/app "Classic" license or a free Basic plan for limited features ([developer.vuforia.com](https://developer.vuforia.com/library/vuforia-engine/FAQ/pricing-and-licensing-options/), [ptc.com](https://www.ptc.com/en/products/vuforia/vuforia-engine/pricing)).

**Smartphone compatibility:** ArUco, AprilTag, STag all run on consumer iOS/Android via OpenCV or NDK. Vuforia Engine is officially supported on iOS, Android, and Unity ([developer.vuforia.com Unity guide](https://developer.vuforia.com/library/vuforia-engine/getting-started/development-environments/getting-started-vuforia-engine-unity/)).

**Research / Product Evidence:**
- AR-based blind navigation with Vuforia Image Targets is the current hackathon design and is used in NaviLens-style deployments.
- FMAC: a Fair Fiducial Marker Accuracy Comparison Software ([arxiv.org/html/2601.07723v1](https://arxiv.org/html/2601.07723v1))
- "Experimental Comparison of Fiducial Markers for Pose Estimation" — Kalaitzakis et al., 2020 ([researchgate.net](https://www.researchgate.net/publication/347154054_Experimental_Comparison_of_Fiducial_Markers_for_Pose_Estimation))
- "Determining and Improving the Localization Accuracy of AprilTag" — Kallwies & Forkel ([semanticscholar.org](https://www.semanticscholar.org/paper/Determining-and-Improving-the-Localization-Accuracy-Kallwies-Forkel/190a6317ebfbe2c6f29b7684f68a5b5a2104c02c))
- "Investigation of ArUco Marker Placement for Planar Indoor …" 2025 ([arxiv.org/html/2509.17345v1](https://arxiv.org/html/2509.17345v1))
- OpenCV ArUco tutorial ([docs.opencv.org](https://docs.opencv.org/4.13.0/d5/dae/tutorial_aruco_detection.html))
- AprilTag 3 reference implementation, BSD license, used by NASA ([github.com/nasa/AprilNav](https://github.com/nasa/AprilNav/blob/master/AprilTags/Tag36h11.h))
- STag, "not actively maintained" fork ([github.com/bbenligiray/stag](https://github.com/bbenligiray/stag))

**Failure modes:** Lighting (glare, low light, shadows), occlusion (people walking in front of wall signs), distance/size tradeoff, viewing-angle limit (~70° off-axis for ArUco), reflective surfaces, vandalism, requires a non-zero camera-uptime (drains battery, useless if phone is in pocket — the very issue Manduchi's UCSC "phone-in-pocket" inertial app is trying to solve; see [dl.acm.org/doi/10.1145/3696005](https://dl.acm.org/doi/10.1145/3696005)).

**Open-source / License:**
- ArUco: BSD originally; re-licensed to GPLv3 in OpenCV contrib ([github.com/opencv/opencv_contrib/issues/2242](https://github.com/opencv/opencv_contrib/issues/2242)).
- AprilTag 3: BSD-2 ([github.com/AprilRobotics/apriltag](https://github.com/AprilRobotics/apriltag)).
- STag: open-source (repository inactive).
- Vuforia Engine: proprietary; free Basic plan + paid Classic/Pro plans.

---

## 2. BLE Beacon Positioning (iBeacon / Eddystone / AoA)

**One-line summary:** Battery-powered Bluetooth Low Energy beacons broadcast IDs that a phone measures by RSSI (or phase, in BLE 5.1 AoA mode) to estimate proximity/angle.

**How it works:** iBeacon (Apple) and Eddystone (Google) are the dominant advertisement formats. Trilateration from 3+ RSSI readings, or fingerprinting of RSSI against a pre-surveyed radio map, yields a 1–5 m position. BLE 5.1+ adds direction-finding via antenna arrays, achieving 0.5–1.5 m with sub-meter precision when the phone has an AoA-capable radio (mostly Android 12+ with compatible hardware; iOS does not expose AoA to apps) ([minew.com](https://www.minew.com/bluetooth-5-1-aoa-guide/), [nextwaves.com](https://nextwaves.com/blog/precision-indoors-finding-the-most-accurate-positioning-system-for-large-venues)).

**Accuracy / Range:** Standard RSSI trilateration 1–3 m, often 2–5 m in real buildings ([dataintelo.com](https://dataintelo.com/report/global-bluetooth-beacon-and-ibeacon-market), [locatify.com](https://locatify.com/ble-beacons-no-bull-beacon-review/)). With BLE 5.1 AoA, 0.5–1 m is achievable but needs a dedicated locator infrastructure, not a phone alone ([indoorsnavi.pro](https://indoorsnavi.pro/en/aoa/)). Range: typically 30 m unobstructed, up to 100 m ideal, falls fast through walls ([wiliot.com](https://www.wiliot.com/bluetooth-beacon)).

**Hardware + Install Cost:** $5–$30 per BLE beacon, plus survey labor for fingerprinting or AoA locator antenna arrays ($100s each). For a single building floor, $500–$3,000 is realistic; for a campus, $10k+. No subscription to Apple/Google for advertising.

**Smartphone compatibility:** Any iPhone 4S+ / Android 4.3+ can scan iBeacon/Eddystone. BLE AoA requires Android 12+ devices with antenna-array-compatible firmware (very rare in 2026). This is the technology behind NavCog3 (CMU).

**Research / Product Evidence:**
- NavCog3 (CMU) — uses BLE fingerprinting + particle-filter PDR hybrid; "provides indoor localization with high accuracy to support reliable navigation assistance" ([publications.ri.cmu.edu p270-sato.pdf](https://publications.ri.cmu.edu/storage/publications/2018/01/p270-sato.pdf), [researchgate.net/.../335592151](https://www.researchgate.net/publication/335592151_NavCog3_in_the_Wild_Large-scale_Blind_Indoor_Navigation_Assistant_with_Semantic_Features)).
- Wayfindr Open Standard (London) — first ITU-recognised standard for BLE-based audio navigation for the blind ([wayfindr.net/open-standard](https://www.wayfindr.net/open-standard), [itu.int/hub/2020/06/...](https://www.itu.int/hub/2020/06/how-the-wayfindr-open-standard-uses-new-tech-to-help-the-visually-impaired/)).
- GuideBeacon 2017 ([cheraghi et al., ieee PerCom](https://scholar.google.com/scholar_lookup?journal=Proceedings%20of%20the%202017%20IEEE%20International%20Conference%20on%20Pervasive%20Computing%20and%20Communications%20(PerCom)&title=GuideBeacon:%20Beacon-based%20indoor%20wayfinding%20for%20the%20blind,%20visually%20impaired,%20and%20disoriented&author=S.A.%20Cheraghi&author=V.%20Namboodiri&author=L.%20Walker)).
- Multi-bluetooth beacon deployment reduces worst-case error from 9.09 m to 2.94 m with just 3 beacons ([pmc.ncbi.nlm.nih.gov/articles/PMC12305822](https://pmc.ncbi.nlm.nih.gov/articles/PMC12305822/)).
- Hybrid BLE + Google Tango (Nair et al., 2018) — Lighthouse Guild ([lighthouseguild.org](https://lighthouseguild.org/a-hybrid-indoor-positioning-system-for-the-blind-and-visually-impaired-using-bluetooth-and-google-tango/)).
- BLE AoA explained — Minew 2024 ([minew.com](https://www.minew.com/bluetooth-5-1-aoa-guide/)).

**Failure modes:** Multipath and reflection from metal, walls, and bodies; rapid RSSI variation with phone orientation; per-deployment radio-map survey and re-survey every time a wall moves; AoA requires line-of-sight to a multi-antenna locator, not just any beacon; battery maintenance on hundreds of beacons.

**Open-source / License:** iBeacon/Eddystone are public formats. Android Beacon Library (AltBeacon) is open-source ([altbeacon.org](https://altbeacon.org/)). Many vendors (Estimote, Kontakt.io, Minew) provide closed SDKs but use the same open protocols.

---

## 3. UWB (Ultra-Wideband) Indoor Positioning

**One-line summary:** Pulses on a >500 MHz-wide spectrum give time-of-flight ranging to <10 cm; with ≥3 anchors a phone can be located.

**How it works:** UWB transceivers (Qorvo DW3000, Apple U1, NXP) exchange precisely-timestamped impulse packets. Time-difference-of-arrival (TDoA) or two-way-ranging (TWR) gives distance; multilateration gives position. Some setups add inertial fusion to push 30 cm accuracy to a moving user.

**Accuracy / Range:** Centimeter to decimeter — Apple Nearby Interaction framework and AirTag reach ~30 cm ranging with direction ([arxiv.org/html/2410.02329v1](https://arxiv.org/html/2410.02329v1), [newly.app](https://newly.app/sensors/uwb-mobile-apps)). Industrial UWB anchors deliver 10–30 cm absolute position over 10–50 m range.

**Hardware + Install Cost:** $20–$100 per anchor, anchors placed every 5–10 m for good geometry. Apple U1 chip is in iPhone 11+ and AirTags; Samsung, Xiaomi, Google Pixel 6+ also include UWB. Third-party anchor hardware (Qorvo, Sewio, Pozyx, BeSpoon) is $200–$2000 per node plus NRE. For a campus, $10k–$100k+ is realistic.

**Smartphone compatibility:** Yes — any iPhone 11/12/13/14/15/16/17 series or recent Samsung/Xiaomi/Google phone can do ranging via Nearby Interaction (iOS) or the Android UWB Jetpack API. Note that an "anchor infrastructure" must be deployed and registered with the OS or a private SDK.

**Research / Product Evidence:**
- iOS 27 UWB indoor navigation closer to commercial reality ([themobileknowledge.com](https://www.themobileknowledge.com/news/ios-27-brings-uwb-indoor-navigation-closer-to-commercial-reality/)).
- "UWB-Based Real-Time Indoor Positioning Systems" 2024 review ([mdpi.com/2076-3417/14/23/11005](https://www.mdpi.com/2076-3417/14/23/11005)).
- "The Effectiveness of UWB-Based Indoor Positioning Systems" ([mdpi.com/2076-3417/14/13/5646](https://www.mdpi.com/2076-3417/14/13/5646)).
- "AirTags for Human Localization, Not Just Objects" 2024 ([arxiv.org/html/2410.02329v1](https://arxiv.org/html/2410.02329v1)).
- Drift-free visual SLAM + UWB integration ([researchgate.net/.../363219018](https://www.researchgate.net/publication/363219018_Drift-Free_Visual_SLAM_for_Mobile_Robot_Localization_by_Integrating_UWB_Technology)).
- Qorvo UWB dev kits ([qorvo.com](https://www.qorvo.com/products/wireless-connectivity/ultra-wideband)).
- "Multi-Sensor Data Fusion Solutions for Blind and Visually Impaired" survey, 2023 — Kalman filter (27%) most preferred ([pmc.ncbi.nlm.nih.gov/articles/PMC10301813](https://pmc.ncbi.nlm.nih.gov/articles/PMC10301813/)).

**Failure modes:** Requires UWB-equipped handsets; not all students will have iPhone 11+; UWB signals blocked by metal and human bodies; battery life of anchors; Apple restricts access to Nearby Interaction for third-party UWB anchors.

**Open-source / License:** iOS Nearby Interaction SDK is free with an Apple developer account. Qorvo/NXP UWB SDKs are closed-source. Some research anchors use open firmware.

---

## 4. Wi-Fi RTT (IEEE 802.11mc) and Wi-Fi RSSI Fingerprinting

**One-line summary:** Either measure the round-trip-time to Wi-Fi access points (802.11mc FTM) for ~1–2 m ranging, or fingerprint the received signal strength (RSSI) from existing APs for ~3–5 m.

**How it works:** **RTT** uses Fine Timing Measurement frames between phone and a compatible AP; the time-of-flight of the frame exchange gives distance, and trilateration gives position. Android exposes this via the Wifi-RTT API. **RSSI fingerprinting** collects signal strengths at known points during a survey; later, real-time readings are matched (k-NN, deep nets) to the map.

**Accuracy / Range:** RTT typically 1–2 m indoors ([people.csail.mit.edu/bkph/FTMRTT](https://people.csail.mit.edu/bkph/FTMRTT), [dl.acm.org/doi/10.1016/j.pmcj.2021.101416](https://dl.acm.org/doi/10.1016/j.pmcj.2021.101416)). RSSI fingerprinting: 3–5 m common, 1–2 m with deep learning. Range 30–100 m per AP.

**Hardware + Install Cost:** Reuses existing campus APs (no new infrastructure for fingerprinting). For RTT, you must upgrade to 802.11mc-capable APs (most modern enterprise APs from Cisco, Aruba, Ruckus, Google Wi-Fi support FTM); consumer APs mostly do not. Survey cost is the dominant cost (1–2 h per floor).

**Smartphone compatibility:** Android only. iOS does not expose the FTM API to apps. Pixel 3+, Samsung Galaxy S10+, and most Wi-Fi 6E/7 Android phones support RTT.

**Research / Product Evidence:**
- "Indoor positioning with Wi-Fi Location: A survey of IEEE 802.11mc" 2025 ([arxiv.org/html/2509.03901v1](https://arxiv.org/html/2509.03901v1), [sciencedirect.com/...S0140366425003573](https://www.sciencedirect.com/science/article/abs/pii/S0140366425003573)).
- "Accurate indoor positioning using IEEE 802.11mc round trip time" — P. Gallo et al., 2021 ([dl.acm.org/doi/10.1016/j.pmcj.2021.101416](https://dl.acm.org/doi/10.1016/j.pmcj.2021.101416)).
- Android RTT docs ([developer.android.com/develop/connectivity/wifi/wifi-rtt](https://developer.android.com/develop/connectivity/wifi/wifi-rtt)).
- "Privacy-Preserving Positioning in Wi-Fi Fine Timing Measurement" 2022 ([crysp.petsymposium.org/popets/2022/popets-2022-0048.pdf](https://crysp.petsymposium.org/popets/2022/popets-2022-0048.pdf)).
- "Study of Wi-Fi Fingerprint-Based Indoor Positioning on a smartphone" ([researchgate.net/.../320054832](https://www.researchgate.net/publication/320054832_Study_of_Wi-Fi_Fingerprint-Based_Indoor_Positioning_on_a_smartphone)).
- Wi-Fi fingerprinting survey 2023 ([mdpi.com/1424-8220/23/18/7961](https://www.mdpi.com/1424-8220/23/18/7961)).

**Failure modes:** RSSI sensitive to bodies, doors, moving trolleys; APs need re-surveying if furniture moves; RTT only works with FTM-capable APs; multipath and NLOS conditions degrade both methods; no FTM on iOS.

**Open-source / License:** RSSI fingerprinting open-source (e.g., [ujwlwifi](https://github.com/location-competition/UJIIndoorLoc), [indoor-pos-loc](https://github.com/yxfcode/indoor-location)). RTT API: native Android.

---

## 5. Visual-Inertial Odometry (ARCore / ARKit / Google Geospatial VPS / Niantic Lightship VPS)

**One-line summary:** Use the phone's camera + IMU to track 6-DoF motion locally; optionally anchor that local map to a global image index.

**How it works:** **ARCore** (Google, Android) and **ARKit** (Apple, iOS) run on-device VIO: feature points are tracked across frames, fused with gyroscope/accelerometer at >100 Hz. Result: a local "world" with origin at session start, accurate to a few cm over a few meters, but drifting over long distances. **Google's Geospatial API / VPS** matches live frames against a Street-View-derived 3D index to get an absolute global pose. **Niantic Lightship VPS** does the same with Niantic's Wayspot scans.

**Accuracy / Range:** VIO local: ~1–3 cm relative, drifts to meters over 100 m. VPS Geospatial API: ~5 m, 5° under typical conditions ([developers.google.com/ar/develop/java/geospatial/enable](https://developers.google.com/ar/develop/java/geospatial/enable)). In low-GPS areas it relies on VPS; indoor reports vary 5–10 m or "incorrectly high" ([github.com/google-ar/arcore-unity-extensions/issues/211](https://github.com/google-ar/arcore-unity-extensions/issues/211)). Niantic claims "centimeter" but real-world retail pilots show meter-level.

**Hardware + Install Cost:** Zero new hardware — uses the phone. VPS requires a pre-scanned 3D model of the building (Street View car or Niantic Scaniverse). For Google Geospatial, you need Street View coverage of the building; for Niantic, you scan the building with the Scaniverse app.

**Smartphone compatibility:** All modern iPhones and Android phones support ARKit/ARCore. VPS availability varies: Google Geospatial works in many transit hubs/airports; Niantic VPS works in Niantic-mapped places (Pokéstops etc.).

**Research / Product Evidence:**
- "An Empirical Evaluation of Four Off-the-Shelf Proprietary Visual SLAMs" — ARKit shown to be most consistent, ARCore close second ([pmc.ncbi.nlm.nih.gov/articles/PMC9785098](https://pmc.ncbi.nlm.nih.gov/articles/PMC9785098/), [ar5iv.labs.arxiv.org/html/2207.06780](https://ar5iv.labs.arxiv.org/html/2207.06780)).
- "Localization Limitations of ARCore, ARKit, and HoloLens in Dynamic Large-Scale Industry Environments" — Feigl et al. ([scitepress.org/Papers/2020/89899/89899.pdf](https://www.scitepress.org/Papers/2020/89899/89899.pdf)).
- Google Geospatial API ([developers.google.com/ar/develop/geospatial](https://developers.google.com/ar/develop/geospatial)).
- Niantic Lightship VPS / 8th Wall ([info.nianticspatial.com/.../introducing-lightship-vps-for-web](https://info.nianticspatial.com/blog/introducing-lightship-vps-for-web)).
- "ASSESSING LOCALIZATION ACCURACY OF GOOGLE ARCore Geospatial API" — Tampere thesis ([trepo.tuni.fi/...HakamäkiSaku.pdf](https://trepo.tuni.fi/bitstream/handle/10024/157627/Hakam%C3%A4kiSaku.pdf)).
- "Why Visual Positioning System (VPS) is The Future of Navigation" ([nianticspatial.com](https://www.nianticspatial.com/campaigns/visual-positioning-system-maps-intro)).
- "ARKit as indoor positioning system" ([researchgate.net/.../337144233](https://www.researchgate.net/publication/337144233_ARKit_as_indoor_positioning_system)).

**Failure modes:** Low-texture walls (white corridors), dark rooms, motion blur, reflective glass, dynamic environments (crowds), phones in pocket; needs continuous camera stream (battery + privacy). VPS fails outside the pre-scanned index; long-term VIO drift is the #1 problem.

**Open-source / License:** ARCore SDK is free, Apache-2.0 with Google APIs ToS ([github.com/google-ar/arcore-android-sdk/blob/main/LICENSE](https://github.com/google-ar/arcore-android-sdk/blob/main/LICENSE)). ARKit is free with Apple developer account. Niantic Lightship SDK has a free tier for non-commercial use.

---

## 6. SLAM (ORB-SLAM3 and Semantic SLAM)

**One-line summary:** Build a map of the environment on the fly with feature/landmark recognition, then localize within that map; semantic SLAM also labels rooms, doors, signs.

**How it works:** ORB-SLAM3 (the leading open-source implementation) extracts ORB features, matches them across frames, builds a sparse 3D map, performs loop closure to correct drift, and tracks the camera. Visual-inertial mode fuses IMU. Semantic SLAM adds a CNN that labels objects ("door", "stairs", "elevator") usable for higher-level wayfinding.

**Accuracy / Range:** ORB-SLAM3 mono: ~0.05–0.3 m relative accuracy in well-textured scenes; absolute global position requires a known origin or loop closure. RGB-D variants reach 1 cm relative; long-term drift is still 1–5 m over 100 m without loop closure.

**Hardware + Install Cost:** No new hardware. Phone-only. Computational cost is high — modern flagships can run ORB-SLAM3 at 20–30 fps, mid-range phones struggle. Semantic SLAM adds GPU/CPU load.

**Smartphone compatibility:** Yes, but performance varies; needs a powerful SoC. Researchers have ported ORB-SLAM3 to iOS and Android NDK.

**Research / Product Evidence:**
- ORB-SLAM3 paper, "An Accurate Open-Source Library for Visual, Visual-Inertial and Multi-Map SLAM" ([ar5iv.labs.arxiv.org/html/2007.11898](https://ar5iv.labs.arxiv.org/html/2007.11898), [github.com/UZ-SLAMLab/ORB_SLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3)).
- "Indoor mobile robot localization system based on ORB-SLAM3 and ..." 2026 ([sciencedirect.com/...S1110016826000475](https://www.sciencedirect.com/science/article/pii/S1110016826000475)).
- "SLAM for Visually Impaired People: a Survey" 2022/24 ([arxiv.org/html/2212.04745v4](https://arxiv.org/html/2212.04745v4)).
- "A SLAM Based Semantic Indoor Navigation System for Visually Impaired Users" — Zhang et al. 2015 ([researchgate.net/.../304293832](https://www.researchgate.net/publication/304293832_A_SLAM_Based_Semantic_Indoor_Navigation_System_for_Visually_Impaired_Users), [scilit.com/.../0422cbaca36f01811b4759a35db97a0a](https://www.scilit.com/publications/0422cbaca36f01811b4759a35db97a0a)).
- "A Wearable Visually Impaired Assistive System Based on Semantic Visual SLAM" 2024 ([mdpi.com/1424-8220/24/11/3593](https://www.mdpi.com/1424-8220/24/11/3593)).
- "Visual-Inertial RGB-D SLAM with Encoder Integration of ORB ..." ([researchgate.net/.../384438218](https://www.researchgate.net/publication/384438218_Visual-Inertial_RGB-D_SLAM_with_Encoder_Integration_of_ORB_Triangulation_and_Depth_Measurement_Uncertainties)).
- Drift-correction research for VIO/SLAM ([arxiv.org/html/2404.10140v1](https://arxiv.org/html/2404.10140v1)).

**Failure modes:** Long-term drift without loop closure; featureless corridors, dynamic crowds, low light, motion blur; battery drain; for blind users the phone cannot stay in pocket, and real-time SLAM typically requires a chest-/head-mounted device.

**Open-source / License:** ORB-SLAM3 GPLv3 ([github.com/UZ-SLAMLab/ORB_SLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3)). Many semantic SLAM datasets/models are open (e.g., NYU Depth, ScanNet).

---

## 7. Depth Sensors / LiDAR on Phones

**One-line summary:** Use the active depth sensor (LiDAR on Pro iPhones/iPads, Time-of-Flight on some Androids) to build a 3D point cloud and localize via cloud matching or plane detection.

**How it works:** iPhone/iPad Pro LiDAR projects an IR dot pattern, measures return, and outputs a 256×192 depth frame at 30 Hz. This gives the OS (ARKit RoomPlan, Scene Reconstruction) a precise 3D mesh of the room. Localization can be: (a) local mesh-based relocalization (return to a previously scanned room), (b) plane/feature matching, (c) obstacle-distance rather than absolute position.

**Accuracy / Range:** Static: iPad Pro LiDAR depth <1 mm vs ground truth in static scenes ([sciencedirect.com/.../S2666165923000510](https://www.sciencedirect.com/science/article/pii/S2666165923000510)). iPhone LiDAR achieves useful landscape accuracy with reference points every 20 m ([mdpi.com/1424-8220/25/19/6141](https://www.mdpi.com/1424-8220/25/19/6141)). Effective range 5 m. ARKit scene reconstruction provides 6-DoF tracking that is "stable, drift-free" on non-LiDAR iPhone 14/15 too ([developer.apple.com/forums/...](https://developer.apple.com/forums/forums/topics/spatial-computing/spatial-computing-arkit)).

**Hardware + Install Cost:** No new infrastructure. But the **handset must be a LiDAR-equipped device** (iPhone Pro/Pro Max, iPad Pro 2020+). Recent Androids use indirect ToF (lower quality).

**Smartphone compatibility:** Limited subset of devices — iPhone 12 Pro/13 Pro/14 Pro/15 Pro/16 Pro/17 Pro and iPad Pro 2020+. Ordinary iPhones/Androids do not have it.

**Research / Product Evidence:**
- "Evaluating the accuracy and quality of an iPad Pro's built-in LiDAR" ([sciencedirect.com/.../S2666165923000510](https://www.sciencedirect.com/science/article/pii/S2666165923000510), [researchgate.net/.../370529618](https://www.researchgate.net/publication/370529618_Evaluating_the_accuracy_and_quality_of_an_iPad_Pro's_built-in_lidar_for_3D_indoor_mapping)).
- "2020 iPad Pro: Does the LiDAR sensor improve spatial tracking?" — empirical test ([vgis.io/2020/04/23/...](https://www.vgis.io/2020/04/23/2020-ipad-pro-does-the-lidar-sensor-improve-spatial-tracking/)).
- "Lidar is the iPad Pro's unexpected new feature" ([geoweeknews.com/articles/lidar-is-the-ipad-pros-unexpected-new-feature](https://www.geoweeknews.com/articles/lidar-is-the-ipad-pros-unexpected-new-feature/)).
- ARKit Scene Reconstruction / RoomPlan docs ([developer.apple.com/documentation/arkit](https://developer.apple.com/documentation/arkit/understanding-world-tracking)).

**Failure modes:** Short range (5 m) makes it useful only for local room-scale; reflective/dark/IR-absorbing surfaces; no outdoor operation; only available on premium devices; for navigation it must be combined with localization (e.g., RoomCapture fingerprinting).

**Open-source / License:** ARKit RoomPlan/RealityKit APIs are free, Apple proprietary. iPad Pro LiDAR data is consumable via ARKit & Swift.

---

## 8. Magnetic-Field-Based Positioning

**One-line summary:** Buildings distort the Earth's magnetic field; the resulting anomalies are unique "fingerprints" that a magnetometer can match.

**How it works:** The phone's magnetometer (3-axis, present on every modern phone) measures the local field. During a one-time survey, a "magnetic map" of the building is recorded. The user's live magnetometer readings are matched (k-NN, CNN regression) to the map.

**Accuracy / Range:** 2–4 m typical, 1–2 m with deep nets ([mdpi.com/1424-8220/22/11/4014](https://www.mdpi.com/1424-8220/22/11/4014), [link.springer.com/chapter/10.1007/978-981-15-8983-6_26](https://link.springer.com/chapter/10.1007/978-981-15-8983-6_26)). Effective range = whole building.

**Hardware + Install Cost:** Zero new infrastructure (existing Earth's field + building steel). One-time survey walkthrough.

**Smartphone compatibility:** All iPhones and Android phones with a magnetometer (essentially 100% of modern devices). Best of any approach for a hackathon from a "no infrastructure" standpoint.

**Research / Product Evidence:**
- "Indoor positioning system using geomagnetic anomalies" — IEEE ([ieeexplore.ieee.org/document/6418947](https://ieeexplore.ieee.org/document/6418947/)).
- "Indoor Localization Using Magnetic Fields" dissertation — UNT ([digital.library.unt.edu/ark:/67531/metadc103371](https://digital.library.unt.edu/ark:/67531/metadc103371/m2/1/high_res_d/dissertation.pdf)).
- "Magnetic-Field-Based Indoor Positioning Using Temporal Convolutional Networks" 2023 ([mdpi.com/1424-8220/23/3/1514](https://www.mdpi.com/1424-8220/23/3/1514)).
- "Magnetic Indoor Localization through CNN Regression and Rotation Augmentation" ([arxiv.org/html/2604.22896v1](https://arxiv.org/html/2604.22896v1)).
- IndoorAtlas (commercial, originally by Univ. of Oulu) — first commercial geomagnetic IPS ([gpsworld.com/indooratlas-announces-geomagnetic-indoor-positioning-service](https://www.gpsworld.com/indooratlas-announces-geomagnetic-indoor-positioning-service/), [indooratlas.com/blog/magnetic-positioning-technology-how-it-works](https://www.indooratlas.com/blog/magnetic-positioning-technology-how-it-works/)).
- LocateMe (Univ. of North Texas) ([scholar.google.com/.../2508037.2508054](https://scholar.google.com/scholar_lookup?journal=ACM%20Trans.%20Intell.%20Syst.%20Technol.&title=LocateMe:%20Magnetic-fields-based%20indoor%20localization%20using%20smartphones&author=K.P.%20Subbu&author=B.%20Gozick&author=R.%20Dantu&volume=4&publication_year=2013&pages=1-27&doi=10.1145/2508037.2508054)).
- GROPING (Zhang et al., 2015) — geomagnetism + crowdsourcing ([ieeexplore.ieee.org/document/6418947](https://ieeexplore.ieee.org/document/6418947/)).
- "A Unique Blend of AR and Indoor Positioning" (IndoorAtlas blog) ([indooratlas.com/blog/a-unique-blend-of-augmented-reality-ar-and-indoor-positioning](https://www.indooratlas.com/blog/a-unique-blend-of-augmented-reality-ar-and-indoor-positioning/)).

**Failure modes:** Requires a calibration survey; sensitivity to moving metal (elevators, doors, chairs) and to electronic devices (lifts, microwaves); magnetometer drift requires periodic re-calibration; the field is **not unique in every cell** — many places share fingerprints, causing ambiguity; Tesla-style new buildings with reinforced concrete have weak anomalies.

**Open-source / License:** IndoorAtlas SDK is closed-source. Research datasets and many reference papers are open. Some Android open-source IMU/Mag fusion tools.

---

## 9. Pedestrian Dead Reckoning (PDR) — Step Counting + Compass

**One-line summary:** Estimate user motion by counting steps from the accelerometer, multiplying by stride length, and integrating heading from compass/gyro.

**How it works:** Accelerometer peaks → step detection; step length is learned per user; magnetometer + gyroscope → heading. Position is integrated (dead-reckoned) from the start point. Used as the spine of almost every modern indoor navigation system because it works with no infrastructure, in any pocket, and provides smooth short-term motion between "absolute" fixes (markers, beacons, Wi-Fi fixes, etc.).

**Accuracy / Range:** Heading drift is the main problem — even a 5° error becomes ~9 m over 100 m. Step-length estimation accuracy is 5–10%. With perfect heading, PDR drifts ~2–5% of distance.

**Hardware + Install Cost:** No new infrastructure; uses phone IMU. Compute cost is tiny.

**Smartphone compatibility:** Every smartphone. Works with phone-in-pocket (which is exactly the constraint Manduchi's UCSC apps target).

**Research / Product Evidence:**
- "All the Way There and Back: Inertial-Based, Phone-in-Pocket Indoor Wayfinding and Backtracking Apps for Blind Travelers" — Manduchi lab, 2024 ([dl.acm.org/doi/10.1145/3696005](https://dl.acm.org/doi/10.1145/3696005), [dl.acm.org/doi/10.1145/3676522](https://dl.acm.org/doi/10.1145/3676522)).
- "Drift Control of Pedestrian Dead Reckoning (PDR) for Long Period ..." ([mdpi.com/2673-4591/10/1/21](https://www.mdpi.com/2673-4591/10/1/21), [ahmedmansoour.github.io/.../drift-control-pdr-long-period-navigation-smartphone-poses.html](https://ahmedmansoour.github.io/indoor-positioning-hub/publications/drift-control-pdr-long-period-navigation-smartphone-poses.html)).
- "Smartphone-Based Pedestrian Dead Reckoning Integrated with Visible Light Positioning" ([arxiv.org/pdf/2301.03471](https://arxiv.org/pdf/2301.03471)).
- "A Robust Step Detection and Stride Length Estimation" ([researchgate.net/.../340879649](https://www.researchgate.net/publication/340879649_A_Robust_Step_Detection_and_Stride_Length_Estimation_for_Pedestrian_Dead_Reckoning_Using_a_Smartphone)).
- "Enhancing real-time heading estimation for pedestrian navigation" 2025 ([nature.com/articles/s41598-025-13390-9](https://www.nature.com/articles/s41598-025-13390-9)).
- RouteNav (Manduchi 2023) — PDR in transit hub ([escholarship.org/content/qt1gv8z9jt](https://escholarship.org/content/qt1gv8z9jt/qt1gv8z9jt.pdf?v=lg)).

**Failure modes:** Heading drift dominates; magnetometer heading has severe indoor errors near metal; step length varies with user gait, walking speed, carrying items, stairs; cannot give absolute position; long walks cause cumulative error; smartphone placement (hand, pocket, bag) changes sensor readings.

**Open-source / License:** Many open-source PDR implementations exist (e.g., [github.com/avestura/PDR](https://github.com/avestura/PDR), IndoorAtlas open-source PDR examples). Android/iOS raw IMU APIs are free.

---

## 10. RFID / NFC Tags, Infrared Audible Signage (Talking Signs), QR-Code-Based Wayfinding (NaviLens)

### 10a. RFID / NFC

**One-line summary:** Tiny passive (NFC) or active (RFID) tags broadcast IDs that the phone reads at point-blank (NFC) or up to ~10 m (active RFID); the tag's known location is the user's location.

**How it works:** NFC is a near-contact protocol (4 cm) that every modern phone supports; "tap to identify". Active RFID beacons broadcast periodically and a phone/RFID reader logs nearby tags with RSSI.

**Accuracy / Range:** NFC — point reading, 0–5 cm. Active RFID — 0.5–5 m with RSSI, several meters range.

**Hardware + Install Cost:** NFC stickers are $0.10–$0.50 each; active RFID tags $5–$30; readers $50–$500.

**Smartphone compatibility:** NFC: 100% of modern phones. Active RFID: requires a reader; phones don't have a UHF RFID radio.

**Research / Product Evidence:**
- "A Blind Navigation System Using RFID for Indoor Environments" — Academia ([academia.edu/9062618](https://www.academia.edu/9062618/A_Blind_Navigation_System_Using_RFID_for_Indoor_Environments)).
- "Indoor navigation system for visually impaired" — ResearchGate ([researchgate.net/.../220795837](https://www.researchgate.net/publication/220795837_Indoor_navigation_system_for_visually_impaired)).
- "Accurate positioning using long range active RFID" — Alghamdi et al. 2014 ([sciencedirect.com/.../10.1016/j.jnca.2013.10.015](https://doi.org/10.1016/j.jnca.2013.10.015)).
- Survey of indoor navigation for VI persons — includes RFID comparison ([pmc.ncbi.nlm.nih.gov/articles/PMC7038337](https://pmc.ncbi.nlm.nih.gov/articles/PMC7038337/)).

**Failure modes:** NFC is too short-range to be useful for continuous navigation — only point-of-interest. Active RFID requires dedicated reader hardware; reader antenna orientation matters; interference.

**Open-source / License:** NFC Android APIs are open (Apache). Active RFID SDKs are vendor-specific.

### 10b. Talking Signs / RIAS (Remote Infrared Audible Signage)

**One-line summary:** Wall-mounted IR transmitters continuously broadcast a short voice message describing the location; a handheld or phone IR receiver converts the signal to audio and points the user toward the sign.

**How it works:** Developed by Smith-Kettlewell Eye Research Institute in the 1990s. An IR LED array at the sign pulses a directional beam carrying a recorded voice message. The user holds a receiver and the audio is louder when the receiver is pointed at the sign.

**Accuracy / Range:** Each sign is heard at 2–10 m; the directional beam means the user can locate the sign by pointing. Sign identification is unambiguous.

**Hardware + Install Cost:** $300–$500 per transmitter (Talking Signs Inc., Comfygo). Requires power at each sign. Receiver hardware: $50–$300, or smartphone adapters. Many transit systems have piloted RIAS.

**Smartphone compatibility:** Native IR receivers are not on most phones; a clip-on IR receiver + headphone jack adapter is required. Newer work ("virtual audible signage") pushes the IR receiver to a smartphone camera + app.

**Research / Product Evidence:**
- Talking Signs overview — Smith-Kettlewell ([ski.org/projects](https://www.ski.org/projects/)).
- "Remote Infrared Audible Signage" Wikipedia ([en.wikipedia.org/wiki/Remote_infrared_audible_signage](https://en.wikipedia.org/wiki/Remote_infrared_audible_signage)).
- FTA pilot program report ([transit.dot.gov/.../FTA0012_Research_Report_Summary.pdf](https://www.transit.dot.gov/sites/fta.dot.gov/files/docs/FTA0012_Research_Report_Summary.pdf)).
- "Smartphone Based Virtual Audible Signage" — CalState ([scholarworks.calstate.edu/downloads/m039k852h](https://scholarworks.calstate.edu/downloads/m039k852h)).
- "Remote infrared audible signage" for transit ([worldtransitresearch.info/research/1227](https://www.worldtransitresearch.info/research/1227/), [pubmed.ncbi.nlm.nih.gov/12366323](https://pubmed.ncbi.nlm.nih.gov/12366323/)).
- "Transit Accessibility Improvement Through Remote Infrared Signage" — Project ACTION ([accessforblind.org/.../Transit%20Accessibility%20Improvement%20Through%20Remote%20Infrared%20Signage.pdf](https://accessforblind.org/publications/ProjectAction/Transit%20Accessibility%20Improvement%20Through%20Remote%20Infrared%20Signage.pdf)).

**Failure modes:** Requires dedicated receivers; power/maintenance; sunlight can interfere with IR; no position on its own (it only labels the sign).

**Open-source / License:** Talking Signs hardware is proprietary (commercial product). Research papers are open.

### 10c. NaviLens / High-density QR

**One-line summary:** Colorful, high-density codes (similar to QR but designed for long-range, wide-angle, motion-blurred reading) that broadcast a short ID and metadata to a phone from meters away.

**How it works:** Each "ddTag" (distant dense tag) can be read from up to 12 m, at 160° field-of-view, while the user is moving and without focusing the camera. The phone app decodes the tag, fetches a multilingual audio description, and may also infer orientation. NaviLens is the leading commercial implementation, developed at Universidad de Alicante.

**Accuracy / Range:** 12 m read range; sub-meter position via known tag coordinates + camera pose; designed to work "in the dark" via high-contrast colors.

**Hardware + Install Cost:** $0.05–$2 per printed tag; free NaviLens app (iOS/Android) and free tag generator ([navilens.com/en](https://www.navilens.com/en)). Optional SDK for embedding in your own app.

**Smartphone compatibility:** Any iPhone/iPad with iOS 12+; Android 7+.

**Research / Product Evidence:**
- NaviLens official ([navilens.com/en](https://www.navilens.com/en)).
- MTA Inclusive Wayfinding through NaviLens (NY) — open data ([github.com/nymta/MTA-USDOT-SMART-Grant-Inclusive-Wayfinding-through-NaviLens](https://github.com/nymta/MTA-USDOT-SMART-Grant-Inclusive-Wayfinding-through-NaviLens-/actions)).
- "NaviLens Provides Blind or Low Vision Individuals Invaluable Support" ([empowerabilities.org/news/navilens-provides-blind-or-low-vision-individuals-invaluable-support](https://www.empowerabilities.org/news/navilens-provides-blind-or-low-vision-individuals-invaluable-support)).
- "NaviLens - Accessibility - Cal Poly, San Luis Obispo" ([accessibility.calpoly.edu/navilens](https://accessibility.calpoly.edu/navilens)).
- Freelens (NaviLens-style generator) ([github.com/sjtrny/freelens](https://github.com/sjtrny/freelens/)).
- Galloways description ([galloways.org.uk/navilens](https://galloways.org.uk/navilens/)).

**Failure modes:** Requires visible tags and a phone camera pointing in their general direction; not a continuous localization; tags can be vandalized or damaged; needs good contrast lighting.

**Open-source / License:** NaviLens SDK and the Freelens alternative are open-source (NaviLens SDK free for non-commercial; commercial terms on request).

---

## 11. Hybrid Approaches (Sensor Fusion) — What Research Says Works Best for Blind Navigation

**One-line summary:** Combine 2+ modalities (e.g., BLE + PDR, UWB + VIO, ArUco + magnetic + PDR) with a Kalman / particle filter; research consensus is that hybrid is dramatically better than any single technique for blind navigation.

**How it works:** A particle filter or extended Kalman filter fuses a "high-rate but drifting" signal (PDR, VIO) with a "low-rate but absolute" signal (BLE fingerprint, UWB, marker pose, Wi-Fi fix, magnetic match) so the user gets smooth continuous position plus no long-term drift.

**Accuracy / Range:** Best results in literature:
- NavCog3 BLE + PDR: ~1–2 m in malls and airports.
- UWB + VIO: ~30 cm; long-term drift bounded.
- BLE + Google Tango: 1–2 m (Nair 2018).
- MDIRECT (magnetic + PDR): 1–2 m.
- Hybrid camera + PDR + Wi-Fi: 0.5–1.5 m.

**Hardware + Install Cost:** Whatever the components require. Often the cheapest "good" system is BLE beacons ($) + PDR ($0) + smartphone.

**Smartphone compatibility:** Yes, as long as component signals are available.

**Research / Product Evidence:**
- "Multi-Sensor Data Fusion Solutions for Blind and Visually Impaired" 2023 — Kalman filter 27%, deep learning close second ([pmc.ncbi.nlm.nih.gov/articles/PMC10301813](https://pmc.ncbi.nlm.nih.gov/articles/PMC10301813/)).
- "Navigation Framework for Blind and Visually Impaired" 2025 — fusion-based navigation framework ([arxiv.org/pdf/2501.15819](https://arxiv.org/pdf/2501.15819)).
- "A comprehensive review of navigation systems for visually impaired" 2024 ([sciencedirect.com/.../S2405844024078563](https://www.sciencedirect.com/science/article/pii/S2405844024078563)).
- "A Hybrid Indoor Positioning System for the Blind and Visually Impaired Using Bluetooth and Google Tango" — Lighthouse Guild ([lighthouseguild.org/.../using-bluetooth-and-google-tango](https://lighthouseguild.org/a-hybrid-indoor-positioning-system-for-the-blind-and-visually-impaired-using-bluetooth-and-google-tango/)).
- NavCog3 — BLE + PDR particle filter ([publications.ri.cmu.edu/storage/publications/2018/01/p270-sato.pdf](https://publications.ri.cmu.edu/storage/publications/2018/01/p270-sato.pdf)).
- "Drift-Free Visual SLAM for Mobile Robot Localization by Integrating UWB" ([researchgate.net/.../363219018](https://www.researchgate.net/publication/363219018_Drift-Free_Visual_SLAM_for_Mobile_Robot_Localization_by_Integrating_UWB_Technology)).
- "Hybrid Indoor Positioning System Based on Visible Light + Bluetooth" 2023 ([mdpi.com/1424-8220/23/16/7199](https://www.mdpi.com/1424-8220/23/16/7199)).
- "Robust Indoor Localization System Integrating Visual Localization aided by CNN" 2019 ([mdpi.com/1424-8220/19/2/249](https://www.mdpi.com/1424-8220/19/2/249)).
- "MDIRECT—Magnetic field strength and peDestrIan dead RECkoning" 2018 ([researchgate.net/...MDIRECT](https://www.researchgate.net/publication/MDIRECT-Magnetic_field_strength_and_peDestrIan_dead_RECkoning_based_indoor_localizaTion)).
- "Real-World Robust Indoor Positioning in Smart Museums" 2024 ([ceur-ws.org/Vol-4206/SOCIALIZE-05.pdf](https://ceur-ws.org/Vol-4206/SOCIALIZE-05.pdf)).
- "Navigation Solutions for Blind and Visually Impaired Persons" 2026 ([tandfonline.com/doi/full/10.1080/10447318.2026.2643358](https://www.tandfonline.com/doi/full/10.1080/10447318.2026.2643358)).
- "Indoor Positioning on Disparate Commercial Smartphones Using Wi-Fi" — Ashraf et al. ([pmc.ncbi.nlm.nih.gov/articles/PMC6806077](https://pmc.ncbi.nlm.nih.gov/articles/PMC6806077/)).

**Failure modes:** Depends on components; usually the failure of the absolute-fix sensor causes large jump errors. The 2024 "comprehensive review" identifies adaptability to change and long-term robustness as the still-open research questions.

**Open-source / License:** Depends on components; the NavCog3 server is partially open; ROS 2 has many sensor-fusion packages; Google research published a particle-filter PDR + Wi-Fi open-source blueprint.

---

## 12. Camera Relocalization / Image-Retrieval-Based Positioning ("Instant Visual Localization")

**One-line summary:** Snap a photo; a server compares the image to a reference database of geotagged images and returns the camera pose (6-DoF).

**How it works:** A global image descriptor (e.g., NetVLAD, AP-GeM) is extracted on-device; the top-N similar reference images are retrieved from a database; local feature matching + PnP solve the 6-DoF pose. Some systems do this entirely on-device (Apple's Visual Look Up, Google Lens, Maps Live View).

**Accuracy / Range:** Niantic Lightship VPS, Google Geospatial: ~5 m typical, decimeter possible with good coverage ([nianticspatial.com](https://www.nianticspatial.com/campaigns/visual-positioning-system-maps-intro)). NAVER Labs, Google research show sub-meter in well-mapped areas.

**Hardware + Install Cost:** Server-side: reference image database of the building (can be crowdsourced). One-time scan.

**Smartphone compatibility:** Yes, any phone with a camera. Compute requirements modest if descriptor is precomputed and the search is approximate-nearest-neighbor.

**Research / Product Evidence:**
- NAVER Labs "Methods for visual localization" ([europe.naverlabs.com/blog/methods-for-visual-localization](https://europe.naverlabs.com/blog/methods-for-visual-localization/)).
- "A Visual Indoor Localization Method Based on Efficient Image Retrieval" 2024 ([scirp.org/journal/paperinformation?paperid=131240](https://www.scirp.org/journal/paperinformation?paperid=131240)).
- Jianfei Cai et al., "Efficient image retrieval based mobile indoor localization" ([jianfei-cai.github.io/mobile-indoor-localization.pdf](https://jianfei-cai.github.io/mobile-indoor-localization.pdf)).
- "A Scene is Worth a Thousand Features: Feed-Forward Camera Localization" 2025 ([arxiv.org/html/2510.00978v2](https://arxiv.org/html/2510.00978v2), [openreview.net/forum?id=rmDA02o8MV](https://openreview.net/forum?id=rmDA02o8MV)).
- Niantic Lightship / 8th Wall VPS documentation ([info.nianticspatial.com/blog/introducing-lightship-vps-for-web](https://info.nianticspatial.com/blog/introducing-lightship-vps-for-web)).
- "A Framework for Selecting Between Image Alignment, Niantic VPS..." ([note.com/thedesignium/n/nff53f3bb42e4](https://note.com/thedesignium/n/nff53f3bb42e4)).
- Multiset commercial VPS ([multiset.ai/visual-positioning-system](https://multiset.ai/visual-positioning-system)).
- "Robust Indoor Localization System Integrating Visual Localization aided by CNN-based image ..." ([mdpi.com/1424-8220/19/2/249](https://www.mdpi.com/1424-8220/19/2/249)).

**Failure modes:** Requires a pre-existing reference image database; lighting/season changes; motion blur; privacy concerns (camera always on); no fix in unmapped areas.

**Open-source / License:** Many descriptor networks (NetVLAD, AP-GeM) are open-source under MIT. Niantic Lightship VPS and Google Geospatial APIs are commercial; the underlying research is largely open.

---

## Deployed / Researched Blind-Navigation Systems (consolidated)

| System | Organization | Modality | Phone? | Open-source | URL |
|---|---|---|---|---|---|
| **NavCog3** | CMU | BLE beacons + PDR + particle filter, semantic landmarks | iOS | Partial | [cs.cmu.edu/~NavCog](https://www.cs.cmu.edu/~NavCog/navcog.html), [publications.ri.cmu.edu p270-sato.pdf](https://publications.ri.cmu.edu/storage/publications/2018/01/p270-sato.pdf) |
| **Microsoft Soundscape** | Microsoft (now community) | 3D audio + GPS + iBeacons; no positioning, just spatial awareness | iOS | MIT (open-sourced Jan 2023) | [github.com/microsoft/soundscape](https://github.com/microsoft/soundscape), [github.com/soundscape-community/soundscape](https://github.com/soundscape-community/soundscape), [microsoft.com/.../soundscape](https://www.microsoft.com/en-us/research/blog/microsoft-soundscape-new-horizons-with-a-community-driven-approach/) |
| **NaviLens** | Univ. of Alicante | ddTag color codes | iOS/Android | Open SDK | [navilens.com/en](https://www.navilens.com/en) |
| **Wayfindr** | Wayfindr / ustwo (London) | BLE audio open standard | iOS/Android | Open Standard (MIT) | [wayfindr.net/open-standard](https://www.wayfindr.net/open-standard), [github.com/Wayfindr-engineering](https://github.com/Wayfindr-engineering), [wayfindr.net/how-audio-navigation-works](https://www.wayfindr.net/how-audio-navigation-works) |
| **Aira** | Aira Tech Corp. | Remote sighted agent via phone camera | iOS/Android | Closed | [aira.io](https://aira.io/) |
| **Be My Eyes** | Be My Eyes (Denmark) | Remote sighted agent + AI (GPT-4V) | iOS/Android | Closed (app only) | [bemyeyes.com](https://www.bemyeyes.com/) |
| **GoodMaps** | GoodMaps (US) | Camera + computer vision matching (Crowd-sourced 3D) | iOS/Android | Closed | [goodmaps.com](https://goodmaps.com/), [apps.apple.com/.../goodmaps-indoor-navigation](https://apps.apple.com/hr/app/goodmaps-indoor-navigation/id6444539843) |
| **BlindSquare** | MIPsoft (Finland) | GPS + BLE beacons | iOS | Closed | [blindsquare.com](https://www.blindsquare.com/) |
| **Talking Signs / RIAS** | Smith-Kettlewell / commercial | IR audio signage + handheld receiver | Receiver hardware | Hardware proprietary | [en.wikipedia.org/wiki/Remote_infrared_audible_signage](https://en.wikipedia.org/wiki/Remote_infrared_audible_signage) |
| **Drishti** | Univ. of Florida (deprecated) | Ultrasonic / sonar wearable | No (custom) | N/A | [ski.org/projects](https://www.ski.org/projects/) |
| **Manduchi's Wayfinding + Backtracking apps** | UC Santa Cruz | Phone-in-pocket PDR, indoor floor maps | iOS | Research code available | [dl.acm.org/doi/10.1145/3696005](https://dl.acm.org/doi/10.1145/3696005), [news.ucsc.edu/2024/10/manduchi-wayfinding-apps](https://news.ucsc.edu/2024/10/manduchi-wayfinding-apps/) |
| **Snap&Nav** | IBM Research / Tsukuba | Camera + floor-map analysis, no prebuilt digital map | Smartphone | Research code | [dl.acm.org/doi/10.1145/3676522](https://dl.acm.org/doi/10.1145/3676522), [wotipati.github.io/.../SnapAndNav.html](https://wotipati.github.io/projects/other_papers/MobileHCI2024_SnapAndNav/MobileHCI2024_SnapAndNav.html) |
| **RouteNav** | UCSC | PDR + AI wayfinding in transit hubs | iOS | Research | [escholarship.org/content/qt1gv8z9jt](https://escholarship.org/content/qt1gv8z9jt/qt1gv8z9jt.pdf?v=lg) |
| **Tactile Maps (Lighthouse)** | Smith-Kettlewell + Rightpoint | Haptic + audio tactile maps | Mobile app | Open | [rightpoint.com/work/lighthouse](https://www.rightpoint.com/work/lighthouse) |
| **Harvard nav app (2026)** | Harvard | Smartphone camera + ML, obstacle guidance | Android | Research | [techxplore.com/.../2026-08-smartphone-app-vision-users-obstacles.html](https://techxplore.com/news/2026-08-smartphone-app-vision-users-obstacles.html) |
| **AI outdoor nav (Nature 2026)** | Stanford/Manduchi | Outdoor PDR + AI guidance | Smartphone | Research | [nature.com/articles/s41551-026-01772-x](https://www.nature.com/articles/s41551-026-01772-x) |
| **A Wearable SLAM** | (Sun Yat-sen Univ.) | RGB-D semantic SLAM | Custom wearable | Research | [pmc.ncbi.nlm.nih.gov/articles/PMC7926395](https://pmc.ncbi.nlm.nih.gov/articles/PMC7926395/) |
| **IndoorATLAS** | IndoorAtlas (commercial) | Magnetic fingerprinting | iOS/Android SDK | Closed | [indooratlas.com](https://www.indooratlas.com/) |

---

## Comparison Table

| # | Approach | Accuracy | Infra cost | Phone-native? | Adapts to change? | Open-source? |
|---|---|---|---|---|---|---|
| 1 | Fiducial markers (ArUco / AprilTag / Vuforia) | 1–10 cm @ 1 m, 1–5° rotation | $ (print) | Yes (any phone w/ camera) | No — re-print if map changes | ArUco GPLv3; AprilTag BSD; Vuforia proprietary |
| 2 | BLE beacons (iBeacon/Eddystone) | 1–3 m (RSSI), 0.5–1.5 m (AoA) | $ (beacons + survey) | Yes | Re-survey if room changes | Open formats; SDKs mostly closed |
| 3 | UWB | 10–30 cm | $$ (anchors + survey) | iPhone 11+/select Android | Yes with anchors | Proprietary; some research open |
| 4 | Wi-Fi RTT / RSSI fingerprint | 1–2 m (RTT), 3–5 m (RSSI) | $ if reusing APs; $$ for FTM APs | Android only (RTT); both for RSSI | Re-survey if APs move | Open |
| 5 | ARCore/ARKit VIO + VPS | 1–3 cm relative, 5 m global (VPS) | $0 (VPS) — $ to scan building | Yes (any modern phone) | Yes (VPS update) | ARCore Apache 2.0; ARKit free; Niantic VPS commercial |
| 6 | SLAM (ORB-SLAM3 + semantic) | 0.05–0.3 m relative; drifts over time | $0 | Yes (flagship recommended) | Requires re-mapping | ORB-SLAM3 GPLv3 |
| 7 | LiDAR / depth sensor | <1 mm (static); 5 m range | $0 but only on Pro devices | iPhone/iPad Pro only | Yes (re-scan) | Apple proprietary APIs |
| 8 | Magnetic field | 2–4 m | $0 (survey) | Yes (any modern phone) | Re-survey after major metal changes | Mostly closed (IndoorAtlas) |
| 9 | PDR (IMU + compass) | drifts ~2–5% of distance | $0 | Yes | n/a (relative) | Open |
| 10 | RFID / NFC / Talking Signs / NaviLens | Point reading (NFC) / 12 m (NaviLens) | $ (tags) | Yes (NFC & NaviLens), special rx for Talking Signs | Re-tagging | Mixed; NaviLens open, Talking Signs proprietary |
| 11 | Hybrid (sensor fusion) | 0.5–2 m typical | Varies | Yes | Most robust of all | Many open frameworks |
| 12 | Image retrieval / VPS | 1–5 m | $ to build image DB | Yes | Re-indexing | Research open; commercial mostly closed |

---

## Top 3 Recommended Approaches for a Hackathon Demo

### 1. Marker-based AR (ArUco or AprilTag) + ARCore/ARKit VIO for in-between localization

**Why:** This is the team's current design and remains the **best demo-time trade-off** because (a) it's printable and free, (b) it gives a satisfying 6-DoF pose visible on a phone in real time, (c) detection is robust in well-lit corridors, (d) it is visually compelling for a hackathon audience, and (e) ArUco and AprilTag are BSD/GPL open-source with strong community support. Vuforia adds polished AR rendering and a Unity plug-in but its license is restrictive for commercial use. For the hackathon, start with OpenCV ArUco in Unity (or AprilTag in a separate C# wrapper) and overlay with Unity AR Foundation. Drift between markers is small enough (<3 m in a corridor) that audio turn-by-turn instructions reset the error.

**Evidence:** FMAC paper shows ArUco pose accuracy is "much better between 500–900 mm in depth" with 100% detection ([arxiv.org/html/2601.07723v1](https://arxiv.org/html/2601.07723v1)). AprilTag is the de-facto standard in robotics and used by NASA ([github.com/nasa/AprilNav](https://github.com/nasa/AprilNav/blob/master/AprilTags/Tag36h11.h)). The biggest practical risk is lighting and the user must hold the phone up — a real failure mode for blind users.

### 2. NaviLens (ddTag) color codes for wayfinding

**Why:** NaviLens is purpose-built for blind/low-vision users, with 12 m read range, 160° field of view, and "read while moving" performance that no QR code or AprilTag can match ([navilens.com/en](https://www.navilens.com/en)). Free SDK, free app, free tag generator, and it has a real-world deployment track record (NY MTA, Cal Poly, EU transit). For a hackathon demo, you can print 20 tags for under $5 and place them at decision points. The downside is that NaviLens is *point-of-interest*, not continuous — you need to combine it with PDR or ARCore to fill the gaps.

**Evidence:** Cal Poly accessibility office deployment ([accessibility.calpoly.edu/navilens](https://accessibility.calpoly.edu/navilens)); MTA Inclusive Wayfinding open data ([github.com/nymta/MTA-USDOT-SMART-Grant](https://github.com/nymta/MTA-USDOT-SMART-Grant-Inclusive-Wayfinding-through-NaviLens-/actions)); NaviLens app is in the iOS and Android stores.

### 3. Hybrid PDR (phone-in-pocket) + BLE beacons + UWB-ready haptics

**Why:** This is what the research consensus says works best for blind users: a "phone-in-pocket" experience with smooth inertial motion and sparse absolute fixes from low-cost BLE beacons. Manduchi's UCSC Wayfinding app proves phone-in-pocket is viable ([dl.acm.org/doi/10.1145/3696005](https://dl.acm.org/doi/10.1145/3696005)); NavCog3 proves BLE + PDR particle filter works at campus scale ([publications.ri.cmu.edu p270-sato.pdf](https://publications.ri.cmu.edu/storage/publications/2018/01/p270-sato.pdf)). For the hackathon, you can drop $200 of BLE beacons into a single test floor, survey in 30 min, and have a 1–2 m system that runs in the pocket. If you also own a UWB-equipped phone, the framework can absorb UWB ranging (iOS Nearby Interaction) for 30 cm without changing the user experience.

**Evidence:** The 2023 multi-sensor fusion review identifies Kalman filter + IMU as the most-preferred method for blind navigation ([pmc.ncbi.nlm.nih.gov/articles/PMC10301813](https://pmc.ncbi.nlm.nih.gov/articles/PMC10301813/)). The 2024 comprehensive review highlights hybrid systems as the practical state of the art ([sciencedirect.com/.../S2405844024078563](https://www.sciencedirect.com/science/article/pii/S2405844024078563)). UWB Nearby Interaction ([themobileknowledge.com](https://www.themobileknowledge.com/news/ios-27-brings-uwb-indoor-navigation-closer-to-commercial-reality/)) is a fast-improving option for forward-compatibility.

---

## Summary recommendation

For the **hackathon demo**, the winning combination is:

> **NaviLens tags at every decision point + ARCore/ARKit VIO (or Vuforia Image Targets) for in-corridor localization + spatial audio + haptics for blind users + AR overlay for low-vision users.**

This minimises infrastructure cost (mostly printing), maximises robustness (NaviLens as absolute fix, VIO/Vuforia for in-between), respects the phone-in-pocket constraint (NaviLens reads while moving), and gives a tangible, demo-friendly AR visual. The team can also keep the **IndoorAtlas magnetic + PDR + BLE-beacon** hybrid as a phase-2 "phone-in-pocket" upgrade — that is the research consensus for actual long-term blind navigation systems.

---

## Source URL Index (for further reading)

### Fiducial markers
- https://docs.opencv.org/4.13.0/d5/dae/tutorial_aruco_detection.html
- https://github.com/AprilRobotics/apriltag
- https://github.com/nasa/AprilNav
- https://github.com/bbenligiray/stag
- https://github.com/opencv/opencv_contrib/issues/2242
- https://arxiv.org/html/2601.07723v1
- https://www.researchgate.net/publication/347154054
- https://www.semanticscholar.org/paper/Determining-and-Improving-the-Localization-Accuracy-Kallwies-Forkel/190a6317ebfbe2c6f29b7684f68a5b5a2104c02c
- https://www.laserscanning-europe.com/en/what-apriltags-what-apriltag-size-should-use
- https://dl.acm.org/doi/10.1145/3807246.3807278
- https://developer.vuforia.com/library/vuforia-engine/FAQ/pricing-and-licensing-options/
- https://www.ptc.com/en/products/vuforia/vuforia-engine/pricing
- https://www.it-jim.com/blog/fiducial-markers-types/
- https://dl.acm.org/doi/10.1145/3793661
- https://www.mdpi.com/2079-9292/15/8/1582
- https://arxiv.org/html/2509.17345v1

### BLE
- https://www.minew.com/bluetooth-5-1-aoa-guide/
- https://www.extronics.com/blog/a-practical-guide-to-setting-up-ble-bluetooth-low-energy-beacons-for-accurate-tracking-in-hazardous-areas/
- https://nextwaves.com/blog/precision-indoors-finding-the-most-accurate-positioning-system-for-large-venues
- https://indoorsnavi.pro/en/aoa/
- https://marvelmind.com/what_is_better_than_ble_for_indoor_positioning_system/
- https://dataintelo.com/report/global-bluetooth-beacon-and-ibeacon-market
- https://locatify.com/ble-beacons-no-bull-beacon-review/
- https://www.wiliot.com/bluetooth-beacon
- https://www.cs.cmu.edu/~NavCog/navcog.html
- https://publications.ri.cmu.edu/storage/publications/2018/01/p270-sato.pdf
- https://www.researchgate.net/publication/335592151
- https://pmc.ncbi.nlm.nih.gov/articles/PMC12305822/
- https://www.wayfindr.net/open-standard
- https://www.itu.int/hub/2020/06/how-the-wayfindr-open-standard-uses-new-tech-to-help-the-visually-impaired/
- https://www.wayfindr.net/how-audio-navigation-works
- https://github.com/Wayfindr-engineering
- https://lighthouseguild.org/a-hybrid-indoor-positioning-system-for-the-blind-and-visually-impaired-using-bluetooth-and-google-tango/

### UWB
- https://www.mdpi.com/2076-3417/14/23/11005
- https://www.mdpi.com/2076-3417/14/13/5646
- https://arxiv.org/html/2410.02329v1
- https://newly.app/sensors/uwb-mobile-apps
- https://www.themobileknowledge.com/news/ios-27-brings-uwb-indoor-navigation-closer-to-commercial-reality/
- https://www.qorvo.com/about/news-events/blogs/qorvo-advances-indoor-navigation-with-uwb-technology
- https://www.qorvo.com/products/wireless-connectivity/ultra-wideband
- https://www.researchgate.net/publication/363219018_Drift-Free_Visual_SLAM_for_Mobile_Robot_Localization_by_Integrating_UWB_Technology
- https://www.nxp.com.cn/company/about-nxp/smarter-world-blog/BL-UWB-EXPANDS-INTO-IOT
- https://pmc.ncbi.nlm.nih.gov/articles/PMC10301813/

### Wi-Fi
- https://developer.android.com/develop/connectivity/wifi/wifi-rtt
- https://arxiv.org/html/2509.03901v1
- https://www.sciencedirect.com/science/article/abs/pii/S0140366425003573
- https://people.csail.mit.edu/bkph/FTMRTT
- https://dl.acm.org/doi/10.1016/j.pmcj.2021.101416
- https://crysp.petsymposium.org/popets/2022/popets-2022-0048.pdf
- https://www.mdpi.com/1424-8220/23/18/7961
- https://www.researchgate.net/publication/320054832

### ARCore / ARKit / VPS
- https://developers.google.com/ar/develop/geospatial
- https://developers.google.com/ar/develop/unity-arf/geospatial/check-vps-availability
- https://developers.google.com/ar/develop/java/geospatial/enable
- https://github.com/google-ar/arcore-unity-extensions/issues/211
- https://trepo.tuni.fi/bitstream/handle/10024/157627/Hakam%C3%A4kiSaku.pdf
- https://info.nianticspatial.com/blog/introducing-lightship-vps-for-web
- https://nianticlabs.com/news/awe-usa-2023
- https://www.nianticspatial.com/campaigns/visual-positioning-system-maps-intro
- https://multiset.ai/visual-positioning-system
- https://developer.apple.com/documentation/arkit/understanding-world-tracking
- https://developer.apple.com/documentation/arkit/argeotrackingstatus/accuracy-swift.enum
- https://developer.apple.com/forums/forums/topics/spatial-computing/spatial-computing-arkit
- https://pmc.ncbi.nlm.nih.gov/articles/PMC9785098/
- https://ar5iv.labs.arxiv.org/html/2207.06780
- https://www.scitepress.org/Papers/2020/89899/89899.pdf
- https://www.researchgate.net/publication/337144233_ARKit_as_indoor_positioning_system
- https://github.com/google-ar/arcore-android-sdk/blob/main/LICENSE
- https://developers.google.com/ar/develop/terms

### SLAM
- https://ar5iv.labs.arxiv.org/html/2007.11898
- https://github.com/UZ-SLAMLab/ORB_SLAM3
- https://www.sciencedirect.com/science/article/pii/S1110016826000475
- https://arxiv.org/html/2212.04745v4
- https://www.researchgate.net/publication/304293832_A_SLAM_Based_Semantic_Indoor_Navigation_System_for_Visually_Impaired_Users
- https://www.scilit.com/publications/0422cbaca36f01811b4759a35db97a0a
- https://pmc.ncbi.nlm.nih.gov/articles/PMC7926395/
- https://www.mdpi.com/1424-8220/24/11/3593
- https://www.researchgate.net/publication/384438218
- https://arxiv.org/html/2404.10140v1
- https://www.emergentmind.com/topics/orb-slam3

### LiDAR
- https://www.sciencedirect.com/science/article/pii/S2666165923000510
- https://www.researchgate.net/publication/370529618
- https://www.vgis.io/2020/04/23/2020-ipad-pro-does-the-lidar-sensor-improve-spatial-tracking/
- https://www.mdpi.com/1424-8220/25/19/6141
- https://www.geoweeknews.com/articles/lidar-is-the-ipad-pros-unexpected-new-feature/

### Magnetic
- https://www.mdpi.com/1424-8220/23/3/1514
- https://ieeexplore.ieee.org/document/6418947/
- https://digital.library.unt.edu/ark:/67531/metadc103371/m2/1/high_res_d/dissertation.pdf
- https://arxiv.org/html/2604.22896v1
- https://www.mdpi.com/1424-8220/22/11/4014
- https://www.researchgate.net/publication/261310654
- https://www.gpsworld.com/indooratlas-announces-geomagnetic-indoor-positioning-service/
- https://www.indooratlas.com/blog/magnetic-positioning-technology-how-it-works/
- https://link.springer.com/chapter/10.1007/978-981-15-8983-6_26
- https://www.indooratlas.com/blog/a-unique-blend-of-augmented-reality-ar-and-indoor-positioning/

### PDR
- https://www.sciencedirect.com/science/article/abs/pii/S1434841123001486
- https://arxiv.org/pdf/2301.03471
- https://www.researchgate.net/publication/340879649
- https://ahmedmansoour.github.io/indoor-positioning-hub/publications/drift-control-pdr-long-period-navigation-smartphone-poses.html
- https://www.nature.com/articles/s41598-025-13390-9
- https://www.mdpi.com/2673-4591/10/1/21
- https://dl.acm.org/doi/10.1145/3696005
- https://dl.acm.org/doi/10.1145/3676522
- https://news.ucsc.edu/2024/10/manduchi-wayfinding-apps/
- https://escholarship.org/content/qt1gv8z9jt/qt1gv8z9jt.pdf?v=lg

### RFID / Talking Signs / NaviLens
- https://www.academia.edu/9062618/A_Blind_Navigation_System_Using_RFID_for_Indoor_Environments
- https://www.researchgate.net/publication/220795837
- https://www.blueiot.com/blog/uwb-vs-ble-vs-wifi-vs-rfid.html
- https://www.ripplesiot.com/indoor-positioning-technology/
- https://www.semanticscholar.org/paper/A-passive-RFID-based-indoor-navigation-system-for-Giampaolo/7354c490398c200d7c742976f4a909c10833fe6e
- https://en.wikipedia.org/wiki/Remote_infrared_audible_signage
- https://www.transit.dot.gov/sites/fta.dot.gov/files/docs/FTA0012_Research_Report_Summary.pdf
- https://www.worldtransitresearch.info/research/1227/
- https://pubmed.ncbi.nlm.nih.gov/12366323/
- https://accessforblind.org/publications/ProjectAction/Transit%20Accessibility%20Improvement%20Through%20Remote%20Infrared%20Signage.pdf
- https://scholarworks.calstate.edu/downloads/m039k852h
- https://www.navilens.com/en
- https://accessibility.calpoly.edu/navilens
- https://galloways.org.uk/navilens/
- https://www.empowerabilities.org/news/navilens-provides-blind-or-low-vision-individuals-invaluable-support
- https://www.navilens.com/en/accessible-qr-code
- https://github.com/sjtrny/freelens/
- https://github.com/nymta/MTA-USDOT-SMART-Grant-Inclusive-Wayfinding-through-NaviLens-/actions
- https://play.google.com/store/apps/details?id=com.neosistec.NaviLens

### Hybrid & system-level
- https://pmc.ncbi.nlm.nih.gov/articles/PMC10301813/
- https://www.mdpi.com/2076-3417/14/13/5646
- https://lighthouseguild.org/a-hybrid-indoor-positioning-system-for-the-blind-and-visually-impaired-using-bluetooth-and-google-tango/
- https://arxiv.org/pdf/2501.15819
- https://www.sciencedirect.com/science/article/pii/S2405844024078563
- https://www.mdpi.com/1424-8220/23/16/7199
- https://www.mdpi.com/1424-8220/19/2/249
- https://www.researchgate.net/publication/MDIRECT-Magnetic_field_strength_and_peDestrIan_dead_RECkoning_based_indoor_localizaTion
- https://ceur-ws.org/Vol-4206/SOCIALIZE-05.pdf
- https://www.tandfonline.com/doi/full/10.1080/10447318.2026.2643358
- https://pmc.ncbi.nlm.nih.gov/articles/PMC7038337/
- https://pmc.ncbi.nlm.nih.gov/articles/PMC9785098/

### Image retrieval / VPS
- https://europe.naverlabs.com/blog/methods-for-visual-localization/
- https://www.scirp.org/journal/paperinformation?paperid=131240
- https://jianfei-cai.github.io/mobile-indoor-localization.pdf
- https://arxiv.org/html/2510.00978v2
- https://openreview.net/forum?id=rmDA02o8MV

### Deployed systems
- https://www.cs.cmu.edu/~NavCog/navcog.html
- https://publications.ri.cmu.edu/navcog-a-navigational-cognitive-assistant-for-the-blind
- https://github.com/microsoft/soundscape
- https://github.com/soundscape-community/soundscape
- https://www.microsoft.com/en-us/research/blog/microsoft-soundscape-new-horizons-with-a-community-driven-approach/
- https://oepatients.org/microsoft-soundscape-2018-a-review-of-what-it-can-do/
- https://doubletaponair.com/soundscape-community-brings-popular-navigation-app-back/
- https://www.applevis.com/blog/microsoft-discontinue-its-soundscape-app-make-code-available-open-source-software
- https://www.navilens.com/en
- https://www.wayfindr.net/open-standard
- https://www.wayfindr.net/how-audio-navigation-works
- https://aira.io/
- https://www.bemyeyes.com/
- https://goodmaps.com/
- https://apps.apple.com/hr/app/goodmaps-indoor-navigation/id6444539843
- https://gnc3.com/review-goodmaps-explore-indoor-navigation-app/
- https://www.blindsquare.com/
- https://www.perkins.org/resource/beginners-guide-blindsquare/
- https://www.ski.org/projects/
- https://www.rightpoint.com/work/lighthouse
- https://dl.acm.org/doi/10.1145/3696005
- https://dl.acm.org/doi/10.1145/3676522
- https://wotipati.github.io/projects/other_papers/MobileHCI2024_SnapAndNav/MobileHCI2024_SnapAndNav.html
- https://www.masakikuribayashi.com/data/project/masaya_kubota_mobilehci_2024/paper.pdf
- https://research.ibm.com/publications/snapandnav-smartphone-based-indoor-navigation-system-for-blind-people-via-floor-map-analysis-and-intersection-detection
- https://techxplore.com/news/2026-08-smartphone-app-vision-users-obstacles.html
- https://www.nature.com/articles/s41551-026-01772-x
- https://navigine.com/blog/indoor-navigation-for-visually-impaired-and-blind-people/
- https://www.mdpi.com/2313-433X/11/1/9

### Other
- https://www.mdpi.com/1424-8220/24/16/5197 (VLC)
- https://arxiv.org/html/2401.13893v1 (VLC survey)
- https://link.springer.com/rwe/10.1007/978-981-97-1522-0_37 (VLC book chapter)
- https://opg.optica.org/abstract.cfm?uri=oe-27-5-7568 (VLC high-accuracy)
- https://www.mdpi.com/1424-8220/24/21/6876 (theories and methods)
- https://www.sciencedirect.com/science/article/pii/S2542660525002665 (smartphone datasets survey)
- https://www.cambridge.org/core/journals/robotica/article/advancing-indoor-positioning-systems-innovations-challenges-and-applications-in-mobile-robotics/179610C4B105839F930B00F8D3EBD7A4
- https://dl.acm.org/doi/10.1145/3191741 (crowdsourcing indoor nav maintenance)
- https://www.semanticscholar.org/paper/Artificial-Markers-A-Comprehensive-Systematic-Review-and-Design (recent survey)
