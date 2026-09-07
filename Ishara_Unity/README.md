# Ishara Unity — Android AR Navigation Client

## Overview

The `Ishara_Unity` project is the mobile client for the **Ishara** assistive navigation platform. It provides visually impaired students with real-time spatial awareness, directional audio cues, and augmented guidance designed to run on Android mobile devices and output an installable **APK / AAB**.

The client interfaces with device camera feeds and computer vision outputs, rendering spatialized navigational feedback without relying on static campus infrastructure.

---

## Features

- **Android Mobile Target**: Optimized for Android smartphones with camera access.
- **Vuforia Engine AR Integration**: Spatial tracking, planar anchoring, and environmental feature extraction.
- **Unity Input System**: Modern event-driven input handling for accessibility gestures and interactions.
- **Spatial Audio Guidance**: Directional acoustic cues indicating obstacles and path clearances (Left, Center, Right).
- **Lightweight UI**: High-contrast, minimal UI designed for low-vision and assistive screen readers.

---

## Prerequisites & Environment Setup

To open, edit, and build the project, ensure you have installed:

1. **Unity Editor**:
   - Unity **2022.3 LTS** or **Unity 6** (Recommended)
   - Modules required:
     - **Android Build Support**
     - **Android SDK & NDK Tools**
     - **OpenJDK**
2. **Vuforia Engine**:
   - Integrated via Unity Package Manager (`com.ptc.vuforia.engine`)
   - Configuration asset located in `Assets/Resources/VuforiaConfiguration.asset`
3. **Android Device**:
   - Android 9.0 (API Level 28) or higher
   - Camera permissions enabled
   - Developer Mode and USB Debugging enabled

---

## Project Structure

```
Ishara_Unity/
├── Assets/
│   ├── Editor/                      # Editor scripts & migration utilities
│   ├── Resources/                   # VuforiaConfiguration and runtime assets
│   ├── Scenes/
│   │   └── SampleScene.unity        # Main AR navigation scene
│   └── InputSystem_Actions.inputactions # Input system bindings
├── Packages/
│   └── manifest.json                # Unity package dependencies
├── ProjectSettings/                 # Build settings, quality, and Android player configs
└── README.md                        # This build guide
```

---

## Building the Android APK

Follow these steps to produce the final `.apk` file:

### 1. Open the Project in Unity Hub
1. Open **Unity Hub**.
2. Click **Add** ➔ **Add project from disk**.
3. Select the `Ishara_Unity` folder.
4. Open the project with the corresponding Unity Editor version.

### 2. Switch Platform to Android
1. Go to **File ➔ Build Settings...**
2. In the **Platform** list, select **Android**.
3. If not already active, click **Switch Platform**.
4. Ensure `Scenes/SampleScene.unity` is checked in the **Scenes In Build** list.

### 3. Configure Player Settings
1. Click **Player Settings...** in the bottom-left of the Build Settings window.
2. Under **Player ➔ Other Settings**:
   - **Package Name**: e.g., `com.ishara.navigation`
   - **Minimum API Level**: Android 9.0 (API Level 28)
   - **Target API Level**: Automatic (highest installed) or Android 14 (API Level 34)
   - **Scripting Backend**: **IL2CPP** (Required for 64-bit builds)
   - **Target Architectures**: Check **ARM64**
3. Under **XR Plug-in Management / Vuforia Engine**:
   - Verify Vuforia license key in `Assets/Resources/VuforiaConfiguration.asset`.

### 4. Build the APK
1. In the **Build Settings** window, click **Build** (or **Build and Run** if a device is connected via USB).
2. Choose an output directory (e.g., `Builds/Ishara_v1.0.apk`).
3. Note: The `Builds/` folder is automatically ignored by `.gitignore` to keep the repository lightweight.

---

## Testing & Deployment

- **Direct Install**: Transfer the `.apk` file to your Android phone and install via your file manager (allow *Install from Unknown Sources* if prompted).
- **ADB Command**:
  ```bash
  adb install -r Builds/Ishara_v1.0.apk
  ```
- **Live Debugging**:
  ```bash
  adb logcat -s Unity
  ```
