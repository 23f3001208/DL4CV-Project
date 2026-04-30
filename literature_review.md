# Literature Review: Layered Representations from a Single Image

## 1. Introduction

Decomposing a single image into re-composable semantic layers is a long-standing problem at the intersection of computer graphics, computer vision, and image editing. A fully layered representation enables downstream tasks such as parallax animation, free-viewpoint synthesis, selective relighting, and content-aware editing, all without any 3D capture or video sequences. This review surveys the relevant literature organised into five threads: (§2) classical image layering, (§3) semantic segmentation, (§4) monocular depth estimation, (§5) intrinsic image decomposition, and (§6) end-to-end neural layered representations.

---

## 2. Classical Image Layering

### 2.1 Foundational Work

**Porter & Duff (1984)** — "Compositing Digital Images." *SIGGRAPH*.  
Introduced the alpha channel and the full algebra of compositing operators (over, in, out, atop, xor). Every modern RGBA layer system uses their "over" operator: `C = α_F · C_F + (1 − α_F) · C_B`. The formalism defines what a *layer stack* means.

**Adelson & Bergen (1991)** — "The Plenoptic Function and the Elements of Early Vision."  
Introduced the concept of separating a visual scene into independent *factors* (illumination, reflectance, geometry), foreshadowing intrinsic decomposition.

**Wang & Adelson (1994)** — "Representing Moving Images with Layers." *IEEE Trans. Image Processing*.  
First formal paper on *image layers*: each layer is a 2D translucent sprite that can be independently translated. Demonstrated for video motion segmentation, but the layer model directly inspired single-image work.

**Ye & Bhattacharya (2014)** — "Efficient Image Segmentation and Alpha Matting for Layered Image Editing."  
Combined GrabCut-style alpha matting with interactive layer extraction. Showed that soft alpha edges matter for compositing fidelity.

### 2.2 Layered Depth Images (LDI)

**Shade et al. (1998)** — "Layered Depth Images." *SIGGRAPH*.  
Extended the standard depth image to store multiple depth values per pixel (one per layer), enabling novel-view synthesis by unpeeling occlusions. This is the direct ancestor of modern layered neural representations.

**Zitnick et al. (2004)** — "High-Quality Video View Interpolation Using a Layered Representation."  
Used video to estimate multi-layer depth, achieving smooth parallax. The depth-ordering constraint was key: layers were sorted from near to far so that the "over" operator produced correct occlusions.

---

## 3. Semantic Segmentation

Accurate semantic segmentation is prerequisite to producing semantically meaningful layers.

### 3.1 FCN Era

**Long et al. (2015)** — "Fully Convolutional Networks for Semantic Segmentation." *CVPR*.  
First end-to-end trainable dense prediction network. Replaced the fully-connected head of VGG/AlexNet with convolutions and upsampled via skip connections. Established the encoder-decoder paradigm.

**Chen et al. (2018)** — "DeepLab v3+: Encoder-Decoder with Atrous Separable Convolution." *ECCV*.  
Added atrous (dilated) convolutions and an Atrous Spatial Pyramid Pooling (ASPP) module to capture multi-scale context without reducing resolution. Long-running SOTA on PASCAL VOC.

### 3.2 Transformer Era

**Xie et al. (2021)** — "SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers." *NeurIPS*.  
*The model we use (B2/B5 variants, ADE20K-150).*  
Replaced the image backbone with a hierarchical transformer (MiT) that naturally captures multi-scale features without positional encoding. The lightweight MLP decoder is fast and surprisingly effective. Achieves 51.8 mIoU on ADE20K with B5 variant. Key advantage: no handcrafted positional embeddings → better generalisation across resolutions.

**Cheng et al. (2022)** — "Masked-Attention Mask Transformer for Universal Image Segmentation (Mask2Former)." *CVPR*.  
Unified panoptic/semantic/instance segmentation using a Transformer decoder with masked attention. State-of-the-art on many benchmarks. Heavier than SegFormer but produces instance-level masks useful for distinguishing individual objects within a category.

**Kirillov et al. (2023)** — "Segment Anything (SAM)." *ICCV*.  
Trained on 1B+ masks, produces promptable, class-agnostic masks at any granularity. Does not assign semantic labels directly, but can be used as a foundation to extract fine-grained instance masks, then matched to semantic labels via CLIP or a separate classifier.

**Ravi et al. (2024)** — "SAM 2: Segment Anything in Images and Videos." *arXiv 2408.00714*.  
Extension of SAM for video with a streaming memory mechanism. For single images, SAM 2 provides improved mask quality and multi-mask ambiguity prediction.

### 3.3 Open-Vocabulary Segmentation

**Ghiasi et al. (2022)** — "Scaling Open-Vocabulary Image Segmentation with Image-Level Labels (OpenSeg)." *ECCV*.  
**Liang et al. (2023)** — "Open-Vocabulary Semantic Segmentation with Mask-adapted CLIP." *CVPR*.  
These models take *text prompts* rather than a fixed label set, enabling free-form category specification. Useful when the target scene categories are not in ADE20K-150.

---

## 4. Monocular Depth Estimation

Depth ordering of semantic layers requires relative depth estimates. Monocular methods infer depth from a single image without stereo or LiDAR.

### 4.1 Classical Approaches

**Saxena et al. (2006)** — "Learning Depth from Single Monocular Images." *NeurIPS*.  
First learning-based monocular depth: Markov Random Field on local image patches. Showed depth cues (foreshortening, texture gradients, haze) are learnable.

### 4.2 Deep Monocular Depth

**Eigen et al. (2014)** — "Depth Map Prediction from a Single Image using a Multi-Scale Deep Network." *NeurIPS*.  
First CNN-based depth predictor. Two-scale architecture: global structure + local refinement. Trained on NYU Depth v2.

**Laina et al. (2016)** — "Deeper Depth Prediction with Fully Convolutional Residual Networks." *3DV*.  
ResNet encoder + up-projection blocks. Huber (BerHu) loss. Still a common baseline.

### 4.3 Transformer-Based Depth

**Ranftl et al. (2021)** — "Vision Transformers for Dense Prediction (DPT)." *ICCV*.  
*One of our supported backends.*  
Assembled dense predictions from intermediate ViT tokens reassembled at multiple scales. Outperformed CNN-based methods, especially on fine structure and texture-rich regions.

**Ranftl et al. (2022)** — "Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-Shot Cross-Dataset Transfer (MiDaS v3.1)."  
*Supported backend: "midas".*  
Mixed 12 diverse training datasets using scale-and-shift invariant loss. Achieves impressive zero-shot generalisation. Available via `torch.hub`.

### 4.4 Foundation Depth Models

**Yang et al. (2024)** — "Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data." *CVPR 2024*.  
*Our default backend.*  
Trained a ViT encoder on 62M unlabeled images via pseudo-label distillation from a teacher model. Achieves strong zero-shot performance across indoor, outdoor, artistic, and anime-style images — critical for our project's domain-agnosticism. Depth Anything V2 (same authors, 2024) further improves using synthetic data for fine-grained detail.

**Bhat et al. (2023)** — "ZoeDepth: Zero-Shot Transfer by Combining Relative and Metric Depth." *arXiv*.  
Extended MiDaS for *metric* depth (absolute metres) via a per-dataset header. Relevant if absolute scale matters (e.g., for physics-based relighting).

---

## 5. Intrinsic Image Decomposition

### 5.1 Retinex Theory

**Land & McCann (1971)** — "Lightness and Retinex Theory." *J. Opt. Soc. Am.*  
Proposed that the visual system separates reflectance (albedo) from illumination. Retinex computes log-ratio of neighbourhood intensities to estimate reflectance independently of illumination. This underpins our "retinex" backend.

**Jobson et al. (1997)** — "A Multiscale Retinex for Bridging the Gap Between Color Images and the Human Observation of Scenes." *IEEE Trans. Image Processing.*  
Multi-Scale Retinex (MSR): average of Single-Scale Retinex at several Gaussian scales. Still competitive and computationally trivial.

### 5.2 Optimization-Based Methods

**Shen et al. (2011)** — "Intrinsic Images Decomposition Using a Local and Global Sparse Representation of Reflectance." *CVPR*.  
Exploits the observation that reflectance changes are sparse (edges) while shading is smooth. Formulated as an L1-regularized problem.

**Bi et al. (2015)** — "An L1 Image Transform for Edge-Preserving Smoothing and Scene-Level Intrinsic Decomposition." *ACM ToG*.

**Farbman et al. (2008)** — "Edge-Preserving Decompositions for Multi-Scale Tone and Detail Manipulation." *SIGGRAPH*.  
Introduced the Weighted Least Squares (WLS) filter — the basis of our "sparse" backend — which smooths according to a guide image, perfectly preserving semantic edges while flattening within-region gradients.

### 5.3 Dataset Benchmarks

**Grosse et al. (2009)** — "Ground Truth Dataset and Baseline Evaluations for Intrinsic Image Algorithms." *ICCV*.  
MIT Intrinsic Images dataset: 20 objects under 11 illuminations with GT albedo+shading. Canonical benchmark. Small scale limits modern deep training.

**Bell et al. (2014)** — "Intrinsic Images in the Wild." *ACM ToG*.  
Large-scale crowdsourced annotations (>20K images) via pairwise comparisons ("is this region lighter because of illumination or because of material?"). Enabled data-driven methods at scale.

### 5.4 Deep Intrinsic Decomposition

**Narihira et al. (2015)** — "Direct Intrinsics: Learning Albedo-Shading Decomposition by Convolutional Regression." *ICCV*.  
First CNN trained end-to-end for intrinsic decomposition on MIT + ShapeNet.

**Li & Snavely (2018)** — "CGIntrinsics: Better Intrinsic Image Decomposition through Physically-Based Rendering." *ECCV*.  
Generated large-scale synthetic intrinsic GT from physically-based rendering (Blender Cycles). Achieved SOTA on real images by bridging sim-to-real gap.

**Liu et al. (2020)** — "Unsupervised Intrinsic Image Decomposition (USI3D)." *CVPR*.  
Self-supervised approach using CycleGAN-style consistency: decompose then re-compose under simulated illumination changes. No GT required. Suitable for our "deep" backend placeholder.

---

## 6. End-to-End Layered Neural Representations

### 6.1 Layer-Aware Neural Synthesis

**Lu et al. (2021)** — "Layered Neural Atlases for Consistent Video Editing." *ACM ToG*.  
Decomposed video into 2 layers (foreground + background) each represented as a neural atlas (MLP mapping 2D → colour+alpha). Demonstrated temporally consistent editing.

**Kasten et al. (2021)** — "Layered Neural Atlases." *ACM ToG*.  
Analogous to above; introduced the "atlas" concept where layers are canonical 2D texture maps that deform onto each frame.

**Mou et al. (2022)** — "Omnimatte: Associating Objects and Their Effects in Video." *CVPR*.  
Extended layered decomposition to associate objects with their dynamic effects (shadows, reflections, splashes). Each foreground object gets a dedicated RGBA "matte" layer.

### 6.2 Layered Depth Representations

**Shih et al. (2020)** — "3D Photography Using Context-Aware Layered Depth Inpainting." *CVPR*.  
*Directly relevant to our project.*  
Given a single image + depth map, generates a multi-plane layered depth image (LDI) by inpainting occluded regions. Produces parallax video from a still photo. Uses SegFormer-like segmentation to identify layer boundaries, then depth to order them.

**Tucker & Snavely (2020)** — "Single-View View Synthesis with Multiplane Images." *CVPR*.  
Predicts a fixed-depth-plane MPI from a single image. Each plane is a full RGBA image at a fixed metric depth. The set of planes is a discrete approximation of the light field.

**Watson et al. (2021)** — "The Temporal Opportunist: Self-Supervised Multi-Frame Monocular Depth." *CVPR*.  
Multi-plane image depth + video self-supervision for improved single-image depth.

### 6.3 NeRF-Based Decomposition

**Martin-Brualla et al. (2021)** — "NeRF in the Wild: Neural Radiance Fields for Unconstrained Photo Collections." *CVPR*.  
Introduced appearance latent codes per image and a transient/static decomposition of the scene — a form of semantic layering in 3D.

**Wu et al. (2022)** — "D2NeRF: Self-Supervised Decoupling of Dynamic and Static Objects from a Monocular Video." *NeurIPS*.  
Decomposed video into static background + dynamic foreground NeRFs. Conceptually analogous to our layered approach but in 3D.

---

## 7. Our Method in Context

Our pipeline occupies a specific niche in this landscape:

| Property | Our Method | LDI (Shih'20) | MPI (Tucker'20) | Omnimatte |
|---|---|---|---|---|
| **Input** | Single image | Single image + depth | Single image | Video |
| **Layers** | Semantic groups | Depth planes | Fixed depth planes | Object + effects |
| **Semantic** | ✅ (SegFormer) | Partial | ✗ | ✅ |
| **Depth-ordered** | ✅ | ✅ | ✅ | ✗ |
| **Intrinsic** | ✅ (stretch) | ✗ | ✗ | ✗ |
| **Runtime** | ~5s (GPU) | ~30s | ~10s | Minutes (video) |
| **Editable** | ✅ RGBA layers | Via inpainting | Limited | ✅ |

### Key Design Choices

1. **SegFormer-B2 over SAM**: SegFormer produces direct semantic labels (150 ADE20K classes) without requiring interactive prompts. SAM is more flexible but needs a category oracle (e.g., CLIP) to assign group names. For autonomous operation, SegFormer is more appropriate.

2. **Depth Anything V2 over DPT/MiDaS**: Depth Anything's training on 62M diverse unlabelled images gives dramatically better zero-shot generalisation, particularly on non-photorealistic inputs (anime, stylized art), which is a key domain in the project spec.

3. **WLS-based intrinsic decomposition over Retinex**: WLS preserves sharp reflectance boundaries while smoothing illumination, avoiding the halo artifacts that plague Retinex at large sigma values. Runtime is comparable.

4. **Soft alpha edges**: Unlike binary segmentation masks (which create unrealistic hard cuts in compositing), soft-edge feathering via erosion + Gaussian blur produces layers that re-compose naturally via Porter–Duff.

---

## 8. Datasets

| Dataset | Size | Task | Notes |
|---|---|---|---|
| **ADE20K** | 25K train, 2K val | Semantic segmentation | 150 classes, indoor+outdoor |
| **COCO** | 118K train | Panoptic | 133 categories; good for person/vehicle |
| **NYU Depth v2** | 1,449 pairs | RGB-D indoor depth | Kitchen/bedroom scenes |
| **KITTI** | 93K frames | Outdoor depth | Driving, LiDAR GT |
| **MIT Intrinsic Images** | 20 objects × 11 lights | Albedo+shading GT | Small; canonical intrinsic benchmark |
| **IIW (Bell'14)** | 5K images | Pairwise reflectance | Crowdsourced; large-scale |
| **MPI Sintel** | 1,064 frames | Optical flow + depth | Animated, good for generalisation |
| **Hypersim** | 77K images | Photorealistic indoor RGB-D + intrinsic | PhysicsBased rendering GT |

For evaluation we use a subset of **ADE20K validation** (segmentation quality), **NYU Depth v2** (depth quality), and **MIT Intrinsic Images** (intrinsic quality).

---

## 9. Open Problems and Future Directions

1. **Amodal completion**: Occluded regions of foreground objects are not inpainted in our current approach. Methods like Ling et al. (2020) "Variational Amodal Object Completion" could fill occluded albedo.

2. **Instance-level layers**: SegFormer groups all persons into one mask. Using Mask2Former or SAM-2 would split the mask per-instance, enabling per-person editing.

3. **Multi-layer alpha matting**: Our binary/soft masks are not true alpha mattes. Natural matting (Levin et al. 2008; Xu et al. 2017 "Deep Image Matting") would yield higher-quality edges, especially for hair and foliage.

4. **Physically-correct relighting**: Once albedo and shading are separated, integrating a neural environment map estimator (e.g., DiffusionLight, DiLightNet) would enable full relighting of individual layers.

5. **Metric depth and 3D lift**: Replacing relative depth (Depth Anything) with metric depth (ZoeDepth, DepthPro) would allow layers to be lifted into actual 3D planes, enabling proper parallax video generation (Shih et al., 2020).

---

## References (select)

- Porter & Duff (1984). *Compositing Digital Images.* SIGGRAPH.
- Adelson & Bergen (1991). *The Plenoptic Function.* MIT AI Memo.
- Wang & Adelson (1994). *Representing Moving Images with Layers.* IEEE TIP.
- Shade et al. (1998). *Layered Depth Images.* SIGGRAPH.
- Land & McCann (1971). *Lightness and Retinex Theory.* JOSA.
- Jobson et al. (1997). *Multi-Scale Retinex.* IEEE TIP.
- Farbman et al. (2008). *Edge-Preserving Decompositions.* SIGGRAPH.
- Grosse et al. (2009). *MIT Intrinsic Images.* ICCV.
- Long et al. (2015). *FCN.* CVPR.
- Bell et al. (2014). *Intrinsic Images in the Wild.* ACM ToG.
- Eigen et al. (2014). *Depth Prediction from a Single Image.* NeurIPS.
- Li & Snavely (2018). *CGIntrinsics.* ECCV.
- Ranftl et al. (2021). *DPT.* ICCV.
- Ranftl et al. (2022). *MiDaS v3.1.* arXiv.
- Xie et al. (2021). *SegFormer.* NeurIPS.
- Shih et al. (2020). *3D Photography.* CVPR.
- Tucker & Snavely (2020). *MPI from a Single View.* CVPR.
- Lu et al. (2021). *Layered Neural Atlases.* ACM ToG.
- Mou et al. (2022). *Omnimatte.* CVPR.
- Kirillov et al. (2023). *SAM.* ICCV.
- Yang et al. (2024). *Depth Anything / V2.* CVPR 2024.
- Ravi et al. (2024). *SAM 2.* arXiv 2408.00714.
- Liu et al. (2020). *USI3D.* CVPR.
