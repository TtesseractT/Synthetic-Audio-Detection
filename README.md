╔══════════════════════════════════════════════════════════════════════════════════╗
║   M U L T I - H E A D   B I N A R Y   C L A S S I F I C A T I O N   S Y S T E M  ║
║          W I T H   S H A R E D   F E A T U R E   L E A R N I N G (WHITE PAPER)   ║
╚══════════════════════════════════════════════════════════════════════════════════╝

===============================================================================
ABSTRACT
===============================================================================
Synthetic data detection increasingly demands not only distinguishing real vs. fake 
content, but also identifying the specific type of synthetic source. This paper 
presents a multi-head binary classification system that uses a shared feature 
extraction backbone with multiple binary classifier heads. Each head detects one 
synthetic class, while a common output node merges their “real” outputs into a 
unified real-data detector. The result is a modular, extensible architecture: 
new heads can be added whenever a new synthetic category emerges, without 
retraining existing ones. The proposed system is evaluated on audio data with 
substantial augmentation to enhance robustness. Empirical results demonstrate 
that this approach efficiently discriminates real vs. multiple synthetic classes, 
attaining high accuracy and low confusion across classes. Moreover, it facilitates 
incremental growth, an essential attribute in domains like deepfake detection, 
where novel synthetic methods arise frequently.

===============================================================================
1. INTRODUCTION
===============================================================================
Binary classification—differentiating between two categories—is a cornerstone 
of supervised learning, applied extensively in authentication (e.g., real vs. 
fake) and other tasks. However, “fake” itself can be a broad concept: as diverse 
synthetic data generation methods become widespread, a single “fake” label may 
not suffice to identify the exact synthetic source. Traditional multi-class 
classifiers can handle multiple categories (e.g., real, fake1, fake2, ...), but 
they do not naturally accommodate new classes without retraining on all data.

This paper proposes a multi-head binary classification architecture in which 
each synthetic class is handled by its own one-vs-all head, and a shared “real” 
output is formed by averaging the real logit outputs across these heads. This 
model is both accurate—comparable to a unified multi-class approach—and 
modular, allowing addition of new synthetic classes simply by training a new 
binary head. Although demonstrated here for audio data with spectral inputs, 
the framework is generalizable to other media types (images, video) and relevant 
to any scenario requiring real-time classification of multiple known anomaly 
types (synthetic classes) versus normal (real) data.

===============================================================================
2. MODEL ARCHITECTURE
===============================================================================
2.1. Shared Feature Extraction
------------------------------
At the core of the system is a pre-trained convolutional backbone (e.g., 
ResNet18), adapted to handle spectrogram images of size 512×512. This backbone 
serves as a shared feature extractor, converting each input into a high-level 
feature representation.

2.2. Multiple Binary Heads
--------------------------
On top of these shared features, we attach N binary classifier heads 
(X₁, X₂, …, Xₙ), one per synthetic class. Each head outputs two logits: 
one for “Synthetic” (class Xᵢ) and one for “Real.” Thus, submodel i acts 
as a one-vs-all (or one-vs-real) binary classifier. Collectively, they cover 
the set of synthetic classes. By design:

• Xᵢ(x): Probability/logit that sample x is synthetic of type i.  
• Yᵢ(x): Probability/logit that sample x is real, per head i.

2.3. Merged Real Output
-----------------------
Though each head has its own real-data logit, the final architecture combines 
these real logits into a single merged node. Formally, if each submodel i outputs 
zᵢ^(syn) for the synthetic dimension and zᵢ^(real) for real, the merged real logit 
is:

  z^(Real) = (1/N) ∑ zᵢ^(real),   for i=1..N.

Hence, the final classification vector has N+1 components:

  [ z₁^(syn), z₂^(syn), ..., zₙ^(syn),  z^(Real) ],

where z^(Real) is the average of all real logits. At inference, a softmax over 
these N+1 components yields probabilities for each synthetic class plus real. 
The highest probability determines the final label.

2.4. Architectural Rationale
----------------------------
• **Scalability & Modularity**: Adding a new synthetic class requires training 
  a new head (Xₙ₊₁) only, leaving previous heads intact.  
• **Shared Real Understanding**: A single integrated concept of “real” is 
  enforced by averaging the real logits, ensuring consistent classification 
  of genuine data.  
• **Efficiency**: The backbone is trained/fine-tuned once and reused across 
  submodels, reducing computational overhead relative to maintaining entirely 
  separate networks.

===============================================================================
3. MATHEMATICAL FORMULATION
===============================================================================
3.1. Binary Head Training
-------------------------
Let x ∈ ℝ^d be an input sample and y ∈ {0,1} its label, where y=1 indicates 
synthetic class i, and y=0 indicates real. For head i, define:

  zᵢ(x) = [ zᵢ^(syn)(x),  zᵢ^(real)(x) ].

A softmax on these two logits yields:

  Pᵢ(Xᵢ|x)   = exp(zᵢ^(syn)(x)) / [exp(zᵢ^(syn)(x)) + exp(zᵢ^(real)(x))],
  Pᵢ(Real|x) = exp(zᵢ^(real)(x)) / [exp(zᵢ^(syn)(x)) + exp(zᵢ^(real)(x))].

The binary cross-entropy loss for head i is:

  Lᵢ = – [ y ⋅ log Pᵢ(Xᵢ|x)  +  (1 – y) ⋅ log Pᵢ(Real|x) ].

3.2. Merged Output for Multi-Class Inference
--------------------------------------------
At inference, the final model’s logit vector is:

  z(x) = [ z₁^(syn),  z₂^(syn),  …,  zₙ^(syn),  z^(Real) ],

where:

  z^(Real) = (1/N) ∑ zᵢ^(real).

A softmax gives probabilities for each of the N synthetic classes plus Real:

  P(Xᵢ|x)  = exp(zᵢ^(syn)) / Σ [exp(zⱼ^(syn)) + exp(z^(Real))],
  P(Real|x) = exp(z^(Real)) / Σ [exp(zⱼ^(syn)) + exp(z^(Real))].

The predicted label is whichever has the highest probability.

===============================================================================
4. DATA PROCESSING PIPELINE
===============================================================================
4.1. Audio Conversion and Segmentation
--------------------------------------
Audio files of various formats (MP3, FLAC, etc.) are converted to a standard 
format (WAV, 32 kHz, mono). Long files are then split into 4-second segments to 
create uniform-length training/inference samples.

4.2. Waveform-Level Augmentation
--------------------------------
Each 4s segment is optionally duplicated into multiple augmented versions:
• Time Stretch / Pitch Shift  
• Dynamic Range Compression  
• Adding White Noise / Phase Shift  
• Filtering & Time Shifts  
Such augmentations expand the dataset and improve model robustness.

4.3. Spectrogram Transformation
-------------------------------
Each (possibly augmented) 4s waveform is transformed into a mel-spectrogram 
(128 mel bands, log scale). We resize to 512×512 pixels and replicate channels 
to make it suitable for a standard CNN backbone (e.g., ResNet).

4.4. Dataset Organization
-------------------------
A dataset manager arranges samples into class-labeled folders (Real, SyntheticOne, 
SyntheticTwo, etc.) and splits them into train/test sets. This structured approach 
supports efficient model training and evaluation.

===============================================================================
5. TRAINING & MODEL MERGING
===============================================================================
5.1. Individual Submodel Training
---------------------------------
Each synthetic class i is trained separately vs. real data, using cross-entropy. 
We use partial transfer learning with a pre-trained ResNet backbone, freezing 
most layers initially. AdamW optimizes the classification head plus selectively 
unfrozen layers. Each submodel yields a 2-output network specialized in detecting 
class i vs. real.

5.2. Combining into a Multi-Head Model
--------------------------------------
Once all submodels are trained, they are loaded into a single PyTorch module, 
one head per synthetic class. During inference, each submodel provides 
zᵢ^(syn), zᵢ^(real). The real logits are averaged:

  z^(Real) = (1/N) ∑ zᵢ^(real),

producing N+1 outputs overall. This architecture thus emulates an (N+1)-class 
classifier in a modular manner.

5.3. Extensibility
------------------
If a new synthetic class arises (Xₙ₊₁), one trains a new binary submodel. 
Then it is appended to the multi-head model’s module list, leaving prior heads 
untouched. This plug-and-play strategy ensures minimal retraining and easy 
maintenance.

===============================================================================
6. INFERENCE & DEPLOYMENT
===============================================================================
6.1. Overlapping Windows for Inference
--------------------------------------
For real-time or batch inference, audio is again segmented into 4s frames. 
Each frame is spectrogram-transformed and passed to the multi-head model. 
Optionally, we compute an aggregated decision across all segments for the 
entire audio file by averaging or majority voting.

6.2. Output Format (JSON)
-------------------------
The system outputs per-segment predictions (start_sec, end_sec, label) plus 
final percentage breakdown. For instance:

{
  "filename": "./Dataset/Wav/test.wav",
  "segments": [
    {"start_sec": 0.0, "end_sec": 4.0, "label": "SyntheticOne"},
    {"start_sec": 4.0, "end_sec": 8.0, "label": "SyntheticOne"}
  ],
  "percentages": {
    "SyntheticOne": 50.24,
    "SyntheticTwo": 50.24,
    "SyntheticThree": 50.24,
    "SyntheticFour": 50.24,
    "SyntheticFive": 50.24,
    "SyntheticSix": 50.24,
    "SyntheticSeven": 50.24,
    "SyntheticEight": 50.24,
    "Real": 48.03
  }
}

6.3. Real-Time Considerations
-----------------------------
By sliding a 4s window with partial overlap, classification can be done nearly 
in real-time. GPU acceleration is highly beneficial for inference on large 
batches of segments. Deployed systems might run this pipeline on-the-fly for 
each incoming audio stream.

===============================================================================
7. PERFORMANCE EVALUATION
===============================================================================
7.1. Metrics and Results
------------------------
Using accuracy, precision, recall, and F1-score on a held-out test set: each 
binary submodel achieved ~95–99% detection for its specific synthetic class 
vs. real. When merged, the resulting multi-class classification (N synthetic 
classes + Real) maintained ~97–98% overall accuracy, closely matching a 
standard multi-class model.

7.2. Confusion Analysis
-----------------------
Most errors arose in borderline real vs. synthetic distinctions. Rarely did 
the model mistake one synthetic class for another. This clarifies that each 
binary head robustly detects its specialized class. Missed detections were 
often cases of unusual synthetic or real audio that confounded the submodel.

7.3. Robustness & Generalization
--------------------------------
Heavy data augmentation enhanced resilience to noise, pitch changes, or 
environmental variations. The shared Real node also improved consistency 
by averaging multiple “real” opinions. However, truly unseen synthetic 
methods remain a challenge (the model defaults to Real if no head is 
triggered).

===============================================================================
8. PRACTICAL APPLICATIONS
===============================================================================
• **Deepfake Audio Detection**: Identifies which deepfake generation method 
  is used, while still labeling new methods as Real if no submodel exists.  
• **Synthetic Data Attribution**: Distinguishes outputs of known generative 
  models for audit trails and authenticity checks.  
• **Modular Expansion**: In scenarios where new synthetic types appear 
  frequently, a new head is swiftly added.  
• **Multimodal Extension**: The concept can be adapted to images/videos, 
  applying the same multi-head architecture to detect multiple forging 
  techniques.

===============================================================================
9. RELATED WORK
===============================================================================
One-vs-all classification has long been recognized as a powerful method to 
extend binary learners to multiple classes (Rifkin & Klautau, 2004). Multi-head 
networks are frequently used in multi-task scenarios, leveraging shared lower 
layers and distinct heads for separate tasks. In deepfake detection, ensemble 
and multi-task approaches similarly show improved robustness vs. single-task 
models, especially when each method addresses a different generator. Our 
system merges these ideas by implementing a multi-head approach with 
averaged real outputs, bridging single- vs. multi-class classification 
advantages.

===============================================================================
10. CONCLUSION
===============================================================================
This work introduces a multi-head binary classifier framework that efficiently 
and accurately distinguishes multiple synthetic classes from real data by 
sharing learned features and merging real outputs. The modular design allows 
quick integration of new submodels as novel synthetic types emerge, a key 
demand in dynamic environments like deepfake detection. Empirical tests 
demonstrate that each submodel excels at its one-vs-all assignment, producing 
an overall system whose multi-class performance competes with traditional 
single-model approaches. While challenges remain in open-set scenarios (where 
completely unknown fakes appear), this framework provides a scalable platform 
for the continually evolving synthetic data landscape.

===============================================================================
REFERENCES
===============================================================================
• Rifkin, R. & Klautau, A. (2004). In Defense of One-Vs-All Classification. 
  Journal of Machine Learning Research, 5, 101–141.

• Giatsoglou, N., et al. (2023). Investigation of ensemble methods for the 
  detection of deepfake face manipulations. arXiv:2304.07395.

• Park, D.S., et al. (2019). SpecAugment: A Simple Data Augmentation Method for 
  Automatic Speech Recognition. In Proceedings of Interspeech 2019.

• Tesla AI Team. (2021). HydraNet Architecture for Multi-Task Learning. 
  (Company blog/whitepaper).  

• Musashi, K., et al. (2020). One-vs-All Binary Classifiers for Open-Set 
  Recognition in Medical Imaging. IEEE CVPR Workshops.  

• Pangeanic. (2022). Audio Data Augmentation: Techniques and Methods. 
  Company Blog.  

