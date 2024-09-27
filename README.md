<<<<<<< HEAD
#  CAPTURE:Towards Cost-Efficient Vulnerability Detection with Cross-Modal Adversarial Reprogramming



In the realm of cybersecurity, software vulnerabilities present a persistent threat, with disclosed vulnerabilities having significantly increased over the past decade. Despite the advances in deep learning (DL) that have spurred data-driven approaches to detect these vulnerabilities, challenges particularly in the efficacy under vulnerable sample scarcity and the high cost of model training persist. Targeting these two major challenges, which are generally overlooked by existing DL-based vulnerability detectors, this paper introduces CAPTURE, a novel Crossmodal Adversarial reProgramming approach Towards costefficient vUlneRability dEtection, with significantly reduced need of vulnerable samples and training time. Specifically, CAPTURE firstly performs normalization, lexical parsing and linearization on the AST of the source code to obtain structure- and type-aware token sequences. On this basis, token embeddings sourced from a learnt universal perturbation dictionary are reshaped and reorganized, producing a perturbation image for the token sequences. By superimposing the perturbation image onto a regular image, the original model trained for classifying the regular image can then be repurposed to support code vulnerability detection, with a dynamic label remapping scheme that reassigns the original labels to a vulnerable or non-vulnerable label. Our evaluations across multiple vulnerability datasets demonstrate that CAPTURE achieves comparable detection accuracy to state-of-the-art methods while significantly enhancing training efficiency due to its minimal quantity of parameters to update during the model training. Particularly, it obviously excels the comparison methods in terms of detection accuracy and F1 scores in scenarios with limited number of vulnerable samples.



## Design of CAPTURE



![Design](C:\Users\sangui\Desktop\Design.png)

**Capture proceeds with the following main steps:**

- Data Preprocessing: which converts the given code snippets into structure-aware token sequences as inputs;

- Input Transformation:we introduce an input transformation function that converts code sequences into a representation compatible with image classifiers by incorporating trainable universal perturbations.
- Non-Linear Processing: Through the non-linear interactions of adversarial reprogramming, these perturbations are translated into effective model parameters.
- Label Remapping:Inspired by the dual-layer optimization strategy, where the upstream task adjusts its strategy based on the downstream task's response, we design a dynamic label mapping (DLM) scheme



## Dataset

We compare our approach with state-of-the-art methods in the experiments, including Baseline-BiLSTM, Baseline-TextCNN, VulCNN, SySeVR, Reveal, and CodeBERT. Two real-world datasets are used: D2A and Devign.

D2A(https://github.com/ibm/D2A) is a real-world vulnerability dataset created by the IBM Research team. It includes several open-source software projects such as FFmpeg, httpd, Libav, LibTIFF, Nginx, and OpenSSL.Reveal

Devign(https://dl.acm.org/doi/10.5555/3454287.3455202) is a dataset comprising function-level C/C++ source code from projects QEMU and FFmpeg. It has been manually annotated and verified by security researchers to represent real-world scenarios.

## The required environment for this project

torch                      1.13.1
transformers        4.27.4            
tree-sitter              0.20.1
torch                      1.13.1               
torchvision             0.14.1
timm                      0.3.0
tqdm                      4.65.0	



## Source Code

The text/sequence dataset configurations are defined in `data_utils.py`. We can  use our custom datasets (defined as csv files) with the same API. To reprogram an image model for a text classification task run:

python FLM_reprogramming.py[ILM_reprogramming.py、RLM_reprogramming.py]  --text_dataset TEXTDATSET --logdir <PATH WHERE /LOG WILL BE SAVED> --cache_dir --reg_alpha 1e-3 --img_patch_size 16 --vision_model vit_base_patch16_384 ;

- TEXTDATSET is one of the dataset keys defined in data_utils.py
- img_patch_size: Image patch size to embed each sequence token into
- vision_model is one of the following [vit_base_patch16_384 ,resnet18, resnet50

