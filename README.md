<p align="center">
    <h1 align="center"><a href="https://arxiv.org/pdf/2409.14874" target="_blank">Segment, Interact, and Fuse: A Multi-Stage Framework for Multi-Phase CT Structured Report Generation</a></h1>
</p>



<h4 align="center">
    <p>
        <a href="https://github.com/ahjolsenbics/EvanySeg/blob/main/README.md#Framework">Introduction</a> |
        <a href="#-Getting Started">Highlights</a> |
        <a href="#-Getting Started">Getting Started</a> |
    <p>
</h4>



<p align="center">
    <a href="https://www.python.org/">
        <img alt="Build" src="https://img.shields.io/badge/Made%20with-Python-1f425f.svg?color=purple">
    </a>
    <a href="https://github.com/facebookresearch/segment-anything/blob/main/LICENSE">
        <img alt="License" src="https://img.shields.io/github/license/confident-ai/deepeval.svg?color=turquoise">
    </a>
</p>

## Introduction

MCT2SMR is a multi-stage framework for generating structured medical reports from multi-phase 3D CT scans. Unlike conventional report generation methods that mainly focus on 2D images or free-text reports, MCT2SMR targets the more clinically practical setting of structured reporting, where findings must be accurate within each section and semantically consistent across the entire report. The framework integrates three key components: a multi-phase segmentation-guided module for pixel-level grounding and quantitative lesion description, a dynamic multi-phase perception interaction module for modeling cross-phase enhancement patterns, and a multi-stage meta-token decoder for producing coherent structured reports. Together, these modules enable the model to better capture lesion morphology, temporal enhancement differences, and logical relationships among report sections. MCT2SMR is evaluated on a multi-phase CT structured report generation task and demonstrates strong performance across both standard natural language generation metrics and LLM-based clinical evaluation, showing its potential for reliable and clinically relevant automated radiology reporting.

<img src="./assets/workflow.png">



## Highlights

MCT2SMR advances structured medical report generation from multi-phase CT scans by jointly modeling pixel-level lesion evidence, cross-phase dynamic changes, and inter-section semantic consistency:

1. End-to-end structured report generation from multi-phase 3D CT scans;
2. Segmentation-guided framework for factual and quantitative findings;
3. Dynamic multi-phase interaction for cross-phase feature perception;
4. Multi-stage meta-token decoding for coherent structured reports.



## Getting Started

#### Requirment

Recommendation: Python version around 3.9, please do not use a version that is too high.

```
pip install -r requirements.txt
```

- torch==2.0.1
- torchvision==0.15.2
- spacy==3.5.0
- transformers==4.30.1
- einops==0.8.0
- ...

#### Test

```python
python test.py
```

#### Train

```python
python main.py
```
