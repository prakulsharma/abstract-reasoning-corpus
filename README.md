# Abstraction and Reasoning Corpus

## Overview

In his paper "On the Measure of Intelligence," François Chollet critiques the prevailing task-specific methods used to assess AI intelligence as insufficient. He argues that true intelligence involves the ability to efficiently acquire skills across various domains, which these methods fail to measure. To address this, Chollet introduced the Abstraction and Reasoning Corpus (ARC), designed to test AI on tasks requiring broad, human-like intelligence. These tasks highlight the significance of inherent understanding and minimal data generalization.

Building on Chollet’s work, my project aims to tackle the ARC challenge through a unique method that integrates human insights with the capabilities of large language models (LLMs). The primary objective is to develop a collection of human-annotated explanations detailing the reasoning behind solving ARC problems. This collection will help fine-tune an open-source LLM, enhancing its problem-solving strategies to more closely resemble human thought processes.

## Project Steps

1. **Annotation Creation**: The project begins by crafting detailed, step-by-step explanations for both the training and evaluation sets of the ARC challenge. These annotations will elucidate the logical and abstract reasoning required to navigate ARC’s complex problem space.

2. **Model Fine-Tuning**: Utilizing the annotations, I will employ a two-stage fine-tuning process for the LLM:

    - Generate detailed guides on solving ARC tasks based on the annotations.
    - Use these guides to predict the outcomes of ARC tasks, employing a chain-of-thought approach.

3. **(Optional) Multimodal LLM Fine-Tuning**: Considering the integration of visual data in human problem-solving, I am contemplating fine-tuning a multimodal LLM, though recent studies suggest these may perform less effectively in reasoning tasks compared to standard LLMs.

4. **Dashboard and Live Demo**: The project will culminate in the creation of a dashboard and a live demonstration where I will compare the human-generated annotations against those produced by the fine-tuned LLM.

## Expected Outcomes and Impact

This project explores the potential to enhance LLMs' problem-solving capabilities by guiding them with human-annotated explanations. The goal is to determine if an LLM can be trained to understand and emulate human-like reasoning paths for complex problem solving and to accurately apply these methods to achieve correct solutions. This serves as a preliminary step towards more sophisticated systems capable of human-like reasoning and problem-solving, setting the stage for further research into overcoming the limitations and challenges faced by current AI models.

## Technical Details

### Abstraction and Reasoning Corpus (ARC)

- Hosted on Kaggle.
- Contains 400 training examples, 400 evaluation examples, and 100 testing examples.
- Each example is stored as a JSON file with sections for training and testing reasoning tasks.
- An interactive app is available for visualizing and interacting with the reasoning tasks.

### Annotation Details

- Annotations will be stored in a CSV file with two columns: the filename and the corresponding human-annotated explanations.
- A custom dashboard will be developed, modifying the existing interactive app to allow for direct annotation creation within the app interface, enhancing the efficiency and intuitiveness of the annotation process.

<!-- https://drive.google.com/file/d/1ZivhhXuwiot2SnvLAL17D5z5SwnLbQyU/view?usp=share_link -->

## Data

The `data` directory contains the following subdirectories:

1. `references`: Partial dataset comprising of gpt4 and human annotations that is used as a basis to form the corpus.
2. `original`: Original Kaggle dataset comprising of images in the form of .json files, that we use to form our main corpus containing images and the annotations.
3. `images`: Images regenerated from the .json files in the `original` folder, that we use for writing the human annotations.
4. `annotations`: Annotations for the images in the form of .tsv files.

## Source

The `src` directory contains the following files:

1. `utils.py`: Python script that converts .json files of images into .png files for visual aid while writing the human annotations.

## References

1. Chollet, François. "On the measure of intelligence." arXiv preprint arXiv:1911.01547 (2019).
2. Wei, Jason, et al. "Chain-of-thought prompting elicits reasoning in large language models." Advances in Neural Information Processing Systems 35 (2022): 24824-24837
3. Mitchell, Melanie, Alessandro B. Palmarini, and Arseny Moskvichev. "Comparing Humans, GPT-4, and GPT-4V on abstraction and reasoning tasks." arXiv preprint arXiv:2311.09247 (2023).
