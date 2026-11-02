# Research background

Mental rotation, introduced by Shepard and Metzler (1971), asks a person to decide whether two objects are the same shape under rotation. Accuracy and response time have been the usual outcomes. Strategy — holistic rotation versus piecemeal comparison — is harder to observe, because think-aloud and interviews depend on what a person can report.

Ashwin et al. (2025) collected a multimodal recording from 38 South Asian post-graduate students while they solved mental-rotation items under three conditions. The recording includes four-channel Muse EEG, exported through Mind Monitor, together with PsychoPy logs of difficulty, correctness and response time.

This repository uses that public dataset to ask a narrower question than "what was the person thinking". The question is whether features of the EEG, computed on the interval before the response, carry information about the accuracy and the duration of that response.

EEG spectral power has been related to strategy and workload in mental rotation (Gardony et al., 2017). Those findings motivate the features used here. They do not licence a claim that a classifier has recovered the strategy.

Source: Ashwin T. S. et al., Scientific Data, 2025. https://doi.org/10.1038/s41597-025-04865-5
