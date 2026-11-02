# Published experimental protocol

This document restates the protocol in Ashwin et al. (2025), Scientific Data (https://doi.org/10.1038/s41597-025-04865-5). It is not a protocol that this repository carried out.

## Objective

Collect multimodal recordings while post-graduate students solved Shepard–Metzler mental-rotation items, so that later work can study strategy, affect and performance. The authors did not publish a trained EEG model.

## Participants

38 participants: 22 men and 16 women, described as South Asian post-graduate students. Consent and IRB approval are reported (IIT Bombay IRB IITB-IRB/2022/006). The public release is described as anonymised and is shared under CC0.

Before the task, participants completed a presurvey (about 25 minutes) and two working-memory tests on PsyToolkit: backward Corsi and a 2-back (about 10 minutes). The three task categories took about 25 minutes.

## Setup

Participants sat about 60 cm from a 1920×1080 laptop. A Tobii eye tracker, a Shimmer GSR sensor on the non-dominant hand, and a Muse EEG headband were calibrated before the task. iMotions collected eye tracking, GSR and facial affect. Mind Monitor collected the Muse stream. An observer behind a one-way mirror logged learning-centred emotion with DLOT every 10 seconds. Participants were told about the observer.

## EEG

Muse headband, four dry electrodes at AF7, AF8, TP9 and TP10 in the 10–20 system. AF7 and AF8 are frontal; TP9 and TP10 sit at the ears. The paper lists the CSV attributes (timestamp, five band powers per channel, raw EEG per channel, auxiliary, accelerometer, gyroscope, headband-on, HSI, battery) and says the file has 40 attributes. It does not state the sampling rate. Alignment with the other streams used Unix time. The authors warn that the alignment may contain errors and ask users to check it.

## Task

PsychoPy presented pairs of images drawn from a subset of the Peters and Battista library. The participant judged match or mismatch. Difficulty was constructed from angular disparity (0 to 180 degrees) and from configuration, arm position, occlusion and axis of rotation. The paper labels items Easy, Medium or Difficult.

Two practice items preceded each category. Categories 1 and 2 allowed 60 seconds for those two items. Category 3 allowed 12 seconds.

| Category | Time restriction | Feedback | Scored items | Category duration |
|---|---|---|---|---|
| 1 | no | no | 10 | 300 s |
| 2 | no | yes, shown after the choice | 10 | 300 s |
| 3 | yes | no | 20 available; nobody finished all 20 | 120 s |

There was no break between categories. NASA-TLX (without the physical-demand item) was completed after each category.

A pilot of six people found response times of about 2.4–3.2 seconds across angular disparities, and a longest single item of 20 seconds. The authors treat the 300-second caps in categories 1 and 2 as "no time restriction" because the pilot participants reported no time pressure.

## Behavioural measures

The PsychoPy file is described as containing: Key, Category, QuestionNumber, match Ornomatch, Difficulty, Verdict, ResponseTime, routineStart, routineStamp, routineEnd, and Cat2FeedbackTime.

## What this repository does not inherit

The paper's bar plots of band power by category are descriptive results of the authors. They are not copied into this analysis. This repository recomputes its own summaries from the files.
