# Proposed follow-up protocol

This experiment has not been run. It is a design that would make the scientific question in this repository answerable with fewer ambiguities than the public Muse recordings allow.

## Hypothesis

Frontal and parietal theta and alpha power, measured in the two seconds before the response, differ between correct and incorrect mental-rotation trials after item difficulty and participant are accounted for. A model using those features predicts held-out participants' accuracy above a difficulty-only baseline.

## Participants

Recruit at least 60 adults, with a preregistered stopping rule. Balance sex if a sex comparison is planned; otherwise treat sex as a covariate and do not test it. Exclude nobody for poor performance after enrolment; report the exclusion rule for unusable EEG before unblinding the outcomes. Obtain ethics approval before recruitment. The IIT Bombay study cannot be cited as approval for a new collection.

## EEG

Use a research-grade amplifier with at least 32 scalp electrodes, a common reference that can be re-referenced offline, and a sampling rate of 500 Hz or higher. Record stimulus onset, response, and feedback as hardware triggers on the same clock as the EEG. Keep the Muse only as a parallel recording if the aim is to test whether a consumer headset is sufficient. Do not use the Muse as the sole system if the claim is about cortical oscillations above 30 Hz.

## Task

Keep the three conditions, because they are the published manipulation. Add a jittered inter-trial interval of 1.5–2.5 seconds so that a pre-stimulus baseline exists. Show feedback only in condition 2, and mark its onset. Use the same image library, with permission, and preregister the difficulty coding. Fix the response window per item in the timed condition rather than a shared pool of 120 seconds, so that unfinished items are missing by design and not by an unknown stopping time.

## Preprocessing

Band-pass 0.5–45 Hz, notch at the local mains frequency, and identify ocular artefacts from frontal channels or from an explicit eye electrode. Segment from stimulus onset to response. Reject segments with amplifier saturation. Do not filter vendor band-power streams; do not compute them if raw data exist.

## Outcomes and analysis

Primary outcome: correct versus incorrect, modelled with a logistic mixed model (participant as a random intercept) and, separately, with grouped cross-validation. Secondary outcome: response time. The difficulty-only model is the baseline that an EEG model must beat. Permutation importance is computed on held-out participants. No claim about strategy is made unless an independent strategy measure, such as a coded gaze pattern, is collected.

## Confounds

Head motion, headset contact, time-on-task, and the feedback screen in condition 2 can all mimic a cognitive effect. The design above separates them by triggers, a baseline interval, and a metadata-only model.
