# Limitations

- The published sample is 38 participants from one institution. A model that works here is not a model of mental rotation in general.
- Four dry electrodes cannot localise a source and cannot support a claim about parietal cortex, even though mental-rotation EEG is often discussed in those terms.
- The paper does not state the EEG sampling rate. The code measures it. A median interval near 1 ms is not treated as a 1000 Hz recording when the 95th percentile interval is several times longer: that pattern is an irregular merged export, and raw filtering or Welch spectra are not run. Vendor band-power columns are used instead, and they are not filtered again.
- The authors state that Unix-time alignment may contain errors. This code rejects trial-level models when coverage is below 80%. That threshold is a decision of this repository, not a property of the data.
- Response-time units are not stated. Trials whose unit cannot be checked against the routine duration are excluded from models rather than converted by assumption.
- HSI encoding is not stated, so signal quality is reported as a distribution rather than a pass/fail.
- Trials from the same person are dependent. Tests and models respect that. A high trial count is not a high sample size.
- Class balance is unknown until the files are read. Accuracy without balanced accuracy would be misleading if most responses are correct.
- Predicting correctness is not decoding the rotation. A feature can predict an outcome because of movement, arousal, or time on task.
- Sex differences are discussed in the mental-rotation literature and in the source paper's descriptive plots. This repository does not test sex. The presurvey is a separate file, and a sex effect is not required to answer the questions posed here.
- Stimulus images are not in the public archive. They require permission from the stimulus-library author. Difficulty labels in the log are used as published.
