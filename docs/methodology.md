# Methodology

Decisions below belong to this repository. Where the published paper is silent, the code measures the quantity and records it. It does not fill the silence with a textbook default.

## Ingestion

`data_loader` finds `STData.zip` or an extracted participant tree. It reads the EEG and PsychoPy tables and renames headers that match the attributes named in the paper. Unrecognised headers are listed in the inventory. They are not dropped from the raw file and they are not given a meaning.

## Quality

Timestamps are parsed as Unix seconds, milliseconds, microseconds, or datetime strings, according to their magnitude. The sampling rate is the reciprocal of the median positive interval. Duplicate timestamps and gaps longer than five median intervals are counted. HSI values are tabulated. The paper does not define the HSI encoding, so a value is not labelled good or bad.

## Preprocessing

If the median rate is below 64 Hz, raw spectral analysis is skipped. Mind Monitor's default CSV interval is 1 Hz, and a 1 Hz series cannot represent delta, let alone gamma. Vendor band-power columns are then the only spectral features, and they are not filtered a second time.

A high median rate is not enough. Welch's method and a digital filter assume a uniform series. If the 95th percentile interval is more than five times the median interval, the rows are an irregular export: short steps mixed with gaps, and values that stay constant while another column updates. Filtering or a spectrum on that grid would invent samples. The pipeline records the intervals and uses the vendor band-power columns, again without filtering them.

When the series is uniform and at least 64 Hz, and the 95th percentile interval is no more than twice the median, raw channels are band-pass filtered from 0.5 to 45 Hz and notched at 50 Hz, the mains frequency in Mumbai. A ratio above 2 means the grid still contains gaps, so those files stay on the vendor band-power columns. A band is estimated only when its upper edge is below the Nyquist frequency, so gamma (30–45 Hz) requires a rate above 90 Hz. Artefacts are samples whose robust z-score exceeds 6. They are excluded from feature extraction, not interpolated into the result.

## Windows

The EEG window starts at `routineStart`. It ends at the response when response time can be expressed in the same unit as the routine duration. Category 2 feedback therefore stays outside the window. Trials where that conversion fails are marked and are excluded from models, because the routine end may include the feedback screen.

Trial-level analysis is accepted per participant when at least 80% of that person's scored trials contain at least three EEG samples inside the response window. Participants who do not meet that rule are excluded from models and listed in the synchronization report. A recording that does not cover the task is not used as evidence that another recording is misaligned. Models are fit only when at least two participants pass. Otherwise the pipeline stops at a participant-by-condition behavioural summary.

## Features

Per channel, when raw analysis is allowed: mean, standard deviation, variance, root mean square, range, skewness, kurtosis, Hjorth activity, mobility and complexity, Welch absolute and relative band power, spectral entropy, and the theta/alpha and beta/alpha ratios. Vendor band means are always computed when the columns exist, under a `vendor_` prefix, so they cannot be confused with recomputed power.

## Statistics

Repeated trials are collapsed to one mean per participant (and per condition, for the condition test) before a test is run. Condition differences use the Friedman test on complete participants. Associations use Spearman correlation across participants. P-values are adjusted with Benjamini–Hochberg within the family of tests that were run.

## Models

Classification target: correct versus incorrect. Regression target: response time in seconds. Three feature sets: EEG only, item metadata only (condition and difficulty), and both. Estimators: most-frequent or median dummy, logistic or linear, RBF SVM or SVR, and a random forest of 300 trees. Hyperparameters are fixed in `configs/default.yaml`. With 38 participants, a nested search would consume the sample. Validation is `GroupKFold` on participant id. Scaling and imputation are fit on the training groups only. Metrics are reported as the mean and standard deviation across folds.

## Importance

Permutation importance is the change in balanced accuracy, or in MAE, when one column is shuffled on a held-out group. Channel and band totals are sums of those values. They describe the model.
