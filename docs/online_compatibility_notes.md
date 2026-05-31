# Online-compatibility implementation notes

The final experimental pipeline must respect the information set available at prediction day \(d_t\).

## Implemented safeguards in `src/`

1. `Avg_Remaining_Time` is a target only and is not used as an input feature.
2. Elapsed-time forecasts may be trained on a change target because current elapsed time is observable.
3. Remaining-time forecasts use a future-level target because current realized remaining time is not observable at prediction time.
4. The modular feature builder does not use non-causal peak detection (`scipy.signal.find_peaks` on the full series).

## Archived notebooks

The output-stripped notebooks are retained as development records. Earlier experimental cells should not be interpreted as the final audited implementation unless they are synchronized with the modular code.
