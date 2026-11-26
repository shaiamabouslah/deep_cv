# Surface index to official fill rate

Per-dam monotone calibration (isotonic regression anchored at empty = 0 and full = 100) fitted on the ONAGRI/DGRE daily bulletins (one per week, 2024 to October 2025) paired with the satellite measurement of the same week. Errors are in fill-rate points, measured on periods the fit did not see (5 contiguous time blocks, each dam predicted from its other blocks).

| Dam | Points | Raw surface error | Constant-guess error | Calibrated error | Skill vs constant | Offered |
|---|---|---|---|---|---|---|
| Bir M'cherga | 81 | 21.1 | 6.0 | 2.4 | 0.60 | yes |
| Lakhmes | 74 | 19.9 | 6.1 | 3.6 | 0.40 | yes |
| Mellègue | 61 | 9.6 | 15.8 | 3.4 | 0.79 | yes |
| Nebhana | 78 | 35.0 | 2.6 | 3.4 | -0.27 | no |
| Sidi El Barrak | 55 | 10.4 | 13.4 | 2.8 | 0.79 | yes |
| Sidi Salem | 61 | 34.1 | 8.4 | 1.6 | 0.82 | yes |

A dam is offered a calibrated rate only if it has at least 30 points and a skill of at least 0.3 over the constant guess (the constant is the average official rate of the training blocks).
