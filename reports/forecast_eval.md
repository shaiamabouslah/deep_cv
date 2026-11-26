# Deep-learning forecaster — evaluation

Task: from the last 12 weeks of a dam (water area, observation freshness, rainfall, evaporation, temperature,
season), predict its water area 1, 2 and 4 weeks ahead. Data: `data/forecast/weekly.csv`, 2,502 dam-weeks
(6 dams, Dec 2018 to Nov 2026), built from Sentinel-2 (cloud-aware, same measurement as the live app),
CHIRPS rainfall and ERA5-Land evaporation/temperature (`python -m src.build_forecast_dataset`).

Protocol: time-disjoint split. Train up to 2024-07, validation 2024-07 to 2025-07 (early stopping only),
test 2025-07 to 2026-11 (425 windows, 71 distinct weeks, one full wet season). Models are GRUs (~8k
parameters), 5 seeds each, predictions averaged (deep ensemble). All methods are scored on the same windows.
Reproduce: `python -m src.train_forecaster`, then `python -m src.forecast_significance`.

## Mean absolute error, percentage points of reservoir extent (lower is better)

| Method | +1 week | +2 weeks | +4 weeks |
|---|---|---|---|
| No change (persistence) | 2.12 | 2.76 | 3.97 |
| **Linear trend (what the app used before)** | 2.30 | 3.06 | 4.70 |
| Seasonal average change | 2.13 | 2.72 | 3.62 |
| Ridge regression | 2.70 | 3.14 | 4.04 |
| GRU, no weather | 2.03 | 2.61 | 3.54 |
| GRU, weather history | **2.00** | 2.55 | **3.44** |
| GRU, weather + next-2-weeks rain | 2.01 | **2.50** | 3.46 |

Paired difference to the no-change baseline, 95% bootstrap interval resampling whole calendar weeks (dams
move together, so windows of one week are not independent):

| | +1 week | +2 weeks | +4 weeks |
|---|---|---|---|
| GRU (weather history) minus persistence | -0.12 [-0.19, -0.05] | -0.21 [-0.33, -0.09] | -0.53 [-0.75, -0.32] |
| GRU minus linear trend | -0.30 [-0.42, -0.20] | -0.52 [-0.70, -0.34] | -1.26 [-1.59, -0.96] |
| Persistence minus linear trend | -0.18 [-0.30, -0.07] | -0.30 [-0.52, -0.10] | -0.73 [-1.13, -0.36] |

## What this shows, and what it does not

- **The GRU beats every baseline at every horizon, and the gain is statistically real** (intervals exclude
  zero). The gain grows with the horizon: about 5% better than "no change" at one week, 13% at four weeks.
- **The linear trend the app used before is worse than predicting no change** at every horizon (significant).
  Replacing it is the clearest practical win.
- **Weather history helps** (GRU without weather is worse at 2 and 4 weeks, significant). **The rain forecast
  for the next two weeks does not**: with or without it the results are indistinguishable, even though training
  used the real rain (an optimistic "perfect forecast"). The served model therefore does not use it, which also
  avoids a training/serving mismatch.
- **The absolute gains are small.** At one week the model is 0.1 point better than "no change". On the weeks
  where the level really moved by 5 points or more (39 to 83 windows) all methods are poor (MAE about 10 to 11
  points) and the GRU barely helps: it improves the ordinary weeks, it does not predict the big jumps.
- It is better than "no change" on 4 of 6 dams at +4 weeks, equal on Lakhmes and Nebhana (Nebhana barely moves).
- On a dam hidden from training (leave-one-dam-out) the GRU keeps most of its advantage (2.05 / 2.57 / 3.55 vs
  2.12 / 2.76 / 3.97), so it should transfer to newly registered sites.
- The target is the satellite area index, measured with noise of its own (a typical week-to-week change is about
  1 point). Part of the remaining error is measurement noise no model can remove.
