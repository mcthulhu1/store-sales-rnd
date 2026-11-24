Среднее по 3 seed; delta = RMSLE(baseline) - RMSLE(эксп.) на том же seed

| experiment | extra_features | rmsle_mean | rmsle_seed_range | delta_mean | delta_min | delta_max | seeds_better | passes_criterion | mae_mean | school_rmsle_f3 | bias_12_13_aug_pct | bias_31jul_1aug_pct |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E0_baseline | - | 0.3989 | 0.0007 | 0.0 | 0.0 | 0.0 | 0 | False | 63.0626 | 0.797 | 19.9716 | -12.3356 |
| E1_yoy | yoy | 0.3991 | 0.0006 | -0.0002 | -0.0008 | 0.0005 | 1 | False | 58.0525 | 0.8269 | 11.0461 | -12.2654 |
| E2_calendar | long_weekend, local_holiday, payday | 0.3981 | 0.0006 | 0.0008 | 0.0004 | 0.0011 | 3 | True | 62.1561 | 0.7901 | 11.8416 | -10.3828 |
| E3_full | yoy, long_weekend, local_holiday, payday | 0.3986 | 0.0012 | 0.0003 | -0.0002 | 0.0008 | 2 | False | 57.9821 | 0.8339 | 5.1444 | -10.5134 |
| A1_full_minus_long_weekend | yoy, local_holiday, payday | 0.3994 | 0.001 | -0.0005 | -0.0013 | 0.0004 | 1 | False | 58.2172 | 0.8423 | 9.9446 | -10.4008 |
| A2_full_minus_local_holiday | yoy, long_weekend, payday | 0.3984 | 0.0013 | 0.0005 | 0.0 | 0.0009 | 3 | False | 57.854 | 0.8396 | 5.1929 | -10.0112 |
| A3_full_minus_payday | yoy, long_weekend, local_holiday | 0.3986 | 0.0007 | 0.0003 | 0.0 | 0.0004 | 3 | False | 57.6126 | 0.8509 | 5.0874 | -11.5906 |
