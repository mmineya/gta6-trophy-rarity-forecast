[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/mmineya/gta6-trophy-rarity-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/mmineya/gta6-trophy-rarity-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

[View the bounty on make.fun](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)

# GTA VI Trophy Rarity Forecast

A deterministic Node.js 22 program that forecasts five quantiles (5%, 25%, 50%, 75%, and 95%) of GTA VI trophy earn rates on Exophase. It uses only the standard library and runs offline.

## Run

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

The program reads each JSON `*.in` case in the input directory and writes its corresponding `*.out` forecast. The default directories are `/input` and `/output`.

## Model

A damped logit growth curve fits each trophy's latest valid reads against log-time, with larger weights for newer observations. Recent and longer-window slopes control damping; negative drift is suppressed. Quantile spread combines forecast horizon, residual variation, and projected growth. With fewer than two valid reads, the model falls back to the house persistence baseline.
