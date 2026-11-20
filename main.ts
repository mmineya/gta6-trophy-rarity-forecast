/**
 * GTA VI trophy-rate forecast: weighted log-time trends with damped drift.
 * Node 22, standard library only. No network, dependencies, or random state.
 *
 * Run with MAKEFUN_INPUT_DIR and MAKEFUN_OUTPUT_DIR, or /input and /output.
 */
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

interface Trophy {
  id: number;
}
interface HistoryRead {
  at: string;
  slot?: string;
  rates: Array<number | null>;
}
interface ForecastCase {
  at: string;
  target: string;
  horizon_minutes: number;
  levels: number[];
  trophies: Trophy[];
  history: HistoryRead[];
}
interface Observation {
  x: number;
  y: number;
  rate: number;
  time: number;
}
interface Fit {
  a: number;
  b: number;
  residualStd: number;
}

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';
const RELEASE_MS = Date.parse('2026-11-19T00:00:00Z');
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
const Z = [-1.645, -0.674, 0, 0.674, 1.645];
const LAST_RATE_WEIGHT = 0.1;
const LOGIT_LIMIT = Math.log(99.99 / 0.01);

function clamp(value: number, low: number, high: number): number {
  return Math.min(high, Math.max(low, value));
}

function logit(rate: number): number {
  const p = clamp(rate, 0.01, 99.99);
  return Math.log(p / (100 - p));
}

function timeCoordinate(time: number): number {
  // The earliest launch-night read precedes midnight by ten minutes.
  // Keep that negative hour; protect log1p if an older read is supplied.
  const hours = (time - RELEASE_MS) / 3_600_000;
  return Math.log1p(Math.max(-0.999999, hours));
}

function sigmoidPercent(value: number): number {
  // This form remains finite even for a very large-magnitude logit.
  if (value >= 0) return 100 / (1 + Math.exp(-value));
  const exponential = Math.exp(value);
  return 100 * exponential / (1 + exponential);
}

function round4(value: number): number {
  return Math.round(value * 10_000) / 10_000;
}

/**
 * Fit recency-weighted log-time trends; forecasting damps positive drift by
 * the recent/long-window slope ratio (0.35–1.15) and suppresses negative drift.
 */
function weightedFit(points: Observation[], checkpointX: number): Fit {
  // Center x at the checkpoint. The intercept a is then the fitted logit
  // AT the checkpoint, rather than the raw intercept at launch midnight.
  // Linear recency weights are 1, 2, ..., K over valid observations only.
  let weight = 0;
  let squaredWeight = 0;
  let sumX = 0;
  let sumY = 0;
  for (let i = 0; i < points.length; i++) {
    const w = i + 1;
    weight += w;
    squaredWeight += w * w;
    sumX += w * (points[i].x - checkpointX);
    sumY += w * points[i].y;
  }
  const meanX = sumX / weight;
  const meanY = sumY / weight;
  let xx = 0;
  let xy = 0;
  for (let i = 0; i < points.length; i++) {
    const dx = points[i].x - checkpointX - meanX;
    xx += (i + 1) * dx * dx;
    xy += (i + 1) * dx * (points[i].y - meanY);
  }
  const identifiable = points.length >= 2 && xx > 1e-12;
  const b = identifiable ? xy / xx : 0;
  const a = meanY - b * meanX;
  let squaredError = 0;
  for (let i = 0; i < points.length; i++) {
    const error = points[i].y - (a + b * (points[i].x - checkpointX));
    squaredError += (i + 1) * error * error;
  }
  // Correct weighted residual variance using the effective sample size.
  const effectiveN = weight * weight / squaredWeight;
  const parameters = identifiable ? 2 : 1;
  const correction = effectiveN > parameters ? effectiveN / (effectiveN - parameters) : 1;
  const residualStd = Math.sqrt(Math.max(0, squaredError / weight * correction));
  return { a, b, residualStd };
}

function normalizeQuantiles(values: number[]): number[] {
  let previous = 0.01;
  return values.map((value) => {
    const finiteValue = Number.isFinite(value) ? value : previous;
    previous = Math.max(previous, round4(clamp(finiteValue, 0.01, 100)));
    return previous;
  });
}

function forecastTrophy(c: ForecastCase, index: number, checkpointMs: number, targetMs: number): number[] {
  const observations: Observation[] = [];
  for (const read of c.history) {
    const rate = read.rates[index];
    const time = Date.parse(read.at ?? read.slot ?? '');
    // A missing rate is not zero. Never admit reads after the checkpoint.
    if (typeof rate !== 'number' || !Number.isFinite(rate) || !Number.isFinite(time) || time > checkpointMs) continue;
    observations.push({ x: timeCoordinate(time), y: logit(rate), rate: clamp(rate, 0, 100), time });
  }
  observations.sort((left, right) => left.time - right.time);
  const points = observations.slice(-Math.min(24, observations.length));
  const last = points[points.length - 1];
  if (points.length < 2) {
    // Too few valid reads to fit a trend: use the house persistence baseline.
    // A wholly missing rate uses zero before the baseline's logit clamp.
    const center = logit(last?.rate ?? 0);
    const spread = 0.5 * Math.sqrt(Math.max(0, c.horizon_minutes / 1440));
    return Z.map((z) => round4(sigmoidPercent(center + z * spread)));
  }
  if (last.rate >= 99.5) return [99.9, 99.95, 99.99, 100, 100];

  const days = Math.max(0, c.horizon_minutes / 1440);
  const checkpointX = timeCoordinate(checkpointMs);
  const deltaX = Math.max(0, timeCoordinate(targetMs) - checkpointX);
  const longFit = weightedFit(points, checkpointX);
  const recentLength = Math.min(points.length, Math.max(3, Math.ceil(points.length / 2)));
  const recentFit = weightedFit(points.slice(-recentLength), checkpointX);

  // A falling rate can reflect an expanding denominator, not lost trophies.
  // Do not extrapolate negative drift. Positive drift is damped by the ratio
  // of recent to longer-window slopes; a recent reversal gets minimum drift.
  let effectiveSlope = 0;
  if (longFit.b > 1e-10) {
    const ratio = recentFit.b / longFit.b;
    effectiveSlope = longFit.b * clamp(ratio, 0.35, 1.15);
  }
  const anchor = longFit.b > 1e-10
    ? (1 - LAST_RATE_WEIGHT) * longFit.a + LAST_RATE_WEIGHT * last.y
    : last.y;
  const drift = effectiveSlope * deltaX;
  const medianLogit = clamp(anchor + drift, -LOGIT_LIMIT, LOGIT_LIMIT);
  const spread = Math.max(
    0.75,
    1.35 * Math.sqrt(days),
    4.2 * longFit.residualStd * Math.sqrt(days / 2),
    Math.abs(drift) * 0.9,
  );

  if (last.rate <= 0.02) {
    // Sparse, rounded-to-zero rates need a strictly positive rare-event band.
    // Keep the requested proportional shape while allowing horizon/trend growth.
    const base = Math.max(0.02, sigmoidPercent(medianLogit) / 2);
    const scale = Math.max(1, Math.sqrt(days));
    return normalizeQuantiles([0.5, 1, 2, 4, 8].map((factor) => base * factor * scale));
  }
  return normalizeQuantiles(Z.map((z) => sigmoidPercent(medianLogit + z * spread)));
}

export function forecastCase(c: ForecastCase): { trophies: Array<{ id: number; q: number[] }> } {
  if (!Array.isArray(c.levels) || c.levels.length !== LEVELS.length || c.levels.some((level, i) => level !== LEVELS[i])) {
    throw new Error('Expected quantile levels [0.05,0.25,0.5,0.75,0.95]');
  }
  if (!Array.isArray(c.history) || c.history.length === 0) throw new Error('Case has no history');
  if (!Number.isFinite(c.horizon_minutes) || c.horizon_minutes < 0) throw new Error('Invalid forecast horizon');
  const checkpointMs = Date.parse(c.at);
  const targetMs = Date.parse(c.target);
  if (!Number.isFinite(checkpointMs) || !Number.isFinite(targetMs) || targetMs < checkpointMs) {
    throw new Error('Invalid checkpoint or target timestamp');
  }
  return { trophies: c.trophies.map((trophy, index) => ({ id: trophy.id, q: forecastTrophy(c, index, checkpointMs, targetMs) })) };
}

mkdirSync(OUTPUT_DIR, { recursive: true });
for (const file of readdirSync(INPUT_DIR).filter((name) => name.endsWith('.in')).sort()) {
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as ForecastCase;
  const output = JSON.stringify(forecastCase(c)) + '\n';
  if (Buffer.byteLength(output, 'utf8') > 65_536) throw new Error(`${file}: output exceeds 65,536 bytes`);
  writeFileSync(join(OUTPUT_DIR, `${file.slice(0, -3)}.out`), output, 'utf8');
}
