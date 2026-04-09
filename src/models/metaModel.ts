// NCAAM Oracle v4.1 — ML Meta-Model (Logistic Regression + Isotonic Calibration)
// Loads trained coefficients from model/model_weights.json
// Falls back to Monte Carlo win probability if model files absent.
// Train with: python python/train_model.py → exports to model/

import { existsSync, readFileSync } from 'fs';
import { resolve } from 'path';
import { logger } from '../logger.js';
import type { FeatureVector } from '../types.js';

const MODEL_DIR = resolve('model');

// ─── JSON artifact shapes ─────────────────────────────────────────────────────

interface CoefficientsJson {
  _intercept: number;
  [featureName: string]: number;
}

interface ScalerJson {
  feature_names: string[];
  mean: number[];
  scale: number[];
}

interface CalibrationJson {
  method: 'isotonic';
  x_thresholds: number[];
  y_thresholds: number[];
  n_thresholds: number;
}

interface ModelMetadataJson {
  version: string;
  model_type: string;
  feature_names: string[];
  train_seasons: string;
  avg_brier: number;
  avg_accuracy: number;
  trained_at: string;
}

interface LoadedModel {
  featureNames: string[];
  coefficients: Float64Array;
  intercept: number;
  scalerMean: Float64Array;
  scalerScale: Float64Array;
  calibX: Float64Array;
  calibY: Float64Array;
  metadata: ModelMetadataJson;
}

let _model: LoadedModel | null = null;

export function isModelLoaded(): boolean {
  return _model !== null;
}

export function getModelInfo(): { version: string; avgBrier: number; trainSeasons: string } | null {
  if (!_model) return null;
  return {
    version: _model.metadata.version,
    avgBrier: _model.metadata.avg_brier,
    trainSeasons: _model.metadata.train_seasons,
  };
}

// ─── Load model from disk ─────────────────────────────────────────────────────

export function loadModel(): boolean {
  const coeffPath = resolve(MODEL_DIR, 'coefficients.json');
  const scalerPath = resolve(MODEL_DIR, 'scaler.json');
  const calibPath  = resolve(MODEL_DIR, 'calibration.json');
  const metaPath   = resolve(MODEL_DIR, 'metadata.json');

  if (!existsSync(coeffPath) || !existsSync(scalerPath) || !existsSync(calibPath) || !existsSync(metaPath)) {
    logger.info('ML model files not found — using Monte Carlo fallback');
    logger.info(`Train with: python python/train_model.py`);
    return false;
  }

  try {
    const coeffs = JSON.parse(readFileSync(coeffPath, 'utf-8')) as CoefficientsJson;
    const scaler  = JSON.parse(readFileSync(scalerPath, 'utf-8')) as ScalerJson;
    const calib   = JSON.parse(readFileSync(calibPath, 'utf-8')) as CalibrationJson;
    const meta    = JSON.parse(readFileSync(metaPath, 'utf-8')) as ModelMetadataJson;

    const featureNames = scaler.feature_names;
    const n = featureNames.length;

    const coeffArr = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      coeffArr[i] = coeffs[featureNames[i]] ?? 0;
    }

    _model = {
      featureNames,
      coefficients: coeffArr,
      intercept: coeffs['_intercept'] ?? 0,
      scalerMean: new Float64Array(scaler.mean),
      scalerScale: new Float64Array(scaler.scale),
      calibX: new Float64Array(calib.x_thresholds),
      calibY: new Float64Array(calib.y_thresholds),
      metadata: meta,
    };

    logger.info(
      { version: meta.version, features: n, avgBrier: meta.avg_brier, seasons: meta.train_seasons },
      'NCAAM ML meta-model loaded'
    );
    return true;
  } catch (err) {
    logger.error({ err }, 'Failed to load ML model — falling back to Monte Carlo');
    _model = null;
    return false;
  }
}

// ─── Sigmoid ──────────────────────────────────────────────────────────────────

function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

// ─── Isotonic calibration ─────────────────────────────────────────────────────

function isotonicCalibrate(rawProb: number, calibX: Float64Array, calibY: Float64Array): number {
  const n = calibX.length;
  if (n === 0) return rawProb;
  if (rawProb <= calibX[0]) return calibY[0];
  if (rawProb >= calibX[n - 1]) return calibY[n - 1];

  let lo = 0;
  let hi = n - 1;
  while (lo < hi - 1) {
    const mid = (lo + hi) >> 1;
    if (calibX[mid] <= rawProb) lo = mid;
    else hi = mid;
  }

  const t = (rawProb - calibX[lo]) / (calibX[hi] - calibX[lo]);
  return calibY[lo] + t * (calibY[hi] - calibY[lo]);
}

// ─── Build feature array ──────────────────────────────────────────────────────

function buildFeatureArray(features: FeatureVector, featureNames: string[]): Float64Array {
  const arr = new Float64Array(featureNames.length);
  const fv = features as unknown as Record<string, number>;
  for (let i = 0; i < featureNames.length; i++) {
    arr[i] = fv[featureNames[i]] ?? 0;
  }
  return arr;
}

// ─── Predict ──────────────────────────────────────────────────────────────────

export function predict(features: FeatureVector, mcWinProb: number): number {
  if (!_model) return mcWinProb;

  const { featureNames, coefficients, intercept, scalerMean, scalerScale, calibX, calibY } = _model;

  const raw = buildFeatureArray(features, featureNames);

  const mcIdx = featureNames.indexOf('mc_win_pct');
  if (mcIdx >= 0) raw[mcIdx] = mcWinProb;

  const n = featureNames.length;
  let logit = intercept;

  for (let i = 0; i < n; i++) {
    if (scalerScale[i] <= 0) continue;
    const z = (raw[i] - scalerMean[i]) / scalerScale[i];
    const scaled = Math.max(-3, Math.min(3, z));
    logit += coefficients[i] * scaled;
  }

  const rawProb   = sigmoid(logit);
  const calibrated = isotonicCalibrate(rawProb, calibX, calibY);
  return Math.max(0.01, Math.min(0.99, calibrated));
}
