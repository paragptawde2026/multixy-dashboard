/**
 * api.js
 * ------
 * All HTTP calls to the FastAPI backend live here.
 * Uses /api prefix which Vite proxies to http://localhost:8000
 */

import axios from 'axios'

// In development: Vite proxies /api → http://localhost:8000 (see vite.config.js).
// In production:  VITE_API_URL is set to the Render backend URL, e.g.
//                 https://multixy-backend.onrender.com
const BASE = (import.meta.env.VITE_API_URL ?? '') + '/api'

const api = axios.create({ baseURL: BASE })

// ── Data endpoints ─────────────────────────────────────────────
export const loadProjectExcel = () => api.post('/data/load-excel')

export const uploadDataset = (formData) =>
  api.post('/data/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })

export const listDatasets = () => api.get('/data/')

export const getDataset = (id) => api.get(`/data/${id}`)

export const previewDataset = (id, rows = 10) =>
  api.get(`/data/${id}/preview?rows=${rows}`)

// ── Model endpoints ────────────────────────────────────────────
export const trainModel      = (payload) => api.post('/model/train', payload)
export const stopTraining    = (runId)   => api.post(`/model/runs/${runId}/stop`)
export const killAllTraining = ()        => api.post('/model/kill-all')
export const getActiveRun    = ()        => api.get('/model/active')
export const listRuns           = ()        => api.get('/model/runs')
export const getModelComparison = ()        => api.get('/model/comparison')
export const getRun          = (id)      => api.get(`/model/runs/${id}`)
export const predictTestData = (runId)   => api.get(`/model/runs/${runId}/predict-test`)
export const predictModel    = (payload) => api.post('/model/predict', payload)
export const whatIfPredict      = (payload) => api.post('/model/whatif',  payload)
export const getFeatureWeights  = (runId)   => api.get(`/model/runs/${runId}/feature_weights`)
export const listPredictions = ()        => api.get('/model/predictions')

// ── Hyperparameter tuning endpoints ───────────────────────────────
export const startTuning      = (payload)   => api.post('/model/tune', payload)
export const getActiveTuning  = ()          => api.get('/model/tuning/active')
export const getTuningRun     = (id)        => api.get(`/model/tuning/${id}`)
export const listTuningRuns   = ()          => api.get('/model/tuning')

// ── Preprocessing / dataset endpoints ─────────────────────────────
export const deleteDataset      = (id, force = false) => api.delete(`/data/${id}?force=${force}`)
export const getDatasetStats    = (id)                => api.get(`/data/${id}/stats`)
export const sampleDatasetRows  = (id, xCols, n = 150) =>
  api.get(`/data/${id}/sample_rows?x_cols=${xCols.join(',')}&n=${n}`)
export const getXCoupling       = (id, varyX, xCols) =>
  api.get(`/data/${id}/x_coupling?vary_x=${encodeURIComponent(varyX)}&x_cols=${xCols.join(',')}`)
export const preprocessDataset  = (id, payload)       => api.post(`/data/${id}/preprocess`, payload)
export const downloadDataset    = (id)                => `${BASE}/data/${id}/download`

// ── Run management endpoints ───────────────────────────────────────
export const deleteRun      = (id)                    => api.delete(`/model/runs/${id}`)
export const bulkDeleteRuns = (keepLast = 0, status = null) =>
  api.delete(`/model/runs?keep_last=${keepLast}${status ? `&status=${status}` : ''}`)
