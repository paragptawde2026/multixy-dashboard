/**
 * WhatIf.jsx
 * ----------
 * What-If Simulator + Feature Analysis
 *
 * Left panel has two tabs:
 *   Controls       — Vary X, Observe Y, Baseline, action buttons
 *   Feature Analysis — Jacobian-based X→Y weight heatmap
 */

import { useState, useEffect } from 'react'
import {
  listRuns, listDatasets, getDatasetStats, whatIfPredict, getFeatureWeights, sampleDatasetRows,
  getXCoupling,
} from '../services/api'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  Legend, ResponsiveContainer, ReferenceLine,
} from 'recharts'

const COLORS = ['#60a5fa', '#34d399', '#f59e0b', '#f87171', '#a78bfa', '#fb923c', '#38bdf8', '#e879f9']

// ── helpers ──────────────────────────────────────────────────────────────────

function fmt(v, dp = 4) {
  if (v == null || isNaN(v)) return '—'
  return Number(v).toFixed(dp)
}

function pctBar(value, max) {
  const pct = max === 0 ? 0 : Math.min(Math.abs(value / max) * 100, 100)
  const pos  = value >= 0
  return (
    <div className="flex items-center gap-2">
      <div className="flex-1 bg-gray-700 rounded-full h-1.5 relative">
        <div
          className={`h-1.5 rounded-full ${pos ? 'bg-blue-500' : 'bg-orange-500'}`}
          style={{ width: `${pct}%` }}
        />
      </div>
      <span className={`text-xs font-bold w-14 text-right ${pos ? 'text-blue-400' : 'text-orange-400'}`}>
        {value > 0 ? '+' : ''}{fmt(value, 3)}
      </span>
    </div>
  )
}

// Colour a heatmap cell by importance % and sign of raw weight
function cellStyle(importance, weight) {
  if (importance == null) return {}
  const alpha = Math.min(importance / 40, 1)   // cap at 40% for full saturation
  if (weight >= 0) {
    return { backgroundColor: `rgba(59,130,246,${0.1 + alpha * 0.65})` }   // blue
  }
  return { backgroundColor: `rgba(249,115,22,${0.1 + alpha * 0.65})` }    // orange
}

// ── component ─────────────────────────────────────────────────────────────────

export default function WhatIf() {
  const [runs, setRuns]               = useState([])
  const [datasets, setDatasets]       = useState([])           // all available datasets
  const [selectedRun, setSelectedRun] = useState(null)
  const [colStats, setColStats]       = useState({})           // stats for the run's training dataset

  // Left-panel tab
  const [leftTab, setLeftTab]         = useState('controls')  // 'controls' | 'feature'

  // Controls
  const [varyX, setVaryX]             = useState('')
  const [xMin, setXMin]               = useState('')
  const [xMax, setXMax]               = useState('')
  const [steps, setSteps]             = useState(60)
  const [observeY, setObserveY]       = useState([])
  const [baseline, setBaseline]       = useState({})
  const [showBaseline, setShowBaseline] = useState(false)

  // Baseline dataset selector
  const [baselineDatasetId, setBaselineDatasetId]   = useState(null)   // which dataset to source baseline from
  const [baselineColStats, setBaselineColStats]     = useState({})     // stats for the chosen baseline dataset
  const [baselineStatsLoading, setBaselineStatsLoading] = useState(false)
  const [showDatasetSelector, setShowDatasetSelector] = useState(false)

  // Simulation results
  const [simResult, setSimResult]     = useState(null)
  const [sensResult, setSensResult]   = useState(null)
  const [activeTab, setActiveTab]     = useState('chart')
  const [loading, setLoading]         = useState(false)
  const [sensLoading, setSensLoading] = useState(false)
  const [error, setError]             = useState('')

  // Row picker (Fill from Dataset)
  const [showRowPicker, setShowRowPicker]   = useState(false)
  const [datasetRows, setDatasetRows]       = useState(null)    // { rows, has_timestamp, total_rows }
  const [rowPickerLoading, setRowPickerLoading] = useState(false)
  const [rowSearch, setRowSearch]           = useState('')

  // Feature Analysis
  const [featureWeights, setFeatureWeights] = useState(null)
  const [featureLoading, setFeatureLoading] = useState(false)
  const [featureError, setFeatureError]     = useState('')
  const [focusY, setFocusY]                 = useState('')   // Y to highlight in left panel list
  const [sortMode, setSortMode]             = useState('importance')  // 'importance' | 'alpha'

  // Full-screen table modal
  const [showTableModal, setShowTableModal] = useState(false)

  // Co-varying X selection
  const [covaryX, setCovaryX]             = useState([])         // X cols that co-move with varyX
  const [xBetas, setXBetas]               = useState({})         // col → β (OLS slope on varyX)
  const [xCorrelations, setXCorrelations] = useState({})         // col → Pearson r
  const [couplingLoading, setCouplingLoading] = useState(false)

  // ── Load completed runs + all datasets on mount ──────────────────────────
  useEffect(() => {
    listRuns()
      .then(r => {
        const done = r.data.filter(run => ['done', 'stopped'].includes(run.status))
        setRuns(done)
        if (done.length > 0) selectRun(done[0])
      })
      .catch(() => {})
    listDatasets()
      .then(r => setDatasets(r.data || []))
      .catch(() => {})
  }, [])

  // ── Select a run ─────────────────────────────────────────────────────────
  const selectRun = async (run) => {
    setSelectedRun(run)
    setSimResult(null)
    setSensResult(null)
    setFeatureWeights(null)
    setFeatureError('')
    setError('')
    setDatasetRows(null)
    setShowDatasetSelector(false)

    try {
      const r   = await getDatasetStats(run.dataset_id)
      const map = {}
      for (const cs of (r.data.columns || [])) {
        map[cs.name] = { min: cs.min, max: cs.max, mean: cs.mean, std: cs.std }
      }
      setColStats(map)
      // Baseline dataset defaults to the run's training dataset
      setBaselineDatasetId(run.dataset_id)
      setBaselineColStats(map)

      const bl = {}
      for (const col of (run.x_columns || [])) {
        bl[col] = map[col]?.mean ?? 0
      }
      setBaseline(bl)

      const firstX = run.x_columns?.[0] ?? ''
      setVaryX(firstX)
      if (firstX && map[firstX]) {
        setXMin(map[firstX].min ?? 0)
        setXMax(map[firstX].max ?? 1)
      }
      setCovaryX([])
      setXBetas({})
      setXCorrelations({})
      setObserveY(run.y_columns || [])
      setFocusY(run.y_columns?.[0] ?? '')
    } catch (e) {
      setError('Failed to load dataset stats: ' + e.message)
    }
  }

  // ── Switch baseline dataset ──────────────────────────────────────────────
  const handleSelectBaselineDataset = async (dsId) => {
    setBaselineDatasetId(dsId)
    setShowDatasetSelector(false)
    setDatasetRows(null)   // clear row cache for new dataset
    if (dsId === selectedRun?.dataset_id) {
      setBaselineColStats(colStats)
      return
    }
    setBaselineStatsLoading(true)
    try {
      const r   = await getDatasetStats(dsId)
      const map = {}
      for (const cs of (r.data.columns || [])) {
        map[cs.name] = { min: cs.min, max: cs.max, mean: cs.mean, std: cs.std }
      }
      setBaselineColStats(map)
    } catch {
      setBaselineColStats({})
    }
    setBaselineStatsLoading(false)
  }

  // ── Fetch X coupling betas from training data ────────────────────────────
  const fetchCoupling = async (primaryCol, datasetId, xCols) => {
    if (!primaryCol || !datasetId || !xCols || xCols.length === 0) return
    setCouplingLoading(true)
    try {
      const r = await getXCoupling(datasetId, primaryCol, xCols)
      setXBetas(r.data.betas || {})
      setXCorrelations(r.data.correlations || {})
    } catch (err) {
      setXBetas({})
      setXCorrelations({})
    }
    setCouplingLoading(false)
  }

  // ── Change primary X (the sweep axis) ────────────────────────────────────
  const handleVaryXChange = (col) => {
    setVaryX(col)
    const s = colStats[col]
    if (s) { setXMin(s.min ?? 0); setXMax(s.max ?? 1) }
    setSimResult(null)
    // Remove from covary if it was there
    setCovaryX(prev => prev.filter(c => c !== col))
    // Re-fetch betas for new primary if covaryX has items
    if (covaryX.length > 0 && selectedRun) {
      fetchCoupling(col, selectedRun.dataset_id, selectedRun.x_columns || [])
    }
  }

  // ── Toggle a single X in/out of covaryX ──────────────────────────────────
  const handleToggleCovary = (col) => {
    setSimResult(null)
    setCovaryX(prev => {
      if (prev.includes(col)) return prev.filter(c => c !== col)
      const next = [...prev, col]
      // Fetch betas if this is the first co-vary selection
      if (prev.length === 0 && selectedRun && varyX) {
        fetchCoupling(varyX, selectedRun.dataset_id, selectedRun.x_columns || [])
      }
      return next
    })
  }

  // Quick: co-vary all / fix all
  const handleCovaryAll = () => {
    if (!selectedRun) return
    const others = (selectedRun.x_columns || []).filter(c => c !== varyX)
    setCovaryX(others)
    setSimResult(null)
    if (others.length > 0 && varyX) fetchCoupling(varyX, selectedRun.dataset_id, selectedRun.x_columns || [])
  }
  const handleFixAll = () => {
    setCovaryX([])
    setSimResult(null)
  }

  // ── Run What-If Simulation ───────────────────────────────────────────────
  const handleRunSim = async () => {
    if (!selectedRun || !varyX || observeY.length === 0) return
    setLoading(true)
    setError('')
    try {
      const n  = Math.max(2, Number(steps))
      const mn = Number(xMin)
      const mx = Number(xMax)
      if (mn >= mx) { setError('Min must be less than Max.'); setLoading(false); return }

      const xVals = Array.from({ length: n }, (_, i) => mn + (i / (n - 1)) * (mx - mn))
      const baselineVaryX = Number(baseline[varyX] ?? 0)
      const input_data = {}
      const activeCovary = []
      for (const col of (selectedRun.x_columns || [])) {
        if (col === varyX) {
          input_data[col] = xVals
        } else if (covaryX.includes(col) && xBetas[col] != null) {
          // Co-move: baseline + β × Δ
          input_data[col] = xVals.map(v => Number(baseline[col] ?? 0) + xBetas[col] * (v - baselineVaryX))
          activeCovary.push(col)
        } else {
          input_data[col] = Array(n).fill(Number(baseline[col] ?? 0))
        }
      }

      const r = await whatIfPredict({ model_run_id: selectedRun.id, input_data })
      setSimResult({
        x_values:  xVals,
        x_series:  input_data,          // ALL X values for every step
        x_columns: selectedRun.x_columns || [],
        y_series:  r.data.predicted_values,
        varyX,
        coupled:   activeCovary.length > 0,
        varyCols:  activeCovary,
      })
    } catch (e) {
      setError('Simulation failed: ' + (e.response?.data?.detail || e.message))
    }
    setLoading(false)
  }

  // ── Sensitivity Analysis ─────────────────────────────────────────────────
  const handleSensitivity = async () => {
    if (!selectedRun || observeY.length === 0) return
    setSensLoading(true)
    setError('')
    try {
      const xCols = selectedRun.x_columns || []
      const n     = xCols.length

      const input_data = {}
      for (const col of xCols) {
        const base = Number(baseline[col] ?? 0)
        const perturbed = xCols.map((xc) => {
          const b = Number(baseline[xc] ?? 0)
          return xc === col ? (b !== 0 ? b * 1.1 : b + Math.max(Math.abs(b) * 0.1, 0.01)) : b
        })
        input_data[col] = [base, ...perturbed]
      }

      const r         = await whatIfPredict({ model_run_id: selectedRun.id, input_data })
      const predicted = r.data.predicted_values

      const rows = xCols.map((xc, i) => {
        const base      = Number(baseline[xc] ?? 0)
        const perturbed = base !== 0 ? base * 1.1 : base + Math.max(Math.abs(base) * 0.1, 0.01)
        const row       = { x_col: xc, base_val: base, perturbed_val: perturbed, deltas: {} }
        for (const yc of observeY) {
          const y0 = predicted[yc]?.[0]
          const yi = predicted[yc]?.[i + 1]
          row.deltas[yc] = (y0 != null && yi != null)
            ? { delta: yi - y0, pct: y0 !== 0 ? ((yi - y0) / Math.abs(y0)) * 100 : null }
            : null
        }
        return row
      })

      const firstY = observeY[0]
      rows.sort((a, b) =>
        Math.abs(b.deltas[firstY]?.delta ?? 0) - Math.abs(a.deltas[firstY]?.delta ?? 0)
      )

      setSensResult(rows)
      setActiveTab('sensitivity')
    } catch (e) {
      setError('Sensitivity analysis failed: ' + (e.response?.data?.detail || e.message))
    }
    setSensLoading(false)
  }

  // ── Feature Weights (Jacobian) ───────────────────────────────────────────
  const handleLoadFeatureWeights = async () => {
    if (!selectedRun) return
    setFeatureLoading(true)
    setFeatureError('')
    try {
      const r = await getFeatureWeights(selectedRun.id)
      setFeatureWeights(r.data)
    } catch (e) {
      setFeatureError('Failed: ' + (e.response?.data?.detail || e.message))
    }
    setFeatureLoading(false)
  }

  // ── Row Picker: Fill Baseline from Dataset ──────────────────────────────
  const handleOpenRowPicker = async () => {
    if (!selectedRun) return
    setShowRowPicker(true)
    setRowSearch('')
    if (datasetRows) return   // already loaded for current dataset
    setRowPickerLoading(true)
    try {
      const dsId = baselineDatasetId ?? selectedRun.dataset_id
      const r = await sampleDatasetRows(dsId, selectedRun.x_columns || [], 150)
      setDatasetRows(r.data)
    } catch {
      setDatasetRows(null)
    }
    setRowPickerLoading(false)
  }

  const handleSelectRow = (row) => {
    const newBaseline = { ...baseline }
    for (const [col, val] of Object.entries(row.x_values)) {
      if (val != null) newBaseline[col] = val
    }
    setBaseline(newBaseline)
    setShowRowPicker(false)
  }

  // Filtered rows for picker
  const filteredRows = datasetRows?.rows?.filter(row => {
    if (!rowSearch.trim()) return true
    const q = rowSearch.toLowerCase()
    if (String(row.row_index + 1).includes(q)) return true
    if (row.timestamp && row.timestamp.toLowerCase().includes(q)) return true
    return false
  }) ?? []

  // ── Chart data ───────────────────────────────────────────────────────────
  const chartData = (simResult?.x_values ?? []).map((xv, i) => {
    const row = { x: Number(xv.toFixed(4)) }
    // Y outputs
    for (const col of observeY) {
      const v = simResult.y_series[col]?.[i]
      row[col] = v != null ? Number(Number(v).toFixed(4)) : null
    }
    // All X values (for table display)
    for (const col of (simResult?.x_columns ?? [])) {
      const v = simResult.x_series?.[col]?.[i]
      row[`__x__${col}`] = v != null ? Number(Number(v).toFixed(6)) : null
    }
    return row
  })

  // ── Download What-If CSV ─────────────────────────────────────────────────
  const downloadWhatIfCSV = () => {
    if (!simResult) return
    const xCols  = simResult.x_columns || [simResult.varyX]
    const yCols  = observeY.length > 0 ? observeY : Object.keys(simResult.y_series)
    const varySet = new Set(simResult.varyCols || [])
    const primary  = simResult.varyX

    // Build headers: for each varying X → value + direction; fixed X → value only; each Y → value + direction
    const headers = []
    for (const col of xCols) {
      headers.push(col)
      if (col === primary || varySet.has(col)) headers.push(`${col}_Direction`)
    }
    for (const col of yCols) {
      headers.push(col)
      headers.push(`${col}_Direction`)
    }

    const csvRows = (simResult.x_values ?? []).map((_, i) => {
      const row = []
      // X values + direction
      for (const col of xCols) {
        const v = simResult.x_series?.[col]?.[i]
        const num = v != null ? Number(v) : null
        row.push(num != null ? num.toFixed(6) : '')
        if (col === primary || varySet.has(col)) {
          row.push(dirInfo(num, baselineXVals[col]).label)
        }
      }
      // Y values + direction
      for (const col of yCols) {
        const v = simResult.y_series[col]?.[i]
        const num = v != null ? Number(v) : null
        row.push(num != null ? num.toFixed(6) : '')
        row.push(dirInfo(num, baselineYVals[col]).label)
      }
      return row
    })

    const csv = [headers, ...csvRows].map(r => r.join(',')).join('\n')
    const blob = new Blob([csv], { type: 'text/csv' })
    const url  = URL.createObjectURL(blob)
    const a    = document.createElement('a')
    a.href     = url
    a.download = `whatif_${simResult.varyX}_${new Date().toISOString().slice(0,19).replace(/:/g,'-')}.csv`
    a.click()
    URL.revokeObjectURL(url)
  }

  const baselineXVal = varyX ? Number(baseline[varyX] ?? 0) : null

  // ── Direction helpers (vs baseline) ─────────────────────────────────────
  // Find baseline row index in chartData (closest step to the baseline X value)
  const baselineRowIdx = (() => {
    if (!simResult || baselineXVal == null) return -1
    let best = -1, bestDist = Infinity
    simResult.x_values.forEach((v, i) => {
      const d = Math.abs(v - baselineXVal)
      if (d < bestDist) { bestDist = d; best = i }
    })
    return best
  })()

  // Baseline Y values (at the closest step to baseline X)
  const baselineYVals = (() => {
    const out = {}
    if (baselineRowIdx < 0 || !simResult) return out
    for (const col of observeY) {
      const v = simResult.y_series[col]?.[baselineRowIdx]
      if (v != null) out[col] = Number(v)
    }
    return out
  })()

  // Baseline X values
  const baselineXVals = (() => {
    const out = {}
    if (!simResult) return out
    for (const col of (simResult.x_columns || [])) {
      out[col] = Number(baseline[col] ?? 0)
    }
    return out
  })()

  function dirInfo(current, ref) {
    if (current == null || ref == null) return { sym: '', label: '', cls: 'text-gray-600' }
    const diff = current - ref
    const absPct = ref !== 0 ? Math.abs(diff / ref) : Math.abs(diff)
    if (absPct < 0.00005) return { sym: '→', label: 'No change', cls: 'text-gray-500' }
    if (diff > 0)         return { sym: '↑', label: 'Increase',  cls: 'text-green-400' }
    return                       { sym: '↓', label: 'Decrease',  cls: 'text-red-400'   }
  }

  const sensMaxDelta = {}
  if (sensResult) {
    for (const yc of observeY) {
      sensMaxDelta[yc] = Math.max(
        ...sensResult.map(r => Math.abs(r.deltas[yc]?.delta ?? 0)), 0.001
      )
    }
  }

  // ── Feature analysis derived data ────────────────────────────────────────
  const xColsSorted = featureWeights
    ? (sortMode === 'alpha'
        ? [...featureWeights.x_cols].sort()
        : [...featureWeights.x_cols].sort((a, b) => {
            const ia = featureWeights.importance[a]?.[focusY] ?? 0
            const ib = featureWeights.importance[b]?.[focusY] ?? 0
            return ib - ia
          })
      )
    : []

  // ── Render ───────────────────────────────────────────────────────────────
  return (
    <div className="p-8 max-w-screen-xl mx-auto">
      {/* Page header */}
      <h1 className="text-2xl font-bold text-white mb-1">🔬 What-If Simulator</h1>
      <p className="text-gray-200 text-sm mb-6">
        Vary one X input across a range — hold all others at baseline — and observe how
        Y outputs respond. Use Sensitivity Analysis to rank which inputs have the most impact.
      </p>

      <div className="flex gap-6 items-start">

        {/* ══════════════════════════════════════════════════════════════
            LEFT CONTROL PANEL
        ══════════════════════════════════════════════════════════════ */}
        <div className="w-80 flex-shrink-0 space-y-4">

          {/* Left-panel tab switcher */}
          <div className="flex bg-gray-800 rounded-xl p-1 gap-1">
            {[
              { key: 'controls', label: '⚙ Controls' },
              { key: 'feature',  label: '⚖ Feature Analysis' },
            ].map(t => (
              <button
                key={t.key}
                onClick={() => setLeftTab(t.key)}
                className={`flex-1 py-2 rounded-lg text-xs font-semibold transition-colors ${
                  leftTab === t.key
                    ? 'bg-blue-600 text-white'
                    : 'text-gray-400 hover:text-white'
                }`}
              >
                {t.label}
              </button>
            ))}
          </div>

          {/* ── CONTROLS TAB ── */}
          {leftTab === 'controls' && (<>

            {/* 1 — Model run */}
            <div className="bg-gray-800 rounded-xl p-4">
              <h2 className="text-white font-semibold text-sm mb-3 flex items-center gap-2">
                <span className="bg-blue-600 text-white text-xs rounded-full w-5 h-5 flex items-center justify-center font-bold">1</span>
                Model Run
              </h2>
              {runs.length === 0 ? (
                <p className="text-gray-500 text-xs">No completed runs. Train a model first.</p>
              ) : (
                <div className="space-y-1.5 max-h-52 overflow-y-auto pr-1">
                  {runs.map(run => (
                    <button
                      key={run.id}
                      onClick={() => selectRun(run)}
                      className={`w-full text-left px-3 py-2.5 rounded-lg text-xs transition-colors border ${
                        selectedRun?.id === run.id
                          ? 'bg-blue-700 border-blue-500 text-white'
                          : 'bg-gray-700 border-transparent text-gray-100 hover:border-blue-500'
                      }`}
                    >
                      <div className="font-semibold">Run #{run.id}</div>
                      <div className="text-gray-200 text-xs mt-0.5">
                        {run.x_columns?.length ?? 0}X → {run.y_columns?.length ?? 0}Y
                        <span className={`ml-2 font-medium ${run.status === 'done' ? 'text-green-400' : 'text-orange-400'}`}>
                          {run.status}
                        </span>
                      </div>
                      {run.metrics?.r2_score != null && (
                        <div className="text-gray-500 mt-0.5">R² {run.metrics.r2_score.toFixed(4)}</div>
                      )}
                    </button>
                  ))}
                </div>
              )}
            </div>

            {selectedRun && (<>

              {/* 2 — Vary X */}
              <div className="bg-gray-800 rounded-xl p-4">

                {/* ── Section A: Primary sweep X ── */}
                <div className="mb-3">
                  <div className="flex items-center gap-2 mb-2">
                    <span className="bg-blue-600 text-white text-xs rounded-full w-5 h-5 flex items-center justify-center font-bold flex-shrink-0">2</span>
                    <h2 className="text-white font-semibold text-sm">Sweep X (Primary)</h2>
                  </div>
                  <p className="text-gray-500 text-xs mb-2">Click to select which X to sweep across its range.</p>
                  <div className="space-y-1 max-h-40 overflow-y-auto pr-1">
                    {(selectedRun.x_columns || []).map(col => {
                      const s = colStats[col]
                      const isPrimary = col === varyX
                      return (
                        <button
                          key={col}
                          onClick={() => handleVaryXChange(col)}
                          className={`w-full text-left px-2.5 py-1.5 rounded-lg text-xs transition-colors border ${
                            isPrimary
                              ? 'bg-blue-700 border-blue-500 text-white'
                              : 'bg-gray-700 border-transparent text-gray-200 hover:border-blue-400'
                          }`}
                        >
                          <span className="font-medium">{col}</span>
                          {s && (
                            <span className="text-gray-400 ml-1.5">[{fmt(s.min,2)} – {fmt(s.max,2)}]</span>
                          )}
                        </button>
                      )
                    })}
                  </div>
                </div>

                {/* ── Variation range ── */}
                {varyX && (
                  <div className="pt-3 border-t border-gray-700 space-y-2 mb-3">
                    <p className="text-blue-300 text-xs font-medium">Range for <span className="font-bold">{varyX}</span></p>
                    <div className="grid grid-cols-2 gap-2">
                      {[['Min', xMin, setXMin], ['Max', xMax, setXMax]].map(([lbl, val, set]) => (
                        <div key={lbl}>
                          <label className="text-gray-500 text-xs">{lbl}</label>
                          <input
                            type="number"
                            value={val}
                            onChange={e => { set(e.target.value); setSimResult(null) }}
                            className="w-full bg-gray-700 border border-gray-600 rounded px-2 py-1
                                       text-white text-xs focus:border-blue-500 focus:outline-none mt-0.5"
                          />
                        </div>
                      ))}
                    </div>
                    <div>
                      <label className="text-gray-500 text-xs">
                        Data Points: <span className="text-white font-medium">{steps}</span>
                      </label>
                      <input
                        type="range" min="20" max="200" step="10"
                        value={steps}
                        onChange={e => { setSteps(Number(e.target.value)); setSimResult(null) }}
                        className="w-full accent-blue-500 mt-1"
                      />
                    </div>
                  </div>
                )}

                {/* ── Section B: Co-varying X ── */}
                {varyX && (selectedRun.x_columns || []).filter(c => c !== varyX).length > 0 && (
                  <div className="pt-3 border-t border-gray-700">
                    <div className="flex items-center justify-between mb-1.5">
                      <div>
                        <p className="text-gray-100 text-xs font-medium">Co-varying X (weighted by training correlation)</p>
                        <p className="text-gray-500 text-xs mt-0.5">
                          Checked X variables shift proportionally with <span className="text-blue-300">{varyX}</span> using β from training data.
                        </p>
                      </div>
                    </div>

                    {/* Quick actions */}
                    <div className="flex gap-1.5 mb-2">
                      <button
                        onClick={handleCovaryAll}
                        className="text-xs px-2.5 py-1 rounded border border-teal-700 text-teal-400 hover:bg-teal-900/30 transition-colors"
                      >
                        ✓ Vary All
                      </button>
                      <button
                        onClick={handleFixAll}
                        className="text-xs px-2.5 py-1 rounded border border-gray-600 text-gray-400 hover:bg-gray-700 transition-colors"
                      >
                        — Fix All
                      </button>
                      {covaryX.length > 0 && (
                        <span className="ml-auto text-xs text-teal-400 self-center">
                          {covaryX.length} co-varying
                        </span>
                      )}
                    </div>

                    {couplingLoading && (
                      <p className="text-teal-400 text-xs animate-pulse mb-2">Computing coupling weights from training data…</p>
                    )}

                    {/* Per-X checkboxes */}
                    <div className="space-y-1 max-h-48 overflow-y-auto pr-1">
                      {(selectedRun.x_columns || []).filter(c => c !== varyX).map(col => {
                        const isChecked = covaryX.includes(col)
                        const corr = xCorrelations[col]
                        const beta = xBetas[col]
                        const s = colStats[col]
                        return (
                          <label
                            key={col}
                            className={`flex items-start gap-2 px-2.5 py-1.5 rounded-lg border cursor-pointer transition-colors ${
                              isChecked
                                ? 'border-teal-600 bg-teal-900/20'
                                : 'border-gray-700 bg-gray-700/40 hover:border-gray-500'
                            }`}
                          >
                            <input
                              type="checkbox"
                              checked={isChecked}
                              onChange={() => handleToggleCovary(col)}
                              className="mt-0.5 accent-teal-500 flex-shrink-0"
                            />
                            <div className="flex-1 min-w-0">
                              <div className="flex items-center justify-between gap-1">
                                <span className={`text-xs font-medium truncate ${isChecked ? 'text-teal-300' : 'text-gray-300'}`} title={col}>
                                  {col}
                                </span>
                                {s && (
                                  <span className="text-gray-500 text-xs flex-shrink-0">[{fmt(s.min,2)}–{fmt(s.max,2)}]</span>
                                )}
                              </div>
                              {/* Correlation + beta shown when checked and data available */}
                              {isChecked && corr != null && (
                                <div className="flex items-center gap-2 mt-1">
                                  <div className="flex-1 bg-gray-600 rounded-full h-1">
                                    <div
                                      className={`h-1 rounded-full ${corr >= 0 ? 'bg-teal-400' : 'bg-orange-400'}`}
                                      style={{ width: `${Math.min(Math.abs(corr) * 100, 100)}%` }}
                                    />
                                  </div>
                                  <span className={`text-xs font-mono flex-shrink-0 ${corr >= 0 ? 'text-teal-400' : 'text-orange-400'}`}>
                                    r={corr >= 0 ? '+' : ''}{corr.toFixed(2)}
                                  </span>
                                  {beta != null && (
                                    <span className={`text-xs font-mono flex-shrink-0 ${beta >= 0 ? 'text-teal-300' : 'text-orange-300'}`}>
                                      β={beta >= 0 ? '+' : ''}{beta.toFixed(3)}
                                    </span>
                                  )}
                                </div>
                              )}
                              {isChecked && corr == null && !couplingLoading && (
                                <p className="text-gray-500 text-xs mt-0.5">Correlation not yet loaded</p>
                              )}
                            </div>
                          </label>
                        )
                      })}
                    </div>
                  </div>
                )}
              </div>

              {/* 3 — Observe Y */}
              <div className="bg-gray-800 rounded-xl p-4">
                <div className="flex items-center justify-between mb-3">
                  <h2 className="text-white font-semibold text-sm flex items-center gap-2">
                    <span className="bg-green-600 text-white text-xs rounded-full w-5 h-5 flex items-center justify-center font-bold">3</span>
                    Observe Y Outputs
                  </h2>
                  <div className="flex gap-2">
                    <button
                      onClick={() => setObserveY(selectedRun.y_columns || [])}
                      className="text-xs text-gray-200 hover:text-white transition-colors"
                    >All</button>
                    <button
                      onClick={() => setObserveY([])}
                      className="text-xs text-gray-200 hover:text-white transition-colors"
                    >None</button>
                  </div>
                </div>
                <div className="space-y-2">
                  {(selectedRun.y_columns || []).map((col, idx) => (
                    <label key={col} className="flex items-center gap-2.5 cursor-pointer group">
                      <input
                        type="checkbox"
                        checked={observeY.includes(col)}
                        onChange={() => setObserveY(p =>
                          p.includes(col) ? p.filter(c => c !== col) : [...p, col]
                        )}
                        className="w-4 h-4 accent-green-500 flex-shrink-0"
                      />
                      <span className="text-xs text-gray-100 group-hover:text-white transition-colors flex-1 leading-snug">
                        {col}
                      </span>
                      <span
                        className="w-3 h-3 rounded-full flex-shrink-0 ring-1 ring-black/20"
                        style={{ backgroundColor: COLORS[idx % COLORS.length] }}
                      />
                    </label>
                  ))}
                </div>
              </div>

              {/* 4 — Baseline X values */}
              <div className="bg-gray-800 rounded-xl p-4">
                {/* Header row */}
                <button
                  onClick={() => setShowBaseline(p => !p)}
                  className="w-full flex items-center justify-between"
                >
                  <h2 className="text-white font-semibold text-sm flex items-center gap-2">
                    <span className="bg-gray-600 text-white text-xs rounded-full w-5 h-5 flex items-center justify-center font-bold">4</span>
                    Baseline X Values
                  </h2>
                  <span className="text-gray-500 text-xs">{showBaseline ? '▲ hide' : '▼ show'}</span>
                </button>
                <p className="text-gray-500 text-xs mt-1">
                  All X variables except the varied one are held at these values.
                </p>

                {/* ── Dataset selector ── */}
                <div className="mt-2.5 border border-gray-700 rounded-lg overflow-hidden">
                  {/* Current dataset badge */}
                  <button
                    onClick={() => setShowDatasetSelector(p => !p)}
                    className="w-full flex items-center justify-between px-3 py-2 bg-gray-750 hover:bg-gray-700 transition-colors"
                    style={{ backgroundColor: '#1f2937' }}
                  >
                    <div className="flex items-center gap-2 min-w-0">
                      <span className="text-gray-500 text-xs flex-shrink-0">Dataset:</span>
                      <span className="text-xs font-medium text-blue-300 truncate">
                        {datasets.find(d => d.id === baselineDatasetId)?.original_name ?? '…'}
                      </span>
                      {baselineDatasetId !== selectedRun?.dataset_id && (
                        <span className="text-xs bg-amber-700/50 text-amber-300 rounded px-1.5 py-0.5 flex-shrink-0">
                          custom
                        </span>
                      )}
                      {baselineDatasetId === selectedRun?.dataset_id && (
                        <span className="text-xs bg-green-900/50 text-green-400 rounded px-1.5 py-0.5 flex-shrink-0">
                          training
                        </span>
                      )}
                    </div>
                    <span className="text-gray-500 text-xs flex-shrink-0 ml-2">
                      {showDatasetSelector ? '▲' : '▼'}
                    </span>
                  </button>

                  {/* Dataset list dropdown */}
                  {showDatasetSelector && (
                    <div className="border-t border-gray-700 max-h-44 overflow-y-auto">
                      {baselineStatsLoading && (
                        <div className="px-3 py-2 text-gray-500 text-xs">Loading stats…</div>
                      )}
                      {datasets.map(ds => {
                        const isTraining = ds.id === selectedRun?.dataset_id
                        const isCurrent  = ds.id === baselineDatasetId
                        return (
                          <button
                            key={ds.id}
                            onClick={() => handleSelectBaselineDataset(ds.id)}
                            className={`w-full text-left px-3 py-2.5 text-xs transition-colors border-b border-gray-700/50 last:border-0 ${
                              isCurrent
                                ? 'bg-blue-900/30 text-white'
                                : 'hover:bg-gray-700 text-gray-300'
                            }`}
                          >
                            <div className="flex items-center gap-1.5 min-w-0">
                              {isCurrent && <span className="text-blue-400">✓</span>}
                              <span className="truncate font-medium">{ds.original_name}</span>
                              {isTraining && (
                                <span className="text-green-400 text-xs flex-shrink-0">(training)</span>
                              )}
                            </div>
                            <div className="text-gray-500 text-xs mt-0.5">
                              {ds.row_count?.toLocaleString() ?? '?'} rows
                              {ds.x_columns?.length ? ` · ${ds.x_columns.length}X` : ''}
                            </div>
                          </button>
                        )
                      })}
                    </div>
                  )}
                </div>

                {/* Quick-action buttons */}
                <div className="flex gap-2 mt-2">
                  <button
                    onClick={() => {
                      const bl = {}
                      for (const col of (selectedRun.x_columns || [])) {
                        bl[col] = baselineColStats[col]?.mean ?? colStats[col]?.mean ?? 0
                      }
                      setBaseline(bl)
                    }}
                    className="flex-1 text-xs text-blue-400 hover:text-blue-300 border border-blue-800
                               hover:border-blue-600 rounded py-1.5 transition-colors"
                  >
                    ↺ Reset to Mean
                  </button>
                  <button
                    onClick={handleOpenRowPicker}
                    className="flex-1 text-xs text-emerald-400 hover:text-emerald-300 border border-emerald-800
                               hover:border-emerald-600 rounded py-1.5 transition-colors"
                  >
                    📂 From Dataset
                  </button>
                </div>

                {/* Collapsible per-variable inputs */}
                {showBaseline && (
                  <div className="mt-3 space-y-2.5 max-h-64 overflow-y-auto pr-1">
                    {(selectedRun.x_columns || []).filter(c => c !== varyX).map(col => {
                      const bStat = baselineColStats[col]
                      const tStat = colStats[col]
                      return (
                        <div key={col}>
                          <div className="flex items-center justify-between">
                            <label className="text-gray-200 text-xs truncate flex-1 mr-2" title={col}>{col}</label>
                            <button
                              onClick={() => setBaseline(p => ({
                                ...p,
                                [col]: bStat?.mean ?? tStat?.mean ?? 0,
                              }))}
                              title="Reset to dataset mean"
                              className="text-xs text-gray-600 hover:text-blue-400 transition-colors flex-shrink-0"
                            >↺</button>
                          </div>
                          <input
                            type="number"
                            step="any"
                            value={baseline[col] ?? ''}
                            onChange={e => setBaseline(p => ({
                              ...p,
                              [col]: e.target.value === '' ? '' : e.target.value,
                            }))}
                            className="w-full bg-gray-700 border border-gray-600 rounded px-2 py-1
                                       text-white text-xs focus:border-blue-500 focus:outline-none mt-0.5"
                          />
                          {(bStat || tStat) && (
                            <p className="text-gray-600 text-xs mt-0.5">
                              μ {fmt((bStat ?? tStat).mean, 3)} · [{fmt((bStat ?? tStat).min, 2)}, {fmt((bStat ?? tStat).max, 2)}]
                            </p>
                          )}
                        </div>
                      )
                    })}
                  </div>
                )}
              </div>

              {/* Action buttons */}
              <button
                onClick={handleRunSim}
                disabled={loading || !varyX || observeY.length === 0}
                className="w-full bg-blue-600 hover:bg-blue-500 disabled:opacity-40
                           disabled:cursor-not-allowed text-white py-3 rounded-xl
                           font-bold text-sm transition-colors"
              >
                {loading ? '⏳ Simulating…' : `▶ Run What-If  (${steps} points)`}
              </button>

              <button
                onClick={handleSensitivity}
                disabled={sensLoading || observeY.length === 0}
                className="w-full bg-purple-700 hover:bg-purple-600 disabled:opacity-40
                           disabled:cursor-not-allowed text-white py-2.5 rounded-xl
                           font-semibold text-sm transition-colors"
              >
                {sensLoading
                  ? '⏳ Analysing…'
                  : `📊 Sensitivity Analysis  (all ${selectedRun.x_columns?.length ?? 0} inputs)`}
              </button>

            </>)}

            {error && (
              <div className="bg-red-900/40 border border-red-700 rounded-xl px-4 py-3">
                <p className="text-red-400 text-xs">{error}</p>
              </div>
            )}
          </>)}

          {/* ── FEATURE ANALYSIS TAB ── */}
          {leftTab === 'feature' && (<>

            {/* Run selector (compact) */}
            <div className="bg-gray-800 rounded-xl p-4">
              <h2 className="text-white font-semibold text-sm mb-3">Model Run</h2>
              {runs.length === 0 ? (
                <p className="text-gray-500 text-xs">No completed runs.</p>
              ) : (
                <div className="space-y-1.5 max-h-40 overflow-y-auto pr-1">
                  {runs.map(run => (
                    <button
                      key={run.id}
                      onClick={() => selectRun(run)}
                      className={`w-full text-left px-3 py-2 rounded-lg text-xs transition-colors border ${
                        selectedRun?.id === run.id
                          ? 'bg-blue-700 border-blue-500 text-white'
                          : 'bg-gray-700 border-transparent text-gray-100 hover:border-blue-500'
                      }`}
                    >
                      <span className="font-semibold">Run #{run.id}</span>
                      <span className="text-gray-400 ml-2">
                        {run.x_columns?.length ?? 0}X → {run.y_columns?.length ?? 0}Y
                      </span>
                      {run.metrics?.r2_score != null && (
                        <span className="text-gray-500 ml-2">R² {run.metrics.r2_score.toFixed(3)}</span>
                      )}
                    </button>
                  ))}
                </div>
              )}
            </div>

            {selectedRun && (<>

              {/* Compute button */}
              <button
                onClick={handleLoadFeatureWeights}
                disabled={featureLoading}
                className="w-full bg-indigo-700 hover:bg-indigo-600 disabled:opacity-40
                           disabled:cursor-not-allowed text-white py-3 rounded-xl
                           font-bold text-sm transition-colors"
              >
                {featureLoading ? '⏳ Computing…' : '⚖ Compute Feature Weights'}
              </button>

              {featureError && (
                <div className="bg-red-900/40 border border-red-700 rounded-xl px-4 py-3">
                  <p className="text-red-400 text-xs">{featureError}</p>
                </div>
              )}

              {featureWeights && (<>

                {/* Y focus selector */}
                <div className="bg-gray-800 rounded-xl p-4">
                  <p className="text-gray-400 text-xs mb-2 font-medium">Focus Y Output</p>
                  <div className="space-y-1 max-h-36 overflow-y-auto pr-1">
                    {featureWeights.y_cols.map((yc, idx) => (
                      <button
                        key={yc}
                        onClick={() => setFocusY(yc)}
                        className={`w-full text-left px-3 py-1.5 rounded-lg text-xs border transition-colors ${
                          focusY === yc
                            ? 'border-blue-500 bg-blue-900/30 text-white'
                            : 'border-transparent bg-gray-700 text-gray-300 hover:border-blue-600'
                        }`}
                      >
                        <span
                          className="inline-block w-2 h-2 rounded-full mr-2"
                          style={{ backgroundColor: COLORS[idx % COLORS.length] }}
                        />
                        {yc}
                      </button>
                    ))}
                  </div>
                </div>

                {/* Sort toggle */}
                <div className="flex gap-2">
                  {[['importance', '⬇ By Impact'], ['alpha', 'A–Z']].map(([k, lbl]) => (
                    <button
                      key={k}
                      onClick={() => setSortMode(k)}
                      className={`flex-1 py-1.5 rounded-lg text-xs font-medium border transition-colors ${
                        sortMode === k
                          ? 'bg-gray-600 border-gray-500 text-white'
                          : 'bg-gray-800 border-gray-700 text-gray-400 hover:text-white'
                      }`}
                    >
                      {lbl}
                    </button>
                  ))}
                </div>

                {/* Ranked list for focusY */}
                {focusY && (
                  <div className="bg-gray-800 rounded-xl p-4">
                    <p className="text-gray-400 text-xs mb-3 font-medium">
                      X impact on <span className="text-blue-300">{focusY}</span>
                    </p>
                    <div className="space-y-2.5 max-h-72 overflow-y-auto pr-1">
                      {xColsSorted.map((xc, rank) => {
                        const imp = featureWeights.importance[xc]?.[focusY] ?? 0
                        const w   = featureWeights.weights[xc]?.[focusY] ?? 0
                        const pos = w >= 0
                        return (
                          <div key={xc}>
                            <div className="flex items-center justify-between mb-0.5">
                              <span className="text-gray-200 text-xs truncate flex-1 mr-2" title={xc}>
                                <span className="text-gray-500 mr-1">#{rank + 1}</span>{xc}
                              </span>
                              <span className={`text-xs font-bold flex-shrink-0 ${pos ? 'text-blue-400' : 'text-orange-400'}`}>
                                {pos ? '+' : ''}{imp.toFixed(1)}%
                              </span>
                            </div>
                            <div className="bg-gray-700 rounded-full h-1.5">
                              <div
                                className={`h-1.5 rounded-full ${pos ? 'bg-blue-500' : 'bg-orange-500'}`}
                                style={{ width: `${Math.min(imp * 2.5, 100)}%` }}
                              />
                            </div>
                          </div>
                        )
                      })}
                    </div>
                    <p className="text-gray-600 text-xs mt-3">
                      Blue = positive influence · Orange = negative
                    </p>
                  </div>
                )}

              </>)}
            </>)}
          </>)}

        </div>

        {/* ══════════════════════════════════════════════════════════════
            RIGHT RESULTS PANEL
        ══════════════════════════════════════════════════════════════ */}
        <div className="flex-1 min-w-0">

          {/* ── FEATURE ANALYSIS HEATMAP ── */}
          {leftTab === 'feature' && (
            <>
              {!featureWeights ? (
                <div className="bg-gray-800 rounded-xl flex flex-col items-center justify-center py-24 text-center">
                  <span className="text-6xl mb-5">⚖</span>
                  <p className="text-gray-100 font-semibold mb-2">Feature Weight Analysis</p>
                  <p className="text-gray-500 text-sm max-w-xs">
                    Select a model run and click <strong className="text-indigo-400">Compute Feature Weights</strong> to
                    see how each X input drives each Y output — computed via the model's Jacobian at the dataset mean.
                  </p>
                </div>
              ) : (
                <div className="bg-gray-800 rounded-xl p-5">
                  <div className="flex items-start justify-between mb-4">
                    <div>
                      <h2 className="text-white font-semibold text-base">
                        Feature Weight Matrix
                        <span className="text-gray-400 font-normal text-sm ml-2">— Run #{selectedRun?.id}</span>
                      </h2>
                      <p className="text-gray-400 text-xs mt-0.5">
                        ∂Y/∂X Jacobian at dataset mean.
                        Cell colour intensity = importance %. Blue = positive, Orange = negative influence.
                      </p>
                    </div>
                    <div className="flex items-center gap-3 text-xs text-gray-500 flex-shrink-0 mt-1">
                      <span className="flex items-center gap-1">
                        <span className="inline-block w-8 h-3 rounded" style={{ background: 'rgba(59,130,246,0.7)' }} />
                        Positive
                      </span>
                      <span className="flex items-center gap-1">
                        <span className="inline-block w-8 h-3 rounded" style={{ background: 'rgba(249,115,22,0.7)' }} />
                        Negative
                      </span>
                    </div>
                  </div>

                  {/* Heatmap table */}
                  <div className="overflow-auto rounded-lg border border-gray-700">
                    <table className="w-full text-xs border-collapse">
                      <thead className="sticky top-0 bg-gray-750 z-10">
                        <tr className="bg-gray-700">
                          <th className="px-3 py-2.5 text-left text-gray-300 font-semibold whitespace-nowrap border-r border-gray-600">
                            X Input
                          </th>
                          {featureWeights.y_cols.map((yc, idx) => (
                            <th
                              key={yc}
                              className="px-3 py-2.5 text-center font-semibold whitespace-nowrap min-w-28"
                              style={{ color: COLORS[idx % COLORS.length] }}
                            >
                              {yc}
                            </th>
                          ))}
                          <th className="px-3 py-2.5 text-center text-gray-400 font-semibold whitespace-nowrap min-w-24">
                            Avg Impact
                          </th>
                        </tr>
                      </thead>
                      <tbody>
                        {xColsSorted.map((xc, ri) => {
                          const avgImp = featureWeights.y_cols.length > 0
                            ? featureWeights.y_cols.reduce((s, yc) =>
                                s + (featureWeights.importance[xc]?.[yc] ?? 0), 0
                              ) / featureWeights.y_cols.length
                            : 0
                          return (
                            <tr
                              key={xc}
                              className={`border-t border-gray-700 ${ri % 2 === 0 ? 'bg-gray-800' : 'bg-gray-800/50'}`}
                            >
                              <td className="px-3 py-2 text-gray-100 font-medium whitespace-nowrap border-r border-gray-700">
                                {xc}
                              </td>
                              {featureWeights.y_cols.map(yc => {
                                const imp = featureWeights.importance[xc]?.[yc]
                                const w   = featureWeights.weights[xc]?.[yc]
                                const sw  = featureWeights.std_weights[xc]?.[yc]
                                return (
                                  <td
                                    key={yc}
                                    className="px-3 py-2 text-center"
                                    style={cellStyle(imp, w)}
                                    title={`dY/dX (orig): ${w != null ? w.toFixed(4) : '—'}\ndY/dX (std): ${sw != null ? sw.toFixed(4) : '—'}`}
                                  >
                                    <div className={`font-bold text-xs ${
                                      w >= 0 ? 'text-blue-200' : 'text-orange-200'
                                    }`}>
                                      {imp != null ? `${imp.toFixed(1)}%` : '—'}
                                    </div>
                                    <div className="text-gray-400 text-xs mt-0.5">
                                      {w != null ? (w >= 0 ? '+' : '') + w.toFixed(3) : '—'}
                                    </div>
                                  </td>
                                )
                              })}
                              <td className="px-3 py-2 text-center">
                                <div className="text-gray-200 font-semibold text-xs">
                                  {avgImp.toFixed(1)}%
                                </div>
                                <div className="mt-1 bg-gray-700 rounded-full h-1 w-16 mx-auto">
                                  <div
                                    className="h-1 rounded-full bg-indigo-500"
                                    style={{ width: `${Math.min(avgImp * 2.5, 100)}%` }}
                                  />
                                </div>
                              </td>
                            </tr>
                          )
                        })}
                      </tbody>
                    </table>
                  </div>

                  {/* Footer legend */}
                  <p className="text-gray-600 text-xs mt-3 border-t border-gray-700 pt-3">
                    <strong className="text-gray-400">%</strong> = normalised |weight| share per Y column (sums to 100% per Y).
                    &nbsp;<strong className="text-gray-400">Value</strong> = raw ∂Y/∂X in original units.
                    &nbsp;Hover a cell to see standardised weight (∂Y_std/∂X_std).
                    &nbsp;Sorted by impact on <span className="text-blue-300">{focusY || featureWeights.y_cols[0]}</span>.
                  </p>
                </div>
              )}
            </>
          )}

          {/* ── CONTROLS TAB RESULTS ── */}
          {leftTab === 'controls' && (<>

            {/* Empty state */}
            {!simResult && !sensResult && (
              <div className="bg-gray-800 rounded-xl flex flex-col items-center justify-center py-24 text-center">
                <span className="text-6xl mb-5">🔬</span>
                <p className="text-gray-100 font-semibold mb-2">
                  {selectedRun
                    ? `Run ${selectedRun.x_columns?.length ?? 0} X inputs → ${selectedRun.y_columns?.length ?? 0} Y outputs`
                    : 'Select a completed model run'}
                </p>
                <p className="text-gray-500 text-sm">
                  {selectedRun
                    ? 'Choose which X to vary, pick Y outputs to observe, then click Run What-If.'
                    : 'Train a model first, then return here.'}
                </p>
              </div>
            )}

            {/* Results */}
            {(simResult || sensResult) && (
              <div className="bg-gray-800 rounded-xl p-5">

                {/* Tab bar */}
                <div className="flex items-center justify-between mb-5">
                  <div>
                    {simResult && (
                      <div>
                        <h2 className="text-white font-semibold flex items-center gap-2 flex-wrap">
                          Effect of <span className="text-blue-400">{simResult.varyX}</span> on Y Outputs
                          {simResult.coupled && (
                            <span className="text-xs bg-teal-900 text-teal-300 border border-teal-700
                                             px-2 py-0.5 rounded-full font-normal">
                              ⟳ {simResult.varyCols?.length} X co-varying
                            </span>
                          )}
                        </h2>
                        {simResult.coupled && simResult.varyCols?.length > 0 && (
                          <p className="text-xs text-teal-400 mt-0.5">
                            Co-varying: {simResult.varyCols.join(', ')}
                          </p>
                        )}
                      </div>
                    )}
                    {sensResult && activeTab === 'sensitivity' && (
                      <h2 className="text-white font-semibold">
                        Sensitivity Analysis — <span className="text-purple-400">+10% perturbation</span> per X
                      </h2>
                    )}
                    <p className="text-gray-200 text-xs mt-0.5">
                      {activeTab === 'sensitivity'
                        ? 'Ranked by absolute impact on first selected Y. Other X held at baseline.'
                        : 'Other X variables held at baseline values. Dashed line = current baseline.'}
                    </p>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="flex bg-gray-700 rounded-lg p-1 gap-1">
                      {[
                        { key: 'chart',       label: '📈 Chart',       show: !!simResult },
                        { key: 'table',       label: '📋 Table',       show: !!simResult },
                        { key: 'sensitivity', label: '📊 Sensitivity', show: !!sensResult },
                      ].filter(t => t.show).map(t => (
                        <button
                          key={t.key}
                          onClick={() => setActiveTab(t.key)}
                          className={`px-3 py-1.5 rounded text-xs font-medium transition-colors ${
                            activeTab === t.key
                              ? 'bg-blue-600 text-white'
                              : 'text-gray-200 hover:text-white'
                          }`}
                        >
                          {t.label}
                        </button>
                      ))}
                    </div>
                    {simResult && (
                      <div className="flex items-center gap-2">
                        {/* Pop-out full view */}
                        <button
                          onClick={() => setShowTableModal(true)}
                          title="Open full-screen table view"
                          className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-gray-700
                                     hover:bg-indigo-700 border border-gray-600 hover:border-indigo-500
                                     text-gray-200 hover:text-white text-xs font-medium transition-colors"
                        >
                          <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round"
                              d="M4 8V6a2 2 0 012-2h2M4 16v2a2 2 0 002 2h2M16 4h2a2 2 0 012 2v2M16 20h2a2 2 0 002-2v-2" />
                          </svg>
                          Full View
                        </button>
                        {/* CSV download */}
                        <button
                          onClick={downloadWhatIfCSV}
                          title="Download What-If results as CSV"
                          className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-gray-700
                                     hover:bg-green-700 border border-gray-600 hover:border-green-500
                                     text-gray-200 hover:text-white text-xs font-medium transition-colors"
                        >
                          <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" d="M4 16v2a2 2 0 002 2h12a2 2 0 002-2v-2M7 10l5 5m0 0l5-5m-5 5V4" />
                          </svg>
                          CSV
                        </button>
                      </div>
                    )}
                  </div>
                </div>

                {/* ── CHART TAB ── */}
                {activeTab === 'chart' && simResult && (
                  <>
                    <ResponsiveContainer width="100%" height={380}>
                      <LineChart data={chartData} margin={{ top: 10, right: 20, left: 10, bottom: 30 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                        <XAxis
                          dataKey="x"
                          stroke="#6b7280"
                          tick={{ fontSize: 10 }}
                          label={{
                            value: simResult.varyX,
                            position: 'insideBottom',
                            offset: -15,
                            fill: '#60a5fa',
                            fontSize: 11,
                            fontWeight: 600,
                          }}
                        />
                        <YAxis stroke="#6b7280" tick={{ fontSize: 10 }} width={65} />
                        <Tooltip
                          contentStyle={{
                            backgroundColor: '#1f2937',
                            border: '1px solid #374151',
                            borderRadius: 8,
                            fontSize: 12,
                          }}
                          labelFormatter={v => `${simResult.varyX} = ${Number(v).toFixed(4)}`}
                          formatter={(v, name) => [Number(v).toFixed(4), name]}
                        />
                        <Legend wrapperStyle={{ fontSize: 12, paddingTop: 12 }} />
                        {baselineXVal != null && (
                          <ReferenceLine
                            x={Number(baselineXVal.toFixed(4))}
                            stroke="#fbbf24"
                            strokeDasharray="5 3"
                            label={{ value: 'baseline', fill: '#fbbf24', fontSize: 10, position: 'top' }}
                          />
                        )}
                        {observeY.map((col, idx) => (
                          <Line
                            key={col}
                            type="monotone"
                            dataKey={col}
                            stroke={COLORS[idx % COLORS.length]}
                            dot={false}
                            strokeWidth={2.5}
                            activeDot={{ r: 4 }}
                          />
                        ))}
                      </LineChart>
                    </ResponsiveContainer>

                    <div className="mt-4 grid grid-cols-2 gap-3">
                      {observeY.map((col, idx) => {
                        const vals = chartData.map(r => r[col]).filter(v => v != null)
                        if (vals.length === 0) return null
                        const mn = Math.min(...vals)
                        const mx = Math.max(...vals)
                        const range = mx - mn
                        return (
                          <div
                            key={col}
                            className="bg-gray-700/60 rounded-lg px-3 py-2 border-l-2"
                            style={{ borderColor: COLORS[idx % COLORS.length] }}
                          >
                            <p className="text-white text-xs font-semibold truncate" title={col}>{col}</p>
                            <div className="flex justify-between text-xs mt-1 text-gray-200">
                              <span>Min <span className="text-white">{fmt(mn, 3)}</span></span>
                              <span>Max <span className="text-white">{fmt(mx, 3)}</span></span>
                              <span>Δ <span className="text-yellow-300">{fmt(range, 3)}</span></span>
                            </div>
                          </div>
                        )
                      })}
                    </div>
                  </>
                )}

                {/* ── TABLE TAB ── */}
                {activeTab === 'table' && simResult && (() => {
                  const allXCols  = simResult.x_columns || []
                  const varyCols  = new Set(simResult.varyCols || [])
                  const primaryX  = simResult.varyX
                  return (
                    <div className="overflow-auto max-h-[500px] rounded-lg border border-gray-700">
                      <table className="w-full text-xs border-collapse">
                        <thead className="sticky top-0 bg-gray-700 text-gray-200 z-10">
                          <tr>
                            {/* ── X columns group ── */}
                            {allXCols.map(col => {
                              const isPrimary = col === primaryX
                              const isVarying = varyCols.has(col)
                              return (
                                <th
                                  key={col}
                                  className="px-3 py-2.5 text-left font-semibold border-r border-gray-600 whitespace-nowrap"
                                  style={{ color: isPrimary ? '#60a5fa' : isVarying ? '#2dd4bf' : '#9ca3af' }}
                                  title={isPrimary ? 'Primary sweep axis' : isVarying ? 'Co-varying with primary' : 'Fixed at baseline'}
                                >
                                  {col}
                                  {isPrimary && <span className="ml-1 text-blue-500 text-xs">▶</span>}
                                  {isVarying && <span className="ml-1 text-teal-500 text-xs">~</span>}
                                </th>
                              )
                            })}
                            {/* ── Y columns group ── */}
                            {observeY.map((col, idx) => (
                              <th
                                key={col}
                                className="px-3 py-2.5 text-right font-semibold whitespace-nowrap"
                                style={{ color: COLORS[idx % COLORS.length] }}
                              >
                                {col}
                              </th>
                            ))}
                          </tr>
                          {/* Sub-header labels */}
                          <tr className="bg-gray-750 border-t border-gray-600" style={{ backgroundColor: '#1f2937' }}>
                            <th
                              colSpan={allXCols.length}
                              className="px-3 py-1 text-left text-gray-500 font-normal text-xs border-r border-gray-600"
                            >
                              X Inputs ({allXCols.length})
                            </th>
                            <th
                              colSpan={observeY.length}
                              className="px-3 py-1 text-right text-gray-500 font-normal text-xs"
                            >
                              Y Outputs ({observeY.length})
                            </th>
                          </tr>
                        </thead>
                        <tbody>
                          {chartData.map((row, i) => {
                            const isBaseline = Math.abs(row.x - Number(baselineXVal?.toFixed(4))) < 1e-6
                            return (
                              <tr
                                key={i}
                                className={`border-t border-gray-700 ${
                                  isBaseline ? 'bg-yellow-900/20' : i % 2 === 0 ? 'bg-gray-800' : 'bg-gray-800/50'
                                }`}
                              >
                                {/* X columns */}
                                {allXCols.map(col => {
                                  const isPrimary = col === primaryX
                                  const isVarying = varyCols.has(col)
                                  const val = row[`__x__${col}`]
                                  const dir = (isPrimary || isVarying)
                                    ? dirInfo(val, baselineXVals[col])
                                    : null
                                  return (
                                    <td
                                      key={col}
                                      className={`px-3 py-1.5 font-mono border-r border-gray-700/50 whitespace-nowrap ${
                                        isPrimary ? 'text-blue-300 font-medium' :
                                        isVarying ? 'text-teal-300' :
                                        'text-gray-500'
                                      }`}
                                    >
                                      <span>{val != null ? val : '—'}</span>
                                      {dir && dir.sym && (
                                        <span className={`ml-1.5 font-bold font-sans ${dir.cls}`} title={dir.label}>
                                          {dir.sym}
                                        </span>
                                      )}
                                      {isPrimary && isBaseline && (
                                        <span className="ml-1 text-yellow-400 text-xs font-sans">←</span>
                                      )}
                                    </td>
                                  )
                                })}
                                {/* Y columns */}
                                {observeY.map(col => {
                                  const val = row[col]
                                  const dir = dirInfo(val, baselineYVals[col])
                                  return (
                                    <td key={col} className="px-3 py-1.5 text-right font-mono whitespace-nowrap">
                                      <span className="text-gray-100">{fmt(val)}</span>
                                      {dir.sym && (
                                        <span className={`ml-1.5 font-bold font-sans ${dir.cls}`} title={dir.label}>
                                          {dir.sym}
                                        </span>
                                      )}
                                    </td>
                                  )
                                })}
                              </tr>
                            )
                          })}
                        </tbody>
                      </table>
                    </div>
                  )
                })()}

                {/* ── SENSITIVITY TAB ── */}
                {activeTab === 'sensitivity' && sensResult && (
                  <div className="overflow-auto max-h-[520px] rounded-lg border border-gray-700">
                    <table className="w-full text-xs border-collapse">
                      <thead className="sticky top-0 bg-gray-700">
                        <tr>
                          <th className="px-3 py-2.5 text-left text-gray-100">X Input</th>
                          <th className="px-3 py-2.5 text-left text-gray-200">Baseline</th>
                          <th className="px-3 py-2.5 text-left text-gray-200">+10%</th>
                          {observeY.map((col, idx) => (
                            <th
                              key={col}
                              className="px-3 py-2.5 text-left min-w-36"
                              style={{ color: COLORS[idx % COLORS.length] }}
                            >
                              {col}
                            </th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {sensResult.map((row, i) => (
                          <tr key={row.x_col} className={`border-t border-gray-700 ${i % 2 === 0 ? 'bg-gray-800' : ''}`}>
                            <td className="px-3 py-2 text-white font-medium">{row.x_col}</td>
                            <td className="px-3 py-2 text-gray-200">{fmt(row.base_val, 3)}</td>
                            <td className="px-3 py-2 text-gray-200">{fmt(row.perturbed_val, 3)}</td>
                            {observeY.map(yc => {
                              const d = row.deltas[yc]
                              return (
                                <td key={yc} className="px-3 py-2 min-w-36">
                                  {d
                                    ? pctBar(d.delta, sensMaxDelta[yc])
                                    : <span className="text-gray-600">—</span>
                                  }
                                  {d?.pct != null && (
                                    <span className={`text-xs ml-1 ${d.pct >= 0 ? 'text-blue-400' : 'text-orange-400'}`}>
                                      ({d.pct >= 0 ? '+' : ''}{d.pct.toFixed(1)}%)
                                    </span>
                                  )}
                                </td>
                              )
                            })}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                    <p className="text-gray-600 text-xs px-3 py-2 border-t border-gray-700">
                      Bar length = relative impact. Value = absolute change in Y when that X increases by 10%.
                      Sorted by impact on <span className="text-white">{observeY[0]}</span>.
                    </p>
                  </div>
                )}

              </div>
            )}
          </>)}
        </div>
      </div>

      {/* ══════════════════════════════════════════════════════════════
          ROW PICKER MODAL — Fill baseline from dataset
      ══════════════════════════════════════════════════════════════ */}
      {showRowPicker && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4"
          style={{ backgroundColor: 'rgba(0,0,0,0.7)' }}>
          <div className="border border-gray-700 rounded-2xl w-full max-w-2xl shadow-2xl flex flex-col"
            style={{ maxHeight: '80vh', backgroundColor: '#111827' }}>

            {/* Header */}
            <div className="flex items-center justify-between px-5 py-4 border-b border-gray-700">
              <div>
                <h2 className="text-white font-semibold text-base">📂 Fill Baseline from Dataset</h2>
                <p className="text-blue-300 text-xs mt-0.5 font-medium">
                  {datasets.find(d => d.id === baselineDatasetId)?.original_name ?? ''}
                  {baselineDatasetId === selectedRun?.dataset_id && (
                    <span className="text-green-400 ml-1">(training dataset)</span>
                  )}
                </p>
                {datasetRows && (
                  <p className="text-gray-400 text-xs mt-0.5">
                    {datasetRows.sample_size} sampled rows of {datasetRows.total_rows.toLocaleString()} total
                    {datasetRows.has_timestamp && (
                      <span className="ml-2 text-emerald-400">· {datasetRows.timestamp_col}</span>
                    )}
                  </p>
                )}
              </div>
              <button
                onClick={() => setShowRowPicker(false)}
                className="text-gray-400 hover:text-white text-xl leading-none transition-colors ml-4"
              >✕</button>
            </div>

            {/* Search */}
            <div className="px-5 py-3 border-b border-gray-700">
              <input
                type="text"
                placeholder={
                  datasetRows?.has_timestamp
                    ? 'Search by timestamp or row number…'
                    : 'Search by row number…'
                }
                value={rowSearch}
                onChange={e => setRowSearch(e.target.value)}
                autoFocus
                className="w-full bg-gray-700 border border-gray-600 rounded-lg px-3 py-2
                           text-white text-xs placeholder-gray-500 focus:border-blue-500 focus:outline-none"
              />
            </div>

            {/* Row list */}
            <div className="flex-1 overflow-y-auto px-5 py-3 space-y-2">
              {rowPickerLoading && (
                <div className="flex items-center justify-center py-12">
                  <span className="text-gray-400 text-sm">Loading dataset rows…</span>
                </div>
              )}

              {!rowPickerLoading && !datasetRows && (
                <div className="py-12 text-center text-red-400 text-sm">
                  Failed to load dataset rows.
                </div>
              )}

              {!rowPickerLoading && datasetRows && filteredRows.length === 0 && (
                <div className="py-12 text-center text-gray-500 text-sm">
                  No rows match your search.
                </div>
              )}

              {!rowPickerLoading && filteredRows.map(row => {
                const previewCols = (datasetRows.x_cols || []).slice(0, 4)
                const extraCount  = (datasetRows.x_cols || []).length - 4
                return (
                  <button
                    key={row.row_index}
                    onClick={() => handleSelectRow(row)}
                    className="w-full text-left bg-gray-800 hover:bg-gray-700 border border-gray-700
                               hover:border-blue-500 rounded-xl px-4 py-3 transition-colors group"
                  >
                    <div className="flex items-center justify-between mb-1.5">
                      <div className="flex items-center gap-3">
                        <span className="text-gray-500 text-xs font-mono">
                          Row {(row.row_index + 1).toLocaleString()}
                        </span>
                        {row.timestamp && (
                          <span className="text-emerald-400 text-xs font-medium">{row.timestamp}</span>
                        )}
                      </div>
                      <span className="text-blue-400 text-xs opacity-0 group-hover:opacity-100 transition-opacity">
                        Use this row →
                      </span>
                    </div>
                    <div className="flex flex-wrap gap-x-4 gap-y-0.5">
                      {previewCols.map(col => (
                        <span key={col} className="text-xs">
                          <span className="text-gray-500">{col}:</span>{' '}
                          <span className="text-gray-200 font-mono">
                            {row.x_values[col] != null ? Number(row.x_values[col]).toFixed(3) : '—'}
                          </span>
                        </span>
                      ))}
                      {extraCount > 0 && (
                        <span className="text-gray-600 text-xs">+{extraCount} more</span>
                      )}
                    </div>
                  </button>
                )
              })}
            </div>

            {/* Footer */}
            <div className="px-5 py-3 border-t border-gray-700 flex justify-between items-center">
              <p className="text-gray-600 text-xs">
                Click a row to fill all X baseline values. Y outputs are not affected.
              </p>
              <button
                onClick={() => setShowRowPicker(false)}
                className="px-4 py-1.5 bg-gray-700 hover:bg-gray-600 text-gray-300 rounded-lg text-xs transition-colors"
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}

      {/* ══════════════════════════════════════════════════════════════
          FULL-SCREEN TABLE MODAL
      ══════════════════════════════════════════════════════════════ */}
      {showTableModal && simResult && (() => {
        const allXCols = simResult.x_columns || []
        const varyCols = new Set(simResult.varyCols || [])
        const primaryX = simResult.varyX
        return (
          <div
            className="fixed inset-0 z-50 flex flex-col bg-gray-950"
            onKeyDown={e => e.key === 'Escape' && setShowTableModal(false)}
            tabIndex={-1}
          >
            {/* ── Modal Header ── */}
            <div className="flex items-center justify-between px-6 py-4 bg-gray-900 border-b border-gray-700 flex-shrink-0">
              <div className="flex items-center gap-4 min-w-0">
                <h2 className="text-white font-bold text-lg flex items-center gap-2 flex-wrap">
                  Effect of <span className="text-blue-400">{primaryX}</span> on Y Outputs
                  {simResult.coupled && (
                    <span className="text-xs bg-teal-900 text-teal-300 border border-teal-700 px-2 py-0.5 rounded-full font-normal">
                      ⟳ {simResult.varyCols?.length} X co-varying
                    </span>
                  )}
                </h2>
                <div className="flex items-center gap-2 text-xs text-gray-400 flex-shrink-0">
                  <span className="text-blue-400">▶ Primary</span>
                  <span className="text-teal-400">~ Co-vary</span>
                  <span className="text-gray-500">— Fixed</span>
                  <span className="text-green-400">↑ Increase</span>
                  <span className="text-red-400">↓ Decrease</span>
                  <span className="text-gray-500">→ No change</span>
                </div>
              </div>
              <div className="flex items-center gap-3 ml-4 flex-shrink-0">
                <button
                  onClick={downloadWhatIfCSV}
                  className="flex items-center gap-1.5 px-4 py-2 rounded-lg bg-gray-700
                             hover:bg-green-700 border border-gray-600 hover:border-green-500
                             text-gray-200 hover:text-white text-sm font-medium transition-colors"
                >
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" d="M4 16v2a2 2 0 002 2h12a2 2 0 002-2v-2M7 10l5 5m0 0l5-5m-5 5V4" />
                  </svg>
                  Download CSV
                </button>
                <button
                  onClick={() => setShowTableModal(false)}
                  className="flex items-center gap-1.5 px-4 py-2 rounded-lg bg-gray-700 hover:bg-gray-600
                             border border-gray-600 text-gray-200 hover:text-white text-sm font-medium transition-colors"
                >
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
                  </svg>
                  Close
                </button>
              </div>
            </div>

            {/* ── Stats bar ── */}
            <div className="flex items-center gap-6 px-6 py-2 bg-gray-900/60 border-b border-gray-800 text-xs text-gray-400 flex-shrink-0">
              <span><span className="text-white font-medium">{chartData.length}</span> rows</span>
              <span><span className="text-blue-300 font-medium">{allXCols.length}</span> X columns ({allXCols.filter(c => c === primaryX || varyCols.has(c)).length} active)</span>
              <span><span className="text-green-300 font-medium">{observeY.length}</span> Y outputs</span>
              <span>Baseline: <span className="text-yellow-300 font-medium">{primaryX} = {fmt(baselineXVal, 4)}</span></span>
              <span className="ml-auto text-gray-600">Press Esc to close</span>
            </div>

            {/* ── Full Table ── */}
            <div className="flex-1 overflow-auto">
              <table className="text-xs border-collapse min-w-max w-full">
                <thead className="sticky top-0 z-10">
                  <tr className="bg-gray-800">
                    <th className="px-4 py-3 text-left text-gray-500 font-medium border-r border-gray-700 sticky left-0 bg-gray-800 z-20">
                      #
                    </th>
                    {/* X column headers */}
                    {allXCols.map(col => {
                      const isPrimary = col === primaryX
                      const isVarying = varyCols.has(col)
                      return (
                        <th
                          key={col}
                          className="px-4 py-3 text-left font-semibold border-r border-gray-700 whitespace-nowrap"
                          style={{ color: isPrimary ? '#60a5fa' : isVarying ? '#2dd4bf' : '#6b7280' }}
                          title={isPrimary ? 'Primary sweep axis' : isVarying ? 'Co-varies with primary' : 'Fixed at baseline'}
                        >
                          {col}
                          {isPrimary && <span className="ml-1 opacity-70">▶</span>}
                          {isVarying && <span className="ml-1 opacity-70">~</span>}
                        </th>
                      )
                    })}
                    {/* Y column headers */}
                    {observeY.map((col, idx) => (
                      <th
                        key={col}
                        className="px-4 py-3 text-right font-semibold whitespace-nowrap"
                        style={{ color: COLORS[idx % COLORS.length] }}
                      >
                        {col}
                      </th>
                    ))}
                  </tr>
                  {/* Group sub-headers */}
                  <tr style={{ backgroundColor: '#111827' }} className="border-b border-gray-700">
                    <th className="px-4 py-1.5 sticky left-0 border-r border-gray-700" style={{ backgroundColor: '#111827' }} />
                    <th
                      colSpan={allXCols.length}
                      className="px-4 py-1.5 text-left text-gray-500 font-normal border-r border-gray-700"
                    >
                      ← X Inputs ({allXCols.length} columns · {allXCols.filter(c => c !== primaryX && !varyCols.has(c)).length} fixed at baseline)
                    </th>
                    <th
                      colSpan={observeY.length}
                      className="px-4 py-1.5 text-right text-gray-500 font-normal"
                    >
                      Y Outputs ({observeY.length} columns) →
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {chartData.map((row, i) => {
                    const isBaseline = Math.abs(row.x - Number(baselineXVal?.toFixed(4))) < 1e-6
                    return (
                      <tr
                        key={i}
                        className={`border-b border-gray-800/60 ${
                          isBaseline
                            ? 'bg-yellow-900/25 border-yellow-800/40'
                            : i % 2 === 0
                            ? 'bg-gray-900'
                            : 'bg-gray-900/50'
                        }`}
                      >
                        {/* Row number */}
                        <td className={`px-4 py-2 text-gray-600 text-right border-r border-gray-800 sticky left-0 font-mono
                                        ${isBaseline ? 'bg-yellow-900/25' : i % 2 === 0 ? 'bg-gray-900' : 'bg-gray-900/50'}`}>
                          {i + 1}
                          {isBaseline && <span className="ml-1 text-yellow-500 text-xs">★</span>}
                        </td>

                        {/* X cells */}
                        {allXCols.map(col => {
                          const isPrimary = col === primaryX
                          const isVarying = varyCols.has(col)
                          const val = row[`__x__${col}`]
                          const dir = (isPrimary || isVarying) ? dirInfo(val, baselineXVals[col]) : null
                          return (
                            <td
                              key={col}
                              className={`px-4 py-2 font-mono border-r border-gray-800/60 whitespace-nowrap ${
                                isPrimary ? 'text-blue-300' :
                                isVarying ? 'text-teal-300' :
                                'text-gray-500'
                              }`}
                            >
                              {val != null ? val : '—'}
                              {dir && dir.sym && (
                                <span className={`ml-1.5 font-bold font-sans text-xs ${dir.cls}`}>{dir.sym}</span>
                              )}
                            </td>
                          )
                        })}

                        {/* Y cells */}
                        {observeY.map(col => {
                          const val = row[col]
                          const dir = dirInfo(val, baselineYVals[col])
                          return (
                            <td key={col} className="px-4 py-2 text-right font-mono whitespace-nowrap">
                              <span className="text-gray-100">{fmt(val)}</span>
                              {dir.sym && (
                                <span className={`ml-1.5 font-bold font-sans text-xs ${dir.cls}`}>{dir.sym}</span>
                              )}
                            </td>
                          )
                        })}
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          </div>
        )
      })()}
    </div>
  )
}
