import { useState, useEffect } from 'react'
import { listRuns, predictTestData, deleteRun } from '../services/api'
import {
  ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceLine, Legend,
} from 'recharts'

const COLORS = ['#60a5fa', '#34d399', '#f87171', '#fbbf24', '#a78bfa', '#fb923c', '#38bdf8']

const r2Color = (r2) => {
  if (r2 == null) return 'text-gray-500'
  if (r2 >= 0.9) return 'text-green-400'
  if (r2 >= 0.7) return 'text-yellow-400'
  return 'text-red-400'
}

const r2Label = (r2) => {
  if (r2 == null) return 'N/A'
  if (r2 >= 0.9) return 'Excellent'
  if (r2 >= 0.7) return 'Good'
  if (r2 >= 0.5) return 'Fair'
  return 'Poor'
}

const fmt = (v, digits = 4) => (v == null ? 'N/A' : v.toFixed(digits))

// Custom tooltip for scatter chart
const ScatterTooltip = ({ active, payload }) => {
  if (!active || !payload?.length) return null
  const d = payload[0].payload
  return (
    <div className="bg-gray-900 border border-gray-600 rounded-lg px-3 py-2 text-xs">
      <p className="text-gray-100">Actual: <span className="text-white font-mono">{fmt(d.actual)}</span></p>
      <p className="text-gray-100">Predicted: <span className="text-white font-mono">{fmt(d.predicted)}</span></p>
      <p className="text-gray-200">Error: {d.predicted != null && d.actual != null ? fmt(d.predicted - d.actual) : 'N/A'}</p>
    </div>
  )
}

export default function Predict() {
  const [runs, setRuns]               = useState([])
  const [selectedRun, setSelectedRun] = useState(null)
  const [result, setResult]           = useState(null)
  const [loading, setLoading]         = useState(false)
  const [error, setError]             = useState('')
  const [activeCol, setActiveCol]     = useState(null)

  // Delete state
  const [confirmDeleteId, setConfirmDeleteId] = useState(null)
  const [deleting, setDeleting]               = useState(false)
  const [deleteError, setDeleteError]         = useState('')

  useEffect(() => {
    listRuns()
      .then(r => setRuns(r.data.filter(r => ['done', 'stopped'].includes(r.status))))
      .catch(() => {})
  }, [])

  const handleDeleteRun = async (id) => {
    setDeleting(true)
    setDeleteError('')
    try {
      await deleteRun(id)
      setRuns(prev => prev.filter(r => r.id !== id))
      if (selectedRun?.id === id) {
        setSelectedRun(null)
        setResult(null)
        setActiveCol(null)
      }
      setConfirmDeleteId(null)
    } catch (e) {
      setDeleteError(e.response?.data?.detail || e.message)
      setConfirmDeleteId(null)
    }
    setDeleting(false)
  }

  const handleSelectRun = async (run) => {
    setSelectedRun(run)
    setResult(null)
    setError('')
    setActiveCol(null)
    setLoading(true)
    try {
      const r = await predictTestData(run.id)
      setResult(r.data)
      // Auto-open first Y column
      const firstCol = Object.keys(r.data.columns)[0]
      setActiveCol(firstCol)
    } catch (e) {
      setError('Failed to predict: ' + (e.response?.data?.detail || e.message))
    }
    setLoading(false)
  }

  return (
    <div className="p-8 max-w-5xl mx-auto">
      <h1 className="text-2xl font-bold text-white mb-1">Predict — Test Data Evaluation</h1>
      <p className="text-gray-200 text-sm mb-6">
        Select a trained model to automatically evaluate it on the held-out test rows
        (20% of Data_DAE.xlsx). Compares <span className="text-green-400">predicted</span> vs{' '}
        <span className="text-blue-400">actual</span> Y values with R² and MAE per output variable.
      </p>

      {/* Model selector */}
      <div className="bg-gray-800 rounded-xl p-6 mb-6">
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-white font-semibold">Select Trained Model</h2>
          <span className="text-gray-500 text-xs">Hover a card to reveal the delete button</span>
        </div>

        {deleteError && (
          <div className="bg-red-900/40 border border-red-700 rounded-lg px-3 py-2 mb-3">
            <p className="text-red-400 text-xs">{deleteError}</p>
          </div>
        )}

        {runs.length === 0 ? (
          <p className="text-gray-500 text-sm">No trained models yet. Go to Train Model first.</p>
        ) : (
          <div className="flex flex-wrap gap-2">
            {runs.map((r) => {
              const isSelected  = selectedRun?.id === r.id
              const isConfirm   = confirmDeleteId === r.id
              const r2          = r.metrics?.r2_score
              const r2Clr       = r2 == null ? 'text-gray-500'
                                : r2 >= 0.9  ? 'text-green-400'
                                : r2 >= 0.7  ? 'text-yellow-400'
                                : 'text-red-400'

              if (isConfirm) {
                // ── Inline delete confirmation ──────────────────────────────
                return (
                  <div
                    key={r.id}
                    className="flex flex-col gap-2 px-3 py-2.5 rounded-lg border border-red-600
                               bg-red-900/30 text-xs min-w-36"
                  >
                    <span className="text-red-200 font-semibold">Delete Run #{r.id}?</span>
                    <span className="text-gray-200 leading-snug">
                      This removes the model files and all predictions.
                    </span>
                    <div className="flex gap-2 mt-0.5">
                      <button
                        onClick={() => handleDeleteRun(r.id)}
                        disabled={deleting}
                        className="bg-red-600 hover:bg-red-500 disabled:opacity-50
                                   text-white px-3 py-1 rounded text-xs font-bold transition-colors"
                      >
                        {deleting ? '…' : 'Yes, Delete'}
                      </button>
                      <button
                        onClick={() => { setConfirmDeleteId(null); setDeleteError('') }}
                        disabled={deleting}
                        className="text-gray-200 hover:text-white text-xs transition-colors"
                      >
                        Cancel
                      </button>
                    </div>
                  </div>
                )
              }

              // ── Normal run card ─────────────────────────────────────────
              return (
                <div key={r.id} className="relative group">
                  {/* Main select button */}
                  <button
                    onClick={() => handleSelectRun(r)}
                    disabled={loading}
                    className={`text-left px-4 py-2.5 pr-8 rounded-lg text-sm border
                                transition-colors disabled:opacity-50 w-full ${
                      isSelected
                        ? 'bg-blue-600 border-blue-500 text-white'
                        : 'bg-gray-700 border-gray-600 text-gray-100 hover:border-blue-500'
                    }`}
                  >
                    <div className="font-medium flex items-center gap-1.5">
                      Run #{r.id}
                      {r.status === 'stopped' && (
                        <span className="text-orange-300 text-xs">⏹ partial</span>
                      )}
                    </div>
                    <div className="text-xs opacity-70 mt-0.5 flex items-center gap-1">
                      <span className={r2Clr + ' font-semibold'}>
                        R²: {r2 != null ? r2.toFixed(3) : 'N/A'}
                      </span>
                      <span className="text-gray-200">·</span>
                      <span>{r.epochs} ep</span>
                      <span className="text-gray-200">·</span>
                      <span className={isSelected ? 'text-blue-200' : 'text-blue-300'}>
                        {r.x_columns?.length}X
                      </span>
                      <span>→</span>
                      <span className={isSelected ? 'text-green-200' : 'text-green-300'}>
                        {r.y_columns?.length}Y
                      </span>
                    </div>
                  </button>

                  {/* Delete button — appears on hover */}
                  <button
                    onClick={(e) => {
                      e.stopPropagation()
                      setConfirmDeleteId(r.id)
                      setDeleteError('')
                    }}
                    disabled={loading || deleting}
                    title="Delete this run"
                    className={`absolute top-1.5 right-1.5 w-5 h-5 rounded flex items-center
                                justify-center opacity-0 group-hover:opacity-100 transition-all
                                disabled:opacity-0
                                ${isSelected
                                  ? 'bg-blue-800 hover:bg-red-700 text-blue-200 hover:text-white'
                                  : 'bg-gray-600 hover:bg-red-700 text-gray-200 hover:text-white'
                                }`}
                  >
                    <svg className="w-3 h-3" fill="none" stroke="currentColor" strokeWidth={2.5} viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round"
                        d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                    </svg>
                  </button>
                </div>
              )
            })}
          </div>
        )}
      </div>

      {/* Loading */}
      {loading && (
        <div className="bg-gray-800 rounded-xl p-8 text-center">
          <div className="text-blue-400 text-lg mb-2 animate-pulse">Running predictions on test data...</div>
          <p className="text-gray-500 text-sm">Evaluating all test rows. This may take a moment.</p>
        </div>
      )}

      {error && (
        <div className="bg-red-900/40 border border-red-700 rounded-lg px-4 py-3 mb-4">
          <p className="text-red-400 text-sm">{error}</p>
        </div>
      )}

      {/* Results */}
      {result && !loading && (
        <>
          {/* Summary header */}
          <div className="bg-gray-800 rounded-xl p-5 mb-4">
            <div className="flex items-start justify-between mb-3">
              <div>
                <p className="text-white font-semibold">
                  Test set: <span className="text-blue-400">{result.n_test_rows.toLocaleString()} rows</span>
                  <span className="text-gray-500 text-sm ml-2">
                    ({result.sample_size} sampled points per chart)
                  </span>
                </p>
                <p className="text-gray-200 text-xs mt-1">
                  Run #{selectedRun.id} ·{' '}
                  <span className="text-blue-300">{(result.x_cols_used ?? selectedRun.x_columns)?.length} X inputs</span>
                  {' → '}
                  <span className="text-green-300">{(result.y_cols_used ?? selectedRun.y_columns)?.length} Y outputs</span>
                  {' · '}test split: {Math.round(selectedRun.test_size * 100)}%
                </p>
              </div>

              {/* Mismatch warning — shown if config.json y_cols differ from DB */}
              {result.y_cols_used &&
               JSON.stringify(result.y_cols_used) !== JSON.stringify(selectedRun.y_columns) && (
                <div className="bg-yellow-900/40 border border-yellow-700 rounded-lg px-3 py-2 text-xs text-yellow-300 max-w-xs">
                  ⚠ Column list was auto-corrected from the saved model file.
                  DB record has been updated.
                </div>
              )}
            </div>

            {/* Y column pills — shows exactly what was trained */}
            <div>
              <p className="text-gray-500 text-xs mb-1.5">Y outputs trained in this model:</p>
              <div className="flex flex-wrap gap-1.5">
                {(result.y_cols_used ?? selectedRun.y_columns ?? []).map((col, i) => (
                  <span
                    key={col}
                    className="text-xs px-2 py-0.5 rounded-full border font-mono"
                    style={{
                      color:           COLORS[i % COLORS.length],
                      borderColor:     COLORS[i % COLORS.length] + '60',
                      backgroundColor: COLORS[i % COLORS.length] + '15',
                    }}
                  >
                    {col}
                  </span>
                ))}
              </div>

              {/* Warn if a DB-listed Y col is absent from the model's config */}
              {result.y_cols_used && selectedRun.y_columns && (() => {
                const missing = selectedRun.y_columns.filter(
                  c => !result.y_cols_used.includes(c)
                )
                return missing.length > 0 ? (
                  <p className="text-red-400 text-xs mt-2">
                    ⚠ The following Y columns listed in the training record were
                    not found in the saved model and have been excluded:{' '}
                    <span className="font-mono font-bold">{missing.join(', ')}</span>
                  </p>
                ) : null
              })()}
            </div>
          </div>

          {/* Per-column metric cards */}
          <div className="grid grid-cols-2 gap-3 mb-6">
            {Object.entries(result.columns).map(([col, data], i) => (
              <button
                key={col}
                onClick={() => setActiveCol(activeCol === col ? null : col)}
                className={`text-left rounded-xl p-4 border transition-all ${
                  activeCol === col
                    ? 'bg-gray-700 border-blue-500'
                    : 'bg-gray-800 border-gray-700 hover:border-gray-500'
                }`}
              >
                <div className="flex items-center justify-between mb-2">
                  <p className="text-gray-100 text-xs font-medium truncate pr-2" title={col}>{col}</p>
                  <span
                    className="text-xs px-2 py-0.5 rounded-full border flex-shrink-0"
                    style={{
                      color: COLORS[i % COLORS.length],
                      borderColor: COLORS[i % COLORS.length] + '60',
                      backgroundColor: COLORS[i % COLORS.length] + '15',
                    }}
                  >
                    {r2Label(data.r2)}
                  </span>
                </div>
                <div className="flex gap-6">
                  <div>
                    <p className={`text-xl font-bold ${r2Color(data.r2)}`}>{fmt(data.r2)}</p>
                    <p className="text-gray-500 text-xs">R² Score</p>
                  </div>
                  <div>
                    <p className="text-xl font-bold text-white">{fmt(data.mae)}</p>
                    <p className="text-gray-500 text-xs">MAE</p>
                  </div>
                </div>
                <p className="text-gray-600 text-xs mt-2">
                  {activeCol === col ? '▲ Click to collapse chart' : '▼ Click to expand chart'}
                </p>
              </button>
            ))}
          </div>

          {/* Expanded scatter chart for selected column */}
          {activeCol && result.columns[activeCol] && (() => {
            const colIdx = Object.keys(result.columns).indexOf(activeCol)
            const colData = result.columns[activeCol]
            const color = COLORS[colIdx % COLORS.length]

            // Build scatter data: [{actual, predicted}]
            const scatterData = colData.actual.map((a, i) => ({
              actual: a,
              predicted: colData.predicted[i],
            }))

            // Perfect-prediction reference line range
            const allVals = [...colData.actual, ...colData.predicted]
            const minVal  = Math.min(...allVals)
            const maxVal  = Math.max(...allVals)

            return (
              <div className="bg-gray-800 rounded-xl p-6 mb-4">
                <div className="flex items-center justify-between mb-1">
                  <h3 className="text-white font-semibold">{activeCol}</h3>
                  <div className="flex gap-4 text-sm">
                    <span className={r2Color(colData.r2)}>R² = {fmt(colData.r2)}</span>
                    <span className="text-gray-200">MAE = {fmt(colData.mae)}</span>
                  </div>
                </div>
                <p className="text-gray-500 text-xs mb-4">
                  Scatter plot: each point = one test row. Points on the diagonal line = perfect prediction.
                </p>

                <ResponsiveContainer width="100%" height={350}>
                  <ScatterChart margin={{ top: 10, right: 20, bottom: 30, left: 20 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                    <XAxis
                      dataKey="actual"
                      name="Actual"
                      type="number"
                      domain={[minVal, maxVal]}
                      stroke="#6b7280"
                      tick={{ fontSize: 11 }}
                      label={{ value: 'Actual', position: 'bottom', fill: '#9ca3af', fontSize: 12 }}
                    />
                    <YAxis
                      dataKey="predicted"
                      name="Predicted"
                      type="number"
                      domain={[minVal, maxVal]}
                      stroke="#6b7280"
                      tick={{ fontSize: 11 }}
                      label={{ value: 'Predicted', angle: -90, position: 'insideLeft', fill: '#9ca3af', fontSize: 12 }}
                    />
                    <Tooltip content={<ScatterTooltip />} />

                    {/* Perfect prediction diagonal */}
                    <ReferenceLine
                      segment={[
                        { x: minVal, y: minVal },
                        { x: maxVal, y: maxVal },
                      ]}
                      stroke="#4b5563"
                      strokeDasharray="6 3"
                      label={{ value: 'Perfect', fill: '#6b7280', fontSize: 10 }}
                    />

                    <Scatter
                      name={activeCol}
                      data={scatterData}
                      fill={color}
                      opacity={0.6}
                      r={3}
                    />
                  </ScatterChart>
                </ResponsiveContainer>

                {/* Mini table: first 10 rows */}
                <div className="mt-4">
                  <p className="text-gray-200 text-xs font-medium mb-2">Sample predictions (first 10 rows)</p>
                  <div className="overflow-x-auto">
                    <table className="w-full text-xs text-gray-100">
                      <thead>
                        <tr className="border-b border-gray-700">
                          <th className="text-left py-1.5 pr-4 text-gray-500">#</th>
                          <th className="text-left py-1.5 pr-4 text-blue-400">Actual</th>
                          <th className="text-left py-1.5 pr-4 text-green-400">Predicted</th>
                          <th className="text-left py-1.5 text-gray-500">Error</th>
                        </tr>
                      </thead>
                      <tbody>
                        {scatterData.slice(0, 10).map((row, i) => {
                          const err = row.predicted - row.actual
                          return (
                            <tr key={i} className="border-b border-gray-700/50">
                              <td className="py-1 pr-4 text-gray-600">{i + 1}</td>
                              <td className="py-1 pr-4 font-mono">{fmt(row.actual)}</td>
                              <td className="py-1 pr-4 font-mono">{fmt(row.predicted)}</td>
                              <td className={`py-1 font-mono ${colData.mae != null && Math.abs(err) < colData.mae ? 'text-green-500' : 'text-orange-400'}`}>
                                {err != null ? (err > 0 ? '+' : '') + fmt(err) : 'N/A'}
                              </td>
                            </tr>
                          )
                        })}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            )
          })()}
        </>
      )}
    </div>
  )
}
