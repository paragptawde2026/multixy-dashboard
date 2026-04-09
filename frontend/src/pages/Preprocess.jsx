import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { listDatasets, getDatasetStats, preprocessDataset, deleteDataset, previewDataset } from '../services/api'

// ── Small helpers ──────────────────────────────────────────────────────────────

const pct = (n, total) => total > 0 ? ((n / total) * 100).toFixed(1) : '0.0'

const nullBg = (nullPct) => {
  if (nullPct === 0) return ''
  if (nullPct < 5)  return 'text-yellow-400'
  if (nullPct < 20) return 'text-orange-400'
  return 'text-red-400'
}

const outlierBg = (count, total) => {
  const p = total > 0 ? (count / total) * 100 : 0
  if (p === 0)  return ''
  if (p < 2)   return 'text-yellow-400'
  if (p < 10)  return 'text-orange-400'
  return 'text-red-400'
}

// ── Main component ─────────────────────────────────────────────────────────────

export default function Preprocess() {
  const navigate = useNavigate()

  const [datasets,         setDatasets]         = useState([])
  const [selectedDataset,  setSelectedDataset]  = useState(null)
  const [stats,            setStats]            = useState(null)
  const [loadingStats,     setLoadingStats]     = useState(false)

  // Step 0 — duplicate column removal
  const [colsToDrop,       setColsToDrop]       = useState(new Set())

  // Step 1/2 — preprocessing options
  const [removeNan,        setRemoveNan]        = useState(true)
  const [outlierMethod,    setOutlierMethod]    = useState('iqr')
  const [threshold,        setThreshold]        = useState(1.5)
  const [selectedCols,     setSelectedCols]     = useState(new Set())

  // Output dataset name
  const [datasetName,  setDatasetName]  = useState('')   // '' = auto-generated on backend

  // Result
  const [applying,  setApplying]  = useState(false)
  const [result,    setResult]    = useState(null)
  const [error,     setError]     = useState('')

  // Delete
  const [confirmDeleteId, setConfirmDeleteId] = useState(null)  // dataset id pending confirmation
  const [deleting,        setDeleting]        = useState(false)
  const [forceDeleteId,   setForceDeleteId]   = useState(null)  // dataset id needing force-delete
  const [forceDeleteMsg,  setForceDeleteMsg]  = useState('')    // reason from backend

  // UI
  const [showAllCols, setShowAllCols] = useState(false)   // expand stats table

  // Duplicate-column preview modal
  const [previewGroup,   setPreviewGroup]   = useState(null)   // { columns, gi } | null
  const [previewRows,    setPreviewRows]    = useState([])
  const [loadingPreview, setLoadingPreview] = useState(false)

  // ── Load datasets ────────────────────────────────────────────────────────────
  useEffect(() => {
    listDatasets().then(r => setDatasets(r.data)).catch(() => {})
  }, [])

  // ── Auto-generate a versioned name (mirrors backend _next_dataset_name) ─────
  const autoDatasetName = (sourceDs, allDatasets) => {
    if (!sourceDs) return ''
    let name = sourceDs.original_name
    while (name.startsWith('cleaned_')) name = name.slice('cleaned_'.length)
    let stem = name.includes('.') ? name.slice(0, name.lastIndexOf('.')) : name
    stem = stem.replace(/_v\d+$/, '')
    const count = allDatasets.filter(d =>
      d.original_name.startsWith(stem + '_v') &&
      /^v\d+$/.test(d.original_name.slice(stem.length + 1))
    ).length
    return `${stem}_v${count + 1}`
  }

  // ── Select dataset & fetch stats ─────────────────────────────────────────────
  const handleSelectDataset = async (ds) => {
    setSelectedDataset(ds)
    setResult(null)
    setError('')
    setStats(null)
    setShowAllCols(false)
    setLoadingStats(true)
    setSelectedCols(new Set(ds.columns))   // all columns selected by default
    setColsToDrop(new Set())               // reset column-drop selection
    setDatasetName('')                     // reset to auto-generated
    try {
      const r = await getDatasetStats(ds.id)
      setStats(r.data)
      // Auto-check all "suggested_drop" columns from duplicate groups
      const suggested = new Set(
        (r.data.duplicate_column_groups ?? []).flatMap(g => g.suggested_drop)
      )
      setColsToDrop(suggested)
    } catch (e) {
      setError('Failed to load stats: ' + (e.response?.data?.detail || e.message))
    }
    setLoadingStats(false)
  }

  // ── Column toggle for outlier scope ─────────────────────────────────────────
  const toggleCol = (col) => {
    setSelectedCols(prev => {
      const next = new Set(prev)
      if (next.has(col)) next.delete(col)
      else next.add(col)
      return next
    })
  }

  const selectColGroup = (cols) => setSelectedCols(new Set(cols))
  const deselectAll    = ()     => setSelectedCols(new Set())

  // ── Delete dataset (with optional force cascade) ──────────────────────────────
  const handleDelete = async (id, force = false) => {
    setDeleting(true)
    setError('')
    try {
      await deleteDataset(id, force)
      if (selectedDataset?.id === id) {
        setSelectedDataset(null)
        setStats(null)
        setResult(null)
      }
      setConfirmDeleteId(null)
      setForceDeleteId(null)
      const refreshed = await listDatasets()
      setDatasets(refreshed.data)
    } catch (e) {
      const detail = e.response?.data?.detail || e.message
      // If backend says the dataset is linked to runs, offer force-delete
      if (e.response?.status === 409 && detail.includes('run')) {
        setForceDeleteId(id)
        setForceDeleteMsg(detail)
        setConfirmDeleteId(null)
      } else {
        setError('Delete failed: ' + detail)
        setConfirmDeleteId(null)
      }
    }
    setDeleting(false)
  }

  // ── Preview duplicate group data ─────────────────────────────────────────────
  const openGroupPreview = async (grp, gi) => {
    setPreviewGroup({ ...grp, gi })
    setPreviewRows([])
    setLoadingPreview(true)
    try {
      const r = await previewDataset(selectedDataset.id, 30)
      // Filter rows to only the columns in this duplicate group
      const filtered = r.data.rows.map(row =>
        Object.fromEntries(grp.columns.map(c => [c, row[c]]))
      )
      setPreviewRows(filtered)
    } catch {
      setPreviewRows([])
    }
    setLoadingPreview(false)
  }

  const closePreview = () => { setPreviewGroup(null); setPreviewRows([]) }

  // ── Threshold defaults per method ────────────────────────────────────────────
  const handleMethodChange = (method) => {
    setOutlierMethod(method)
    if (method === 'iqr')    setThreshold(1.5)
    if (method === 'zscore') setThreshold(3.0)
  }

  // ── Estimate rows to be removed (rough upper bound shown as preview) ─────────
  const estimateNanRows = stats?.null_rows ?? 0

  const estimateOutlierRows = (() => {
    if (!stats || outlierMethod === 'none') return null
    // Per-column sum (overlap not deducted — shown as rough upper bound)
    return stats.columns
      .filter(c => selectedCols.has(c.name))
      .reduce((acc, c) => {
        if (outlierMethod === 'iqr') {
          // Interpolate linearly between the two pre-computed counts
          if (threshold <= 1.5) return acc + c.iqr_outliers_15
          if (threshold >= 3.0) return acc + c.iqr_outliers_30
          const t = (threshold - 1.5) / 1.5
          return acc + Math.round(c.iqr_outliers_15 + t * (c.iqr_outliers_30 - c.iqr_outliers_15))
        }
        return acc + c.zscore_outliers_3
      }, 0)
  })()

  // ── Apply preprocessing ──────────────────────────────────────────────────────
  const handleApply = async () => {
    setApplying(true)
    setError('')
    setResult(null)
    try {
      const r = await preprocessDataset(selectedDataset.id, {
        dataset_name:       datasetName.trim() || null,
        columns_to_drop:    colsToDrop.size > 0 ? [...colsToDrop] : [],
        remove_nan:         removeNan,
        outlier_method:     outlierMethod,
        outlier_threshold:  threshold,
        columns_to_check:   outlierMethod === 'none'
          ? []
          : [...selectedCols].filter(c => !colsToDrop.has(c)),
      })
      setResult(r.data)
      // Refresh datasets list so Train page shows the new cleaned dataset immediately
      const refreshed = await listDatasets()
      setDatasets(refreshed.data)
    } catch (e) {
      setError('Preprocessing failed: ' + (e.response?.data?.detail || e.message))
    }
    setApplying(false)
  }

  // ── Derived display values ───────────────────────────────────────────────────
  const colsWithNulls    = stats?.columns.filter(c => c.null_count > 0) ?? []
  const colsWithOutliers = stats?.columns.filter(c =>
    outlierMethod === 'iqr'
      ? c.iqr_outliers_15 > 0
      : c.zscore_outliers_3 > 0
  ) ?? []
  const displayedCols = showAllCols ? (stats?.columns ?? []) : (stats?.columns.slice(0, 10) ?? [])

  // ── Render ───────────────────────────────────────────────────────────────────
  return (
    <div className="p-8 max-w-5xl mx-auto">

      <h1 className="text-2xl font-bold text-white mb-1">Data Preprocessing</h1>
      <p className="text-gray-200 text-sm mb-6">
        Clean your dataset before training — remove missing values and outliers,
        then send the cleaned data directly to the model trainer.
      </p>

      {/* ── 1. Select Dataset ─────────────────────────────────────────────────── */}
      <div className="bg-gray-800 rounded-xl p-6 mb-4">
        <h2 className="text-white font-semibold mb-3">Select Dataset</h2>
        {datasets.length === 0 ? (
          <p className="text-gray-500 text-sm">No datasets loaded yet. Go to Upload Data or Train Model to load one.</p>
        ) : (
          <div className="flex flex-wrap gap-2">
            {datasets.map(d => (
              <div key={d.id} className="relative group flex items-stretch">
                {forceDeleteId === d.id ? (
                  /* ── Force-delete prompt (dataset linked to runs) ── */
                  <div className="flex flex-col gap-2 px-3 py-2 rounded-lg border border-orange-600 bg-orange-900/30 text-xs max-w-sm">
                    <span className="text-orange-200 font-semibold">⚠ Dataset is used by existing run(s)</span>
                    <span className="text-gray-200">{forceDeleteMsg}</span>
                    <div className="flex gap-2 mt-1">
                      <button
                        onClick={() => handleDelete(d.id, true)}
                        disabled={deleting}
                        className="bg-red-600 hover:bg-red-500 disabled:opacity-50 text-white px-3 py-1 rounded text-xs font-bold"
                      >
                        {deleting ? '…' : '⚠ Force Delete + Runs'}
                      </button>
                      <button
                        onClick={() => { setForceDeleteId(null); setForceDeleteMsg('') }}
                        className="text-gray-200 hover:text-white text-xs"
                      >Cancel</button>
                    </div>
                  </div>
                ) : confirmDeleteId === d.id ? (
                  /* ── Inline confirmation ── */
                  <div className="flex items-center gap-2 px-3 py-2 rounded-lg border border-red-600 bg-red-900/40 text-sm">
                    <span className="text-red-300">Delete <span className="font-semibold">{d.original_name}</span>?</span>
                    <button
                      onClick={() => handleDelete(d.id)}
                      disabled={deleting}
                      className="bg-red-600 hover:bg-red-500 disabled:opacity-50 text-white px-2 py-0.5 rounded text-xs font-bold transition-colors"
                    >
                      {deleting ? '…' : 'Yes, delete'}
                    </button>
                    <button
                      onClick={() => setConfirmDeleteId(null)}
                      disabled={deleting}
                      className="text-gray-200 hover:text-white px-1 text-xs transition-colors"
                    >
                      Cancel
                    </button>
                  </div>
                ) : (
                  /* ── Normal dataset chip ── */
                  <>
                    <button
                      onClick={() => handleSelectDataset(d)}
                      disabled={loadingStats}
                      className={`px-4 py-2 rounded-l-lg text-sm border-y border-l transition-colors disabled:opacity-50 ${
                        selectedDataset?.id === d.id
                          ? 'bg-blue-600 border-blue-500 text-white'
                          : 'bg-gray-700 border-gray-600 text-gray-100 hover:border-blue-500'
                      }`}
                    >
                      <span className="font-medium">{d.original_name}</span>
                      <span className="text-xs opacity-60 ml-2">{d.row_count?.toLocaleString()} rows</span>
                    </button>
                    <button
                      onClick={() => setConfirmDeleteId(d.id)}
                      disabled={loadingStats || deleting}
                      title="Remove dataset"
                      className={`px-2 py-2 rounded-r-lg text-sm border-y border-r transition-colors
                        disabled:opacity-30 opacity-0 group-hover:opacity-100
                        ${selectedDataset?.id === d.id
                          ? 'bg-blue-700 border-blue-500 text-blue-200 hover:bg-red-700 hover:border-red-500 hover:text-white'
                          : 'bg-gray-700 border-gray-600 text-gray-500 hover:bg-red-700 hover:border-red-500 hover:text-white'
                        }`}
                    >
                      ✕
                    </button>
                  </>
                )}
              </div>
            ))}
          </div>
        )}
      </div>

      {loadingStats && (
        <div className="bg-gray-800 rounded-xl p-6 mb-4 text-center text-blue-400 animate-pulse text-sm">
          Loading column statistics...
        </div>
      )}

      {/* ── 2. Data Summary ───────────────────────────────────────────────────── */}
      {stats && selectedDataset && (
        <>
          {/* Summary cards */}
          <div className="grid grid-cols-4 gap-3 mb-4">
            {[
              { label: 'Total Rows',      value: stats.total_rows.toLocaleString(),            color: 'text-white' },
              { label: 'Rows with Nulls', value: stats.null_rows.toLocaleString(),             color: stats.null_rows > 0 ? 'text-orange-400' : 'text-green-400' },
              { label: 'Columns',         value: stats.columns.length,                         color: 'text-blue-400' },
              { label: 'Cols with Nulls', value: colsWithNulls.length,                        color: colsWithNulls.length > 0 ? 'text-orange-400' : 'text-green-400' },
            ].map(card => (
              <div key={card.label} className="bg-gray-800 rounded-xl p-4 text-center">
                <p className={`text-2xl font-bold ${card.color}`}>{card.value}</p>
                <p className="text-gray-500 text-xs mt-1">{card.label}</p>
              </div>
            ))}
          </div>

          {/* ── Step 0: Duplicate Columns ────────────────────────────────────── */}
          {stats.duplicate_column_groups?.length > 0 ? (
            <div className="bg-gray-800 rounded-xl p-6 mb-4 border border-yellow-700/50">
              <div className="flex items-center gap-3 mb-4">
                <span className="text-yellow-400 text-lg">⚠</span>
                <div>
                  <h2 className="text-white font-semibold">
                    Step 1 — Duplicate Columns Detected
                  </h2>
                  <p className="text-gray-200 text-xs mt-0.5">
                    {stats.duplicate_column_groups.length} group{stats.duplicate_column_groups.length > 1 ? 's' : ''} of
                    columns with identical data found. Duplicates are pre-selected for removal.
                  </p>
                </div>
                <span className="ml-auto text-xs bg-yellow-900 text-yellow-300 px-2 py-1 rounded-full font-semibold">
                  {colsToDrop.size} column{colsToDrop.size !== 1 ? 's' : ''} marked for removal
                </span>
              </div>

              <div className="space-y-3">
                {stats.duplicate_column_groups.map((grp, gi) => (
                  <div key={gi} className="bg-gray-900/50 rounded-lg p-4 border border-gray-700">
                    <div className="flex items-center justify-between mb-2">
                      <p className="text-gray-200 text-xs font-medium uppercase tracking-wide">
                        Duplicate Group {gi + 1} — {grp.columns.length} identical columns
                      </p>
                      <button
                        onClick={() => openGroupPreview(grp, gi)}
                        className="flex items-center gap-1.5 text-xs text-blue-400 hover:text-blue-300
                                   border border-blue-800 hover:border-blue-500 px-2.5 py-1 rounded-lg
                                   transition-colors"
                      >
                        🔍 Preview Data
                      </button>
                    </div>
                    <div className="flex flex-wrap gap-2">
                      {grp.columns.map((col, ci) => {
                        const isKept    = ci === 0                 // first = original/kept
                        const isDropped = colsToDrop.has(col)
                        const isX = selectedDataset.x_columns?.includes(col)
                        const isY = selectedDataset.y_columns?.includes(col)
                        return (
                          <button
                            key={col}
                            onClick={() => {
                              if (isKept) return               // cannot drop the primary
                              setColsToDrop(prev => {
                                const next = new Set(prev)
                                if (next.has(col)) next.delete(col)
                                else next.add(col)
                                return next
                              })
                            }}
                            disabled={isKept}
                            title={isKept ? 'Primary column — kept by default' : isDropped ? 'Click to keep' : 'Click to remove'}
                            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs border transition-colors
                              ${isKept
                                ? 'bg-green-900/40 border-green-700 text-green-300 cursor-default'
                                : isDropped
                                  ? 'bg-red-900/40 border-red-600 text-red-300 line-through hover:border-red-400'
                                  : 'bg-gray-700 border-gray-600 text-gray-100 hover:border-yellow-500'
                              }`}
                          >
                            {isKept && <span title="Kept">✓</span>}
                            {!isKept && isDropped && <span title="Will be removed">✕</span>}
                            {!isKept && !isDropped && <span title="Toggle">○</span>}
                            <span className="font-mono">{col}</span>
                            {isX && <span className="text-blue-400 font-bold text-[10px]">X</span>}
                            {isY && <span className="text-green-400 font-bold text-[10px]">Y</span>}
                            {isKept && <span className="ml-1 text-[10px] text-green-500 not-italic no-underline">(primary)</span>}
                          </button>
                        )
                      })}
                    </div>
                    <p className="text-gray-600 text-xs mt-2">
                      Click any duplicate to toggle keep/remove. Primary column (✓) is always kept.
                    </p>
                  </div>
                ))}
              </div>

              {colsToDrop.size === 0 && (
                <p className="text-gray-500 text-xs mt-3 italic">
                  No columns currently marked for removal — all duplicates will be kept.
                </p>
              )}
            </div>
          ) : stats.duplicate_column_groups?.length === 0 ? (
            <div className="bg-gray-800/50 rounded-xl px-5 py-3 mb-4 flex items-center gap-2 border border-gray-700">
              <span className="text-green-400">✓</span>
              <span className="text-gray-200 text-sm">
                <span className="text-white font-medium">Step 1 — No duplicate columns</span> found in this dataset.
              </span>
            </div>
          ) : null}

          {/* Column stats table */}
          <div className="bg-gray-800 rounded-xl p-6 mb-4">
            <div className="flex items-center justify-between mb-3">
              <h2 className="text-white font-semibold">Column Statistics</h2>
              <span className="text-gray-500 text-xs">
                {stats.columns.length} columns · IQR outliers shown at 1.5× threshold
              </span>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-xs text-gray-100">
                <thead>
                  <tr className="border-b border-gray-700 text-gray-500">
                    <th className="text-left py-2 pr-4">Column</th>
                    <th className="text-left py-2 pr-4">Type</th>
                    <th className="text-right py-2 pr-4">Nulls</th>
                    <th className="text-right py-2 pr-4">Null %</th>
                    <th className="text-right py-2 pr-4">Mean</th>
                    <th className="text-right py-2 pr-4">Std</th>
                    <th className="text-right py-2 pr-4">Min</th>
                    <th className="text-right py-2 pr-4">Max</th>
                    <th className="text-right py-2">IQR Outliers</th>
                  </tr>
                </thead>
                <tbody>
                  {displayedCols.map((c) => {
                    const isX = selectedDataset.x_columns?.includes(c.name)
                    const isY = selectedDataset.y_columns?.includes(c.name)
                    return (
                      <tr key={c.name} className="border-b border-gray-700/40 hover:bg-gray-700/20">
                        <td className="py-1.5 pr-4 font-mono font-medium text-gray-200 max-w-[160px] truncate" title={c.name}>
                          {c.name}
                        </td>
                        <td className="py-1.5 pr-4">
                          {isX && <span className="text-blue-400 text-[10px] font-semibold">X</span>}
                          {isY && <span className="text-green-400 text-[10px] font-semibold">Y</span>}
                          {!isX && !isY && <span className="text-gray-600">—</span>}
                        </td>
                        <td className={`py-1.5 pr-4 text-right font-mono ${nullBg(c.null_pct)}`}>
                          {c.null_count.toLocaleString()}
                        </td>
                        <td className={`py-1.5 pr-4 text-right font-mono ${nullBg(c.null_pct)}`}>
                          {c.null_pct}%
                        </td>
                        <td className="py-1.5 pr-4 text-right font-mono">{c.mean ?? '—'}</td>
                        <td className="py-1.5 pr-4 text-right font-mono">{c.std  ?? '—'}</td>
                        <td className="py-1.5 pr-4 text-right font-mono">{c.min  ?? '—'}</td>
                        <td className="py-1.5 pr-4 text-right font-mono">{c.max  ?? '—'}</td>
                        <td className={`py-1.5 text-right font-mono ${outlierBg(c.iqr_outliers_15, stats.total_rows)}`}>
                          {c.iqr_outliers_15.toLocaleString()}
                          <span className="text-gray-600 ml-1">
                            ({pct(c.iqr_outliers_15, stats.total_rows)}%)
                          </span>
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>

            {stats.columns.length > 10 && (
              <button
                onClick={() => setShowAllCols(v => !v)}
                className="mt-3 text-xs text-blue-400 hover:text-blue-300 transition-colors"
              >
                {showAllCols
                  ? '▲ Show fewer columns'
                  : `▼ Show all ${stats.columns.length} columns`}
              </button>
            )}
          </div>

          {/* ── 3. Preprocessing Options ────────────────────────────────────────── */}
          <div className="bg-gray-800 rounded-xl p-6 mb-4 space-y-6">
            <h2 className="text-white font-semibold">Row Cleaning Options</h2>

            {/* NaN / Null handling */}
            <div>
              <p className="text-gray-100 text-sm font-medium mb-2">Step 2 — Missing Values (NaN / Null)</p>
              <label className="flex items-center gap-3 cursor-pointer">
                <input
                  type="checkbox"
                  checked={removeNan}
                  onChange={e => setRemoveNan(e.target.checked)}
                  className="w-4 h-4 accent-blue-500"
                />
                <span className="text-gray-100 text-sm">
                  Drop all rows containing any NaN / Null value
                </span>
                {stats.null_rows > 0 && (
                  <span className="text-orange-400 text-xs ml-auto">
                    Will remove {stats.null_rows.toLocaleString()} rows
                    ({pct(stats.null_rows, stats.total_rows)}%)
                  </span>
                )}
                {stats.null_rows === 0 && (
                  <span className="text-green-400 text-xs ml-auto">Dataset has no nulls</span>
                )}
              </label>
            </div>

            {/* Outlier method */}
            <div>
              <p className="text-gray-100 text-sm font-medium mb-2">Step 3 — Outlier Detection Method</p>
              <div className="flex gap-3">
                {[
                  { val: 'none',   label: 'None',    desc: 'Keep all rows' },
                  { val: 'iqr',    label: 'IQR',     desc: 'Interquartile range' },
                  { val: 'zscore', label: 'Z-Score', desc: 'Standard deviations' },
                ].map(opt => (
                  <button
                    key={opt.val}
                    onClick={() => handleMethodChange(opt.val)}
                    className={`flex-1 px-4 py-3 rounded-lg border text-sm transition-colors ${
                      outlierMethod === opt.val
                        ? 'bg-blue-600 border-blue-500 text-white'
                        : 'bg-gray-700 border-gray-600 text-gray-100 hover:border-gray-500'
                    }`}
                  >
                    <div className="font-semibold">{opt.label}</div>
                    <div className="text-xs opacity-70 mt-0.5">{opt.desc}</div>
                  </button>
                ))}
              </div>
            </div>

            {/* Threshold slider (only when a method is selected) */}
            {outlierMethod !== 'none' && (
              <div>
                <div className="flex items-center justify-between mb-2">
                  <p className="text-gray-100 text-sm font-medium">
                    {outlierMethod === 'iqr' ? 'IQR Multiplier' : 'Z-Score Threshold'}
                  </p>
                  <span className="text-white font-bold text-sm bg-gray-700 px-3 py-1 rounded-lg">
                    {threshold.toFixed(1)}
                  </span>
                </div>

                <input
                  type="range"
                  min={outlierMethod === 'iqr' ? 1.0 : 2.0}
                  max={outlierMethod === 'iqr' ? 4.0 : 5.0}
                  step="0.5"
                  value={threshold}
                  onChange={e => setThreshold(Number(e.target.value))}
                  className="w-full accent-blue-500"
                />

                <div className="flex justify-between text-xs text-gray-500 mt-1">
                  {outlierMethod === 'iqr' ? (
                    <>
                      <span>1.0× — Strict (removes more)</span>
                      <span>4.0× — Lenient (removes less)</span>
                    </>
                  ) : (
                    <>
                      <span>2.0σ — Strict (removes more)</span>
                      <span>5.0σ — Lenient (removes less)</span>
                    </>
                  )}
                </div>

                <p className="text-gray-500 text-xs mt-2">
                  {outlierMethod === 'iqr'
                    ? `Removes rows where any selected column falls outside [Q1 − ${threshold}×IQR, Q3 + ${threshold}×IQR].`
                    : `Removes rows where any selected column has a Z-score above ${threshold} (i.e., more than ${threshold} standard deviations from the mean).`}
                </p>
              </div>
            )}

            {/* Column selection for outlier scope */}
            {outlierMethod !== 'none' && (
              <div>
                <div className="flex items-center justify-between mb-2">
                  <p className="text-gray-100 text-sm font-medium">
                    Apply Outlier Detection to Columns
                    <span className="text-gray-500 ml-2 font-normal">
                      ({selectedCols.size} / {selectedDataset.columns.length} selected)
                    </span>
                  </p>
                  <div className="flex gap-2">
                    {selectedDataset.x_columns?.length > 0 && (
                      <button
                        onClick={() => selectColGroup(selectedDataset.x_columns)}
                        className="text-xs text-blue-400 hover:text-blue-300 px-2 py-1 border border-blue-800 rounded"
                      >X only</button>
                    )}
                    {selectedDataset.y_columns?.length > 0 && (
                      <button
                        onClick={() => selectColGroup(selectedDataset.y_columns)}
                        className="text-xs text-green-400 hover:text-green-300 px-2 py-1 border border-green-800 rounded"
                      >Y only</button>
                    )}
                    <button
                      onClick={() => selectColGroup(selectedDataset.columns)}
                      className="text-xs text-gray-100 hover:text-white px-2 py-1 border border-gray-600 rounded"
                    >All</button>
                    <button
                      onClick={deselectAll}
                      className="text-xs text-gray-500 hover:text-gray-100 px-2 py-1 border border-gray-700 rounded"
                    >None</button>
                  </div>
                </div>

                <div className="flex flex-wrap gap-1.5 max-h-40 overflow-y-auto p-2 bg-gray-900/40 rounded-lg border border-gray-700">
                  {selectedDataset.columns.map(col => {
                    const isX = selectedDataset.x_columns?.includes(col)
                    const isY = selectedDataset.y_columns?.includes(col)
                    const active = selectedCols.has(col)
                    return (
                      <button
                        key={col}
                        onClick={() => toggleCol(col)}
                        className={`text-xs px-2.5 py-1 rounded-full border transition-colors ${
                          active
                            ? isX ? 'bg-blue-600 border-blue-500 text-white'
                              : isY ? 'bg-green-600 border-green-500 text-white'
                              : 'bg-gray-600 border-gray-500 text-white'
                            : 'bg-gray-800 border-gray-700 text-gray-500 hover:border-gray-500'
                        }`}
                        title={isX ? 'X column' : isY ? 'Y column' : ''}
                      >
                        {col}
                      </button>
                    )
                  })}
                </div>
              </div>
            )}
          </div>

          {/* ── 4. Preview ───────────────────────────────────────────────────────── */}
          <div className="bg-gray-800/60 border border-gray-700 rounded-xl p-5 mb-4">
            <h2 className="text-gray-100 text-sm font-semibold mb-3">Estimated Impact</h2>
            <div className="grid grid-cols-3 gap-4 text-center">
              <div>
                <p className="text-white text-xl font-bold">{stats.total_rows.toLocaleString()}</p>
                <p className="text-gray-500 text-xs mt-1">Original rows</p>
              </div>
              <div>
                <p className="text-red-400 text-xl font-bold">
                  −{(removeNan ? estimateNanRows : 0).toLocaleString()}
                  {estimateOutlierRows != null && outlierMethod !== 'none' && (
                    <span className="text-orange-400"> −~{estimateOutlierRows.toLocaleString()}</span>
                  )}
                </p>
                <p className="text-gray-500 text-xs mt-1">
                  Rows to remove
                  {estimateOutlierRows != null && outlierMethod !== 'none' && (
                    <span className="text-gray-600"> (outlier est. is upper bound)</span>
                  )}
                </p>
              </div>
              <div>
                <p className="text-green-400 text-xl font-bold">
                  ≈{Math.max(0, stats.total_rows - (removeNan ? estimateNanRows : 0)).toLocaleString()}
                </p>
                <p className="text-gray-500 text-xs mt-1">Rows remaining (before outlier step)</p>
              </div>
            </div>

            {/* Visual retention bar */}
            {stats.total_rows > 0 && (() => {
              const nanPct   = removeNan ? (estimateNanRows / stats.total_rows) * 100 : 0
              const cleanPct = 100 - nanPct
              return (
                <div className="mt-4">
                  <div className="flex rounded-lg overflow-hidden h-5 text-xs">
                    <div
                      className="bg-green-700 flex items-center justify-center text-white transition-all"
                      style={{ width: `${cleanPct}%` }}
                    >
                      {cleanPct > 10 ? `${cleanPct.toFixed(1)}% kept` : ''}
                    </div>
                    {nanPct > 0 && (
                      <div
                        className="bg-red-700 flex items-center justify-center text-white transition-all"
                        style={{ width: `${nanPct}%` }}
                      >
                        {nanPct > 5 ? `${nanPct.toFixed(1)}% null` : ''}
                      </div>
                    )}
                  </div>
                </div>
              )
            })()}
          </div>

          {/* ── 5. Output name + Apply ───────────────────────────────────────────── */}
          <div className="bg-gray-800 rounded-xl p-5 mb-4">
            <label className="block text-gray-100 text-sm font-medium mb-1">
              Output Dataset Name
            </label>
            <div className="flex gap-3 items-center">
              <input
                type="text"
                value={datasetName}
                onChange={e => setDatasetName(e.target.value)}
                placeholder={autoDatasetName(selectedDataset, datasets)}
                maxLength={80}
                className="flex-1 bg-gray-700 border border-gray-600 rounded-lg px-3 py-2
                           text-white text-sm placeholder-gray-500 focus:border-blue-500
                           focus:outline-none transition-colors"
              />
              {datasetName.trim() && (
                <button
                  onClick={() => setDatasetName('')}
                  className="text-xs text-gray-500 hover:text-gray-100 px-2 py-1 rounded border
                             border-gray-700 hover:border-gray-500 transition-colors whitespace-nowrap"
                >
                  ↩ Use auto
                </button>
              )}
            </div>
            <p className="text-gray-500 text-xs mt-1.5">
              {datasetName.trim()
                ? <>Will be saved as <span className="text-white font-mono">{datasetName.trim()}</span></>
                : <>Leave blank to auto-name:&nbsp;
                    <span className="text-blue-300 font-mono">
                      {autoDatasetName(selectedDataset, datasets)}
                    </span>
                  </>
              }
            </p>
          </div>

          {error && (
            <div className="bg-red-900/40 border border-red-700 rounded-lg px-4 py-3 mb-4">
              <p className="text-red-400 text-sm">{error}</p>
            </div>
          )}

          <button
            onClick={handleApply}
            disabled={applying}
            className="w-full bg-blue-600 hover:bg-blue-500 disabled:opacity-50 disabled:cursor-not-allowed
                       text-white py-3 rounded-xl font-bold text-sm transition-colors mb-6"
          >
            {applying ? (
              <span className="flex items-center justify-center gap-2">
                <span className="animate-spin">⏳</span> Applying preprocessing...
              </span>
            ) : (
              'Apply Preprocessing & Create Cleaned Dataset'
            )}
          </button>

          {/* ── 6. Result card ───────────────────────────────────────────────────── */}
          {result && (
            <div className="bg-green-900/30 border border-green-700 rounded-xl p-6">
              <h2 className="text-green-400 font-bold text-lg mb-4">Preprocessing Complete</h2>

              <div className="grid grid-cols-3 gap-3 mb-5">
                {[
                  { label: 'Original Rows',       value: result.original_rows.toLocaleString(),                                   color: 'text-white' },
                  { label: 'Cleaned Rows',         value: result.cleaned_rows.toLocaleString(),                                    color: 'text-green-400' },
                  { label: 'Columns Removed',      value: (result.cols_dropped?.length ?? 0).toString(),                          color: (result.cols_dropped?.length ?? 0) > 0 ? 'text-yellow-400' : 'text-gray-200' },
                  { label: 'NaN Rows Removed',     value: result.nan_rows_removed.toLocaleString(),                               color: result.nan_rows_removed > 0 ? 'text-orange-400' : 'text-gray-200' },
                  { label: 'Outlier Rows Removed', value: result.outlier_rows_removed.toLocaleString(),                           color: result.outlier_rows_removed > 0 ? 'text-orange-400' : 'text-gray-200' },
                ].map(m => (
                  <div key={m.label} className="bg-gray-800/60 rounded-lg p-4 flex items-center justify-between">
                    <span className="text-gray-200 text-sm">{m.label}</span>
                    <span className={`text-xl font-bold ${m.color}`}>{m.value}</span>
                  </div>
                ))}
              </div>

              {/* Retention bar */}
              <div className="mb-5">
                <div className="flex justify-between text-xs text-gray-200 mb-1">
                  <span>Data retained</span>
                  <span className="text-green-400 font-bold">{result.rows_retained_pct}%</span>
                </div>
                <div className="w-full bg-gray-700 rounded-full h-3">
                  <div
                    className="bg-green-500 h-3 rounded-full transition-all"
                    style={{ width: `${result.rows_retained_pct}%` }}
                  />
                </div>
              </div>

              <div className="bg-gray-800/60 rounded-lg px-4 py-3 mb-5">
                <p className="text-gray-200 text-sm">
                  Cleaned dataset saved as{' '}
                  <span className="text-white font-mono font-medium">{result.original_name}</span>
                  {' '}(Dataset ID: <span className="text-blue-400 font-bold">{result.dataset_id}</span>)
                </p>
                <p className="text-gray-500 text-xs mt-1">
                  X and Y column assignments are preserved from the source dataset.
                </p>
              </div>

              <button
                onClick={() => navigate('/train', { state: { datasetId: result.dataset_id } })}
                className="w-full bg-green-600 hover:bg-green-500 text-white py-3 rounded-xl
                           font-bold text-sm transition-colors"
              >
                Send to Train Model →
              </button>
            </div>
          )}
        </>
      )}
      {/* ── Duplicate column preview modal ─────────────────────────────────── */}
      {previewGroup && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm"
          onClick={closePreview}
        >
          <div
            className="bg-gray-900 border border-gray-700 rounded-2xl shadow-2xl w-full max-w-4xl mx-4
                       max-h-[85vh] flex flex-col"
            onClick={e => e.stopPropagation()}
          >
            {/* Modal header */}
            <div className="flex items-start justify-between p-5 border-b border-gray-700">
              <div>
                <h3 className="text-white font-bold text-base">
                  Preview — Duplicate Group {previewGroup.gi + 1}
                </h3>
                <p className="text-gray-200 text-xs mt-1">
                  Showing first 30 rows for the {previewGroup.columns.length} identical columns.
                  Values are the same across all columns in this group.
                </p>
                {/* Column legend */}
                <div className="flex flex-wrap gap-2 mt-2">
                  {previewGroup.columns.map((col, ci) => {
                    const isX = selectedDataset?.x_columns?.includes(col)
                    const isY = selectedDataset?.y_columns?.includes(col)
                    return (
                      <span
                        key={col}
                        className={`inline-flex items-center gap-1 text-xs px-2 py-0.5 rounded-full border font-mono
                          ${ci === 0
                            ? 'bg-green-900/50 border-green-700 text-green-300'
                            : colsToDrop.has(col)
                              ? 'bg-red-900/50 border-red-700 text-red-300 line-through'
                              : 'bg-gray-700 border-gray-600 text-gray-100'
                          }`}
                      >
                        {ci === 0 ? '✓ ' : '✕ '}{col}
                        {isX && <span className="text-blue-400 font-bold not-italic">X</span>}
                        {isY && <span className="text-green-400 font-bold not-italic">Y</span>}
                        {ci === 0 && <span className="text-green-500 text-[10px] not-italic">(kept)</span>}
                        {ci > 0 && colsToDrop.has(col) && <span className="text-red-400 text-[10px] not-italic">(drop)</span>}
                        {ci > 0 && !colsToDrop.has(col) && <span className="text-gray-500 text-[10px] not-italic">(kept)</span>}
                      </span>
                    )
                  })}
                </div>
              </div>
              <button
                onClick={closePreview}
                className="text-gray-500 hover:text-white text-xl leading-none ml-4 mt-1 transition-colors"
              >
                ✕
              </button>
            </div>

            {/* Modal body — data table */}
            <div className="overflow-auto flex-1 p-4">
              {loadingPreview ? (
                <div className="flex items-center justify-center h-40 text-blue-400 animate-pulse text-sm">
                  Loading data…
                </div>
              ) : previewRows.length === 0 ? (
                <div className="flex items-center justify-center h-40 text-gray-500 text-sm">
                  No data available.
                </div>
              ) : (
                <table className="w-full text-xs border-collapse">
                  <thead className="sticky top-0">
                    <tr className="bg-gray-800">
                      <th className="text-left text-gray-500 font-medium py-2 px-3 border-b border-gray-700 w-12">
                        Row
                      </th>
                      {previewGroup.columns.map((col, ci) => (
                        <th
                          key={col}
                          className={`text-left font-medium py-2 px-3 border-b border-gray-700 font-mono
                            ${ci === 0 ? 'text-green-300' : colsToDrop.has(col) ? 'text-red-300' : 'text-gray-100'}`}
                        >
                          <div className="flex items-center gap-1">
                            {ci === 0 ? '✓' : '✕'} {col}
                            {ci === 0 && <span className="text-green-600 text-[10px] font-sans">(primary)</span>}
                          </div>
                        </th>
                      ))}
                      <th className="text-center text-gray-500 font-medium py-2 px-3 border-b border-gray-700">
                        Identical?
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {previewRows.map((row, ri) => {
                      const vals    = previewGroup.columns.map(c => row[c])
                      const allSame = vals.every(v => String(v) === String(vals[0]))
                      return (
                        <tr
                          key={ri}
                          className={`border-b border-gray-800 hover:bg-gray-800/50 transition-colors
                            ${!allSame ? 'bg-red-900/10' : ''}`}
                        >
                          <td className="py-1.5 px-3 text-gray-600">{ri + 1}</td>
                          {previewGroup.columns.map((col, ci) => (
                            <td
                              key={col}
                              className={`py-1.5 px-3 font-mono
                                ${ci === 0 ? 'text-gray-200' : 'text-gray-200'}`}
                            >
                              {row[col] ?? <span className="text-red-500 italic">null</span>}
                            </td>
                          ))}
                          <td className="py-1.5 px-3 text-center">
                            {allSame
                              ? <span className="text-green-400 text-sm">✓</span>
                              : <span className="text-red-400 text-sm" title="Values differ!">✗</span>
                            }
                          </td>
                        </tr>
                      )
                    })}
                  </tbody>
                </table>
              )}
            </div>

            {/* Modal footer */}
            <div className="flex items-center justify-between px-5 py-3 border-t border-gray-700 bg-gray-900/50 rounded-b-2xl">
              <p className="text-gray-500 text-xs">
                Showing 30 rows · Click outside or ✕ to close
              </p>
              <div className="flex gap-2">
                {/* Quick toggle buttons from within the modal */}
                {previewGroup.columns.slice(1).map(col => (
                  <button
                    key={col}
                    onClick={() => setColsToDrop(prev => {
                      const next = new Set(prev)
                      if (next.has(col)) next.delete(col); else next.add(col)
                      return next
                    })}
                    className={`text-xs px-3 py-1.5 rounded-lg border transition-colors font-mono
                      ${colsToDrop.has(col)
                        ? 'bg-green-900/40 border-green-700 text-green-300 hover:bg-green-900/70'
                        : 'bg-red-900/40 border-red-700 text-red-300 hover:bg-red-900/70'
                      }`}
                  >
                    {colsToDrop.has(col) ? `↩ Keep ${col}` : `✕ Drop ${col}`}
                  </button>
                ))}
                <button
                  onClick={closePreview}
                  className="text-xs px-4 py-1.5 bg-gray-700 hover:bg-gray-600 text-white rounded-lg
                             border border-gray-600 transition-colors"
                >
                  Close
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
