import { useState, useEffect, useCallback } from 'react'
import { useLocation } from 'react-router-dom'
import {
  listDatasets, loadProjectExcel,
  trainModel, getRun, stopTraining, killAllTraining, getActiveRun,
  deleteDataset, downloadDataset,
  startTuning, getTuningRun, getActiveTuning,
} from '../services/api'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer
} from 'recharts'

const STATUS_COLOR = {
  pending:  'text-yellow-400',
  training: 'text-blue-400',
  done:     'text-green-400',
  stopped:  'text-orange-400',
  error:    'text-red-400',
}

const STATUS_BG = {
  pending:  'bg-yellow-900/50 border-yellow-700',
  training: 'bg-blue-900/50 border-blue-700',
  done:     'bg-green-900/50 border-green-700',
  stopped:  'bg-orange-900/50 border-orange-700',
  error:    'bg-red-900/50 border-red-700',
}

export default function Train() {
  const location = useLocation()
  const [datasets, setDatasets]           = useState([])
  const [selectedDataset, setSelectedDataset] = useState(null)
  const [xCols, setXCols]                 = useState([])
  const [yCols, setYCols]                 = useState([])
  const [params, setParams]               = useState({ noise_factor: 0.1, epochs: 100, hidden_dim: 64, test_size: 20 })
  const [modelType, setModelType]         = useState('dae')
  const [modelParams, setModelParams]     = useState({
    num_layers:    2,
    n_estimators:  100,
    max_depth:     6,
    learning_rate: 0.1,
    gpr_kernel:    'rbf',
    ssm_q_scale:   0.01,
    ssm_r_scale:   0.1,
  })
  const [run, setRun]                     = useState(null)
  const [isActive, setIsActive]           = useState(false)   // any run currently training?
  const [loadingDataset, setLoadingDataset] = useState(false)
  const [stopping, setStopping]           = useState(false)
  const [killing, setKilling]             = useState(false)
  const [error, setError]                 = useState('')
  const [confirmDeleteId, setConfirmDeleteId] = useState(null)
  const [deleting, setDeleting]           = useState(false)
  const [forceDeleteId, setForceDeleteId] = useState(null)
  const [forceDeleteMsg, setForceDeleteMsg] = useState('')

  // ── Hyperparameter Tuning state ─────────────────────────────────────────────
  const [showTunePanel, setShowTunePanel] = useState(false)
  const [tuneCfg, setTuneCfg]             = useState({
    strategy:        'random',
    n_iterations:    10,
    r2_threshold:    85,   // stored as % in UI, converted to 0-1 on send
    per_y_threshold: 80,
  })
  const [focusYCols, setFocusYCols]       = useState([])  // Y cols selected for tuning
  const [tuningRun, setTuningRun]         = useState(null)
  const [isTuning, setIsTuning]           = useState(false)
  const [tuneError, setTuneError]         = useState('')

  // ── Load datasets on mount ──────────────────────────────────────────────────
  const refreshDatasets = () =>
    listDatasets().then(r => setDatasets(r.data)).catch(() => {})

  useEffect(() => { refreshDatasets() }, [])

  // Auto-select dataset: prefer one passed via navigation state (from Preprocess page),
  // otherwise fall back to the project Data_DAE.xlsx
  useEffect(() => {
    if (datasets.length > 0 && !selectedDataset) {
      const stateId = location.state?.datasetId
      if (stateId) {
        const target = datasets.find(d => d.id === stateId)
        if (target) { handleSelectDataset(target); return }
      }
      const dae = datasets.find(d => d.original_name === 'Data_DAE.xlsx')
      if (dae) handleSelectDataset(dae)
    }
  }, [datasets])

  // ── Poll for active run every 2 s ───────────────────────────────────────────
  const pollActive = useCallback(async () => {
    try {
      const r = await getActiveRun()
      if (r.data.active) {
        setIsActive(true)
        // If we're watching a run, refresh its full record
        if (run && run.id === r.data.active.id) {
          const full = await getRun(run.id)
          setRun(full.data)
        } else if (!run) {
          // Someone else started a run (e.g. page reload mid-training)
          const full = await getRun(r.data.active.id)
          setRun(full.data)
        }
      } else {
        if (isActive && run) {
          // Just finished — fetch final state
          const full = await getRun(run.id)
          setRun(full.data)
          // Auto-select poor Y columns for tuning focus
          const perYR2 = full.data.metrics?.per_y_r2 || {}
          const threshold = tuneCfg.per_y_threshold / 100
          const poor = Object.entries(perYR2)
            .filter(([, v]) => v != null && v < threshold)
            .map(([k]) => k)
          if (poor.length > 0) setFocusYCols(poor)
        }
        setIsActive(false)
        setStopping(false)
      }
    } catch {}
  }, [run, isActive])

  useEffect(() => {
    const interval = setInterval(pollActive, 2000)
    return () => clearInterval(interval)
  }, [pollActive])

  // ── Dataset selection ───────────────────────────────────────────────────────
  const handleSelectDataset = (dataset) => {
    setSelectedDataset(dataset)
    setRun(null)
    setError('')
    setConfirmDeleteId(null)

    let xCols = dataset.x_columns || []
    let yCols = dataset.y_columns || []

    // If this dataset has no X/Y assignments (e.g. cleaned dataset created before
    // the copy-fix), auto-inherit from another dataset whose assignments fit
    // within this dataset's columns (e.g. the original Data_DAE.xlsx).
    if (xCols.length === 0 && yCols.length === 0 && dataset.columns?.length > 0) {
      const colSet = new Set(dataset.columns)
      const reference = datasets.find(d =>
        d.id !== dataset.id &&
        (d.x_columns?.length ?? 0) > 0 &&
        d.x_columns.every(c => colSet.has(c))
      )
      if (reference) {
        xCols = reference.x_columns.filter(c => colSet.has(c))
        yCols = (reference.y_columns || []).filter(c => colSet.has(c))
      }
    }

    setXCols(xCols)
    setYCols(yCols)
  }

  const handleLoadExcel = async () => {
    setLoadingDataset(true)
    setError('')
    try {
      const r = await loadProjectExcel()
      await refreshDatasets()
      handleSelectDataset(r.data)
    } catch (e) {
      setError('Failed to load: ' + (e.response?.data?.detail || e.message))
    }
    setLoadingDataset(false)
  }

  const handleDeleteDataset = async (id, force = false) => {
    setDeleting(true)
    setError('')
    try {
      await deleteDataset(id, force)
      if (selectedDataset?.id === id) {
        setSelectedDataset(null)
        setXCols([])
        setYCols([])
      }
      setConfirmDeleteId(null)
      setForceDeleteId(null)
      setForceDeleteMsg('')
      await refreshDatasets()
    } catch (e) {
      const detail = e.response?.data?.detail || e.message
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

  const toggleCol = (col, type) => {
    if (type === 'x') {
      setXCols(p => p.includes(col) ? p.filter(c => c !== col) : [...p, col])
      setYCols(p => p.filter(c => c !== col))
    } else {
      setYCols(p => p.includes(col) ? p.filter(c => c !== col) : [...p, col])
      setXCols(p => p.filter(c => c !== col))
    }
  }

  // ── Start training ──────────────────────────────────────────────────────────
  const handleTrain = async () => {
    setError('')
    if (!selectedDataset) return setError('Please load a dataset first.')
    if (xCols.length === 0) return setError('At least one X (input) column is required.')
    if (yCols.length === 0) return setError('At least one Y (output) column is required.')

    try {
      const r = await trainModel({
        dataset_id:    selectedDataset.id,
        x_columns:     xCols,
        y_columns:     yCols,
        model_type:    modelType,
        noise_factor:  Number(params.noise_factor),
        epochs:        Number(params.epochs),
        hidden_dim:    Number(params.hidden_dim),
        test_size:     Number(params.test_size) / 100,
        num_layers:    Number(modelParams.num_layers),
        n_estimators:  Number(modelParams.n_estimators),
        max_depth:     Number(modelParams.max_depth),
        learning_rate: Number(modelParams.learning_rate),
        gpr_kernel:    modelParams.gpr_kernel,
        ssm_q_scale:   Number(modelParams.ssm_q_scale),
        ssm_r_scale:   Number(modelParams.ssm_r_scale),
      })
      setRun(r.data)
      setIsActive(true)
    } catch (e) {
      const detail = e.response?.data?.detail || e.message
      setError(detail)
    }
  }

  // ── Stop training ───────────────────────────────────────────────────────────
  const handleStop = async () => {
    if (!run || stopping) return
    setStopping(true)
    try {
      await stopTraining(run.id)
    } catch (e) {
      setError('Stop failed: ' + (e.response?.data?.detail || e.message))
      setStopping(false)
    }
  }

  const handleKillAll = async () => {
    if (killing) return
    setKilling(true)
    setError('')
    try {
      await killAllTraining()
      setIsActive(false)
      setStopping(false)
      // Refresh run state
      if (run) {
        const r = await getRun(run.id)
        setRun(r.data)
      }
    } catch (e) {
      setError('Force kill failed: ' + (e.response?.data?.detail || e.message))
    }
    setKilling(false)
  }

  // ── Poll for active tuning run every 2 s ───────────────────────────────────
  const pollTuning = useCallback(async () => {
    try {
      const r = await getActiveTuning()
      if (r.data.active) {
        setIsTuning(true)
        const full = await getTuningRun(r.data.active.id)
        setTuningRun(full.data)
      } else {
        if (isTuning && tuningRun) {
          const full = await getTuningRun(tuningRun.id)
          setTuningRun(full.data)
        }
        setIsTuning(false)
      }
    } catch {}
  }, [isTuning, tuningRun])

  useEffect(() => {
    const id = setInterval(pollTuning, 2000)
    return () => clearInterval(id)
  }, [pollTuning])

  // ── Start tuning ────────────────────────────────────────────────────────────
  const handleStartTuning = async () => {
    setTuneError('')
    if (!selectedDataset) return setTuneError('Please select a dataset first.')
    if (xCols.length === 0) return setTuneError('Select at least one X column.')
    if (yCols.length === 0) return setTuneError('Select at least one Y column.')
    try {
      const r = await startTuning({
        dataset_id:       selectedDataset.id,
        x_columns:        xCols,
        y_columns:        yCols,
        strategy:         tuneCfg.strategy,
        n_iterations:     Number(tuneCfg.n_iterations),
        r2_threshold:     Number(tuneCfg.r2_threshold) / 100,
        per_y_threshold:  Number(tuneCfg.per_y_threshold) / 100,
        test_size:        Number(params.test_size) / 100,
        focus_y_cols:     focusYCols.length > 0 ? focusYCols : null,
      })
      setTuningRun(r.data)
      setIsTuning(true)
    } catch (e) {
      setTuneError(e.response?.data?.detail || e.message)
    }
  }

  // ── Apply best tuning params ────────────────────────────────────────────────
  const handleApplyBest = () => {
    const best = tuningRun?.results?.best_params
    if (!best) return
    setParams(prev => ({
      ...prev,
      noise_factor: best.noise_factor ?? prev.noise_factor,
      epochs:       best.epochs       ?? prev.epochs,
      hidden_dim:   best.hidden_dim   ?? prev.hidden_dim,
    }))
  }

  // ── Chart data ──────────────────────────────────────────────────────────────
  const chartData = run?.metrics?.train_loss_history?.map((v, i) => ({
    epoch:        i + 1,
    'Train Loss': v,
    'Val Loss':   run.metrics.val_loss_history?.[i] ?? null,
  })) ?? []

  const currentEpoch = run?.metrics?.current_epoch ?? 0
  const totalEpochs  = run?.metrics?.total_epochs  ?? Number(params.epochs)
  const progressPct  = totalEpochs > 0 ? Math.round((currentEpoch / totalEpochs) * 100) : 0

  const isDataDae = selectedDataset?.original_name === 'Data_DAE.xlsx'
  const alreadyLoaded = datasets.some(d => d.original_name === 'Data_DAE.xlsx')

  return (
    <div className="p-8 max-w-5xl mx-auto">
      <h1 className="text-2xl font-bold text-white mb-1">Train Model</h1>
      <p className="text-gray-200 text-sm mb-6">
        Select a model type, configure parameters, and train on your dataset.
      </p>

      {/* ── Dataset loader ── */}
      <div className="bg-gray-800 rounded-xl p-6 mb-4">
        <div className="flex items-center justify-between mb-3">
          <div>
            <h2 className="text-white font-semibold">
              {selectedDataset
                ? <>Active Dataset — <span className="text-blue-400">{selectedDataset.original_name}</span></>
                : 'Project Dataset — Data_DAE.xlsx'}
            </h2>
            <p className="text-gray-200 text-xs mt-0.5">
              {selectedDataset ? (
                <>
                  <span className="text-white font-medium">{selectedDataset.row_count?.toLocaleString()}</span> rows
                  {' · '}
                  <span className="text-blue-300 font-medium">{xCols.length} X</span> inputs
                  {' · '}
                  <span className="text-green-300 font-medium">{yCols.length} Y</span> outputs
                  {(xCols.length === 0 || yCols.length === 0) && (
                    <span className="text-yellow-400 ml-2">⚠ Select X and Y columns below</span>
                  )}
                </>
              ) : (
                '44,788 rows · 32 X inputs · 7 Y outputs'
              )}
            </p>
          </div>
          <button
            onClick={handleLoadExcel}
            disabled={loadingDataset || alreadyLoaded}
            className="bg-blue-600 hover:bg-blue-500 disabled:opacity-40 disabled:cursor-not-allowed
                       text-white px-5 py-2 rounded-lg text-sm font-medium transition-colors"
          >
            {loadingDataset ? 'Loading...' : alreadyLoaded ? 'Already Loaded' : 'Load Dataset'}
          </button>
        </div>
        {datasets.length > 0 && (
          <div className="flex flex-wrap gap-2 mt-2">
            {datasets.map(d => (
              <div key={d.id} className="relative group flex items-stretch">
                {forceDeleteId === d.id ? (
                  <div className="flex flex-col gap-2 px-3 py-2 rounded-lg border border-orange-600 bg-orange-900/30 text-xs max-w-xs">
                    <span className="text-orange-200 font-semibold">⚠ Linked to existing run(s)</span>
                    <span className="text-gray-200 leading-snug">{forceDeleteMsg}</span>
                    <div className="flex gap-2">
                      <button onClick={() => handleDeleteDataset(d.id, true)} disabled={deleting}
                        className="bg-red-600 hover:bg-red-500 disabled:opacity-50 text-white px-3 py-1 rounded font-bold">
                        {deleting ? '…' : '⚠ Force Delete + Runs'}
                      </button>
                      <button onClick={() => { setForceDeleteId(null); setForceDeleteMsg('') }}
                        className="text-gray-200 hover:text-white">Cancel</button>
                    </div>
                  </div>
                ) : confirmDeleteId === d.id ? (
                  <div className="flex items-center gap-2 px-3 py-2 rounded-lg border border-red-600 bg-red-900/40 text-sm">
                    <span className="text-red-300 text-xs">Delete <span className="font-semibold">{d.original_name}</span>?</span>
                    <button
                      onClick={() => handleDeleteDataset(d.id)}
                      disabled={deleting}
                      className="bg-red-600 hover:bg-red-500 disabled:opacity-50 text-white px-2 py-0.5 rounded text-xs font-bold transition-colors"
                    >
                      {deleting ? '…' : 'Yes'}
                    </button>
                    <button
                      onClick={() => setConfirmDeleteId(null)}
                      disabled={deleting}
                      className="text-gray-200 hover:text-white text-xs transition-colors"
                    >
                      Cancel
                    </button>
                  </div>
                ) : (
                  <>
                    <button
                      onClick={() => handleSelectDataset(d)}
                      disabled={isActive}
                      className={`px-4 py-2 rounded-l-lg text-sm border-y border-l transition-colors disabled:opacity-50 ${
                        selectedDataset?.id === d.id
                          ? 'bg-blue-600 border-blue-500 text-white'
                          : 'bg-gray-700 border-gray-600 text-gray-100 hover:border-blue-500'
                      }`}
                    >
                      {d.original_name}
                      <span className="text-xs opacity-60 ml-2">{d.row_count?.toLocaleString()} rows</span>
                    </button>
                    {/* Download button */}
                    <a
                      href={downloadDataset(d.id)}
                      download
                      title="Download as CSV"
                      className={`flex items-center px-2 py-2 text-sm border-y transition-colors
                        opacity-0 group-hover:opacity-100
                        ${selectedDataset?.id === d.id
                          ? 'bg-blue-700 border-blue-500 text-blue-200 hover:bg-blue-600 hover:text-white'
                          : 'bg-gray-700 border-gray-600 text-gray-500 hover:bg-gray-600 hover:text-white'
                        }`}
                    >
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" d="M4 16v2a2 2 0 002 2h12a2 2 0 002-2v-2M7 10l5 5m0 0l5-5m-5 5V4" />
                      </svg>
                    </a>
                    <button
                      onClick={() => setConfirmDeleteId(d.id)}
                      disabled={isActive || deleting}
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

      {/* ── Column assignments ── */}
      {selectedDataset && (
        <div className="bg-gray-800 rounded-xl p-6 mb-4">
          <div className="flex items-center gap-3 mb-4">
            <h2 className="text-white font-semibold">Column Assignments</h2>
            {isDataDae && (
              <span className="text-xs bg-green-900 text-green-300 px-2 py-0.5 rounded-full">
                Auto-detected from Variables_Type sheet
              </span>
            )}
            {!isDataDae && selectedDataset?.original_name?.startsWith('cleaned_') && (
              <span className="text-xs bg-blue-900 text-blue-300 px-2 py-0.5 rounded-full">
                Cleaned dataset — X/Y inherited from source
              </span>
            )}
          </div>
          <div className="grid grid-cols-2 gap-6">
            <div>
              <p className="text-blue-400 text-sm font-medium mb-2">X — Inputs ({xCols.length})</p>
              <div className="flex flex-wrap gap-1.5 max-h-40 overflow-y-auto">
                {(selectedDataset.columns || []).map(col => (
                  <button
                    key={col}
                    onClick={() => toggleCol(col, 'x')}
                    disabled={isActive}
                    className={`text-xs px-2.5 py-1 rounded-full border transition-colors ${
                      xCols.includes(col)
                        ? 'bg-blue-600 border-blue-500 text-white'
                        : yCols.includes(col)
                        ? 'bg-gray-700 border-gray-600 text-gray-500 cursor-default'
                        : 'bg-gray-700 border-gray-600 text-gray-200 hover:border-blue-500'
                    } disabled:cursor-not-allowed`}
                  >{col}</button>
                ))}
              </div>
            </div>
            <div>
              <p className="text-green-400 text-sm font-medium mb-2">Y — Outputs ({yCols.length})</p>
              <div className="flex flex-wrap gap-1.5 max-h-40 overflow-y-auto">
                {(selectedDataset.columns || []).map(col => (
                  <button
                    key={col}
                    onClick={() => toggleCol(col, 'y')}
                    disabled={isActive}
                    className={`text-xs px-2.5 py-1 rounded-full border transition-colors ${
                      yCols.includes(col)
                        ? 'bg-green-600 border-green-500 text-white'
                        : xCols.includes(col)
                        ? 'bg-gray-700 border-gray-600 text-gray-500 cursor-default'
                        : 'bg-gray-700 border-gray-600 text-gray-200 hover:border-green-500'
                    } disabled:cursor-not-allowed`}
                  >{col}</button>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── Parameters ── */}
      {selectedDataset && (
        <div className="bg-gray-800 rounded-xl p-6 mb-4">
          <h2 className="text-white font-semibold mb-4">Model Parameters</h2>

          {/* ── Model type selector ── */}
          <div className="mb-6">
            <p className="text-sm text-gray-100 font-medium mb-3">Model Type</p>
            <div className="grid grid-cols-5 gap-2">
              {[
                { id: 'dae',     label: 'DAE',     sub: 'Denoising AutoEncoder', color: 'blue',   icon: '🧠' },
                { id: 'lstm',    label: 'LSTM',    sub: 'Long Short-Term Memory', color: 'purple', icon: '🔁' },
                { id: 'xgboost',label: 'XGBoost', sub: 'Gradient Boosting',      color: 'amber',  icon: '🌲' },
                { id: 'gpr',     label: 'GPR',     sub: 'Gaussian Process',       color: 'cyan',   icon: '📊' },
                { id: 'ssm',     label: 'SSM',     sub: 'State Space / Kalman',   color: 'teal',   icon: '📡' },
              ].map(m => {
                const active = modelType === m.id
                const colorMap = {
                  blue:   { active: 'bg-blue-600 border-blue-500 text-white', hover: 'hover:border-blue-500' },
                  purple: { active: 'bg-purple-600 border-purple-500 text-white', hover: 'hover:border-purple-500' },
                  amber:  { active: 'bg-amber-600 border-amber-500 text-white', hover: 'hover:border-amber-500' },
                  cyan:   { active: 'bg-cyan-700 border-cyan-500 text-white', hover: 'hover:border-cyan-500' },
                  teal:   { active: 'bg-teal-700 border-teal-500 text-white', hover: 'hover:border-teal-500' },
                }
                const cls = colorMap[m.color]
                return (
                  <button
                    key={m.id}
                    onClick={() => !isActive && setModelType(m.id)}
                    disabled={isActive}
                    className={`flex flex-col items-center gap-1 px-2 py-3 rounded-xl border text-center
                                transition-colors disabled:opacity-50 disabled:cursor-not-allowed
                                ${active ? cls.active : `bg-gray-700 border-gray-600 text-gray-200 ${cls.hover}`}`}
                  >
                    <span className="text-xl">{m.icon}</span>
                    <span className="font-bold text-sm">{m.label}</span>
                    <span className={`text-xs leading-tight ${active ? 'text-white/80' : 'text-gray-400'}`}>{m.sub}</span>
                  </button>
                )
              })}
            </div>
            {modelType === 'gpr' && (
              <p className="text-cyan-400 text-xs mt-2">
                ℹ GPR training is capped at 2,000 rows for performance. O(n³) complexity.
              </p>
            )}
            {modelType === 'ssm' && (
              <p className="text-teal-400 text-xs mt-2">
                ℹ State Space Model uses a Kalman Filter — ideal for time-ordered data.
              </p>
            )}
          </div>

          {/* Train / Test split slider */}
          <div className="mb-6">
            <div className="flex items-center justify-between mb-2">
              <label className="text-sm text-gray-100 font-medium">Train / Test Split</label>
              <span className="text-sm font-bold text-white">
                {100 - params.test_size}% Train &nbsp;/&nbsp; {params.test_size}% Test
              </span>
            </div>

            {/* Stacked bar showing the split visually */}
            <div className="flex rounded-lg overflow-hidden h-7 mb-2">
              <div
                className="bg-blue-600 flex items-center justify-center text-xs text-white font-medium transition-all duration-200"
                style={{ width: `${100 - params.test_size}%` }}
              >
                {100 - params.test_size}% Train
              </div>
              <div
                className="bg-green-600 flex items-center justify-center text-xs text-white font-medium transition-all duration-200"
                style={{ width: `${params.test_size}%` }}
              >
                {params.test_size}% Test
              </div>
            </div>

            <input
              type="range"
              min="5" max="50" step="5"
              value={params.test_size}
              disabled={isActive}
              onChange={e => setParams(prev => ({ ...prev, test_size: Number(e.target.value) }))}
              className="w-full accent-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
            />

            <div className="flex justify-between text-xs text-gray-500 mt-1">
              <span>5% Test (95% Train)</span>
              <span>50% Test (50% Train)</span>
            </div>

            {selectedDataset?.row_count && (
              <p className="text-gray-200 text-xs mt-2">
                ≈&nbsp;
                <span className="text-blue-300 font-medium">
                  {Math.round(selectedDataset.row_count * (1 - params.test_size / 100)).toLocaleString()} training rows
                </span>
                &nbsp;·&nbsp;
                <span className="text-green-300 font-medium">
                  {Math.round(selectedDataset.row_count * (params.test_size / 100)).toLocaleString()} test rows
                </span>
                &nbsp;(from {selectedDataset.row_count.toLocaleString()} total)
              </p>
            )}
          </div>

          {/* Model-specific parameters */}
          {(modelType === 'dae' || modelType === 'lstm') && (
            <div className="grid grid-cols-3 gap-4">
              {[
                { key: 'noise_factor', label: 'Noise Factor', hint: '0.0–1.0', step: '0.05', min: '0', max: '1', shared: true },
                { key: 'epochs',       label: 'Epochs',       hint: 'Training cycles', step: '10', min: '10', max: '1000', shared: true },
                { key: 'hidden_dim',   label: 'Hidden Dim',   hint: 'Layer size', step: '16', min: '8', max: '512', shared: true },
              ].map(p => (
                <div key={p.key}>
                  <label className="block text-sm text-gray-100 mb-1">{p.label}</label>
                  <input
                    type="number" step={p.step} min={p.min} max={p.max}
                    value={params[p.key]}
                    disabled={isActive}
                    onChange={e => setParams(prev => ({ ...prev, [p.key]: e.target.value }))}
                    className="w-full bg-gray-700 border border-gray-600 rounded-lg px-3 py-2
                               text-white text-sm focus:border-blue-500 focus:outline-none
                               disabled:opacity-50 disabled:cursor-not-allowed"
                  />
                  <p className="text-gray-500 text-xs mt-1">{p.hint}</p>
                </div>
              ))}
              {modelType === 'lstm' && (
                <div>
                  <label className="block text-sm text-gray-100 mb-1">LSTM Layers</label>
                  <input
                    type="number" step="1" min="1" max="8"
                    value={modelParams.num_layers}
                    disabled={isActive}
                    onChange={e => setModelParams(prev => ({ ...prev, num_layers: e.target.value }))}
                    className="w-full bg-gray-700 border border-gray-600 rounded-lg px-3 py-2
                               text-white text-sm focus:border-purple-500 focus:outline-none
                               disabled:opacity-50 disabled:cursor-not-allowed"
                  />
                  <p className="text-gray-500 text-xs mt-1">Number of stacked layers</p>
                </div>
              )}
            </div>
          )}

          {modelType === 'xgboost' && (
            <div className="grid grid-cols-3 gap-4">
              {[
                { key: 'n_estimators',  label: 'Estimators',     hint: 'Number of trees',   step: '10', min: '10', max: '1000' },
                { key: 'max_depth',     label: 'Max Depth',      hint: 'Tree depth (3–10)', step: '1',  min: '1',  max: '20'   },
                { key: 'learning_rate', label: 'Learning Rate',  hint: '0.01–0.3',          step: '0.01', min: '0.001', max: '1' },
              ].map(p => (
                <div key={p.key}>
                  <label className="block text-sm text-gray-100 mb-1">{p.label}</label>
                  <input
                    type="number" step={p.step} min={p.min} max={p.max}
                    value={modelParams[p.key]}
                    disabled={isActive}
                    onChange={e => setModelParams(prev => ({ ...prev, [p.key]: e.target.value }))}
                    className="w-full bg-gray-700 border border-gray-600 rounded-lg px-3 py-2
                               text-white text-sm focus:border-amber-500 focus:outline-none
                               disabled:opacity-50 disabled:cursor-not-allowed"
                  />
                  <p className="text-gray-500 text-xs mt-1">{p.hint}</p>
                </div>
              ))}
            </div>
          )}

          {modelType === 'gpr' && (
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm text-gray-100 mb-1">Kernel</label>
                <div className="flex rounded-lg overflow-hidden border border-gray-600">
                  {[
                    { id: 'rbf', label: 'RBF (Squared Exp.)' },
                    { id: 'matern52', label: 'Matérn 5/2' },
                  ].map(k => (
                    <button
                      key={k.id}
                      onClick={() => !isActive && setModelParams(prev => ({ ...prev, gpr_kernel: k.id }))}
                      disabled={isActive}
                      className={`flex-1 py-2 px-3 text-sm font-medium transition-colors disabled:opacity-50 ${
                        modelParams.gpr_kernel === k.id
                          ? 'bg-cyan-700 text-white'
                          : 'bg-gray-700 text-gray-200 hover:bg-gray-600'
                      }`}
                    >
                      {k.label}
                    </button>
                  ))}
                </div>
                <p className="text-gray-500 text-xs mt-1">Covariance kernel for GP prior</p>
              </div>
            </div>
          )}

          {modelType === 'ssm' && (
            <div className="grid grid-cols-2 gap-4">
              {[
                { key: 'ssm_q_scale', label: 'Process Noise (Q)', hint: 'State transition noise', step: '0.001', min: '0.0001', max: '1' },
                { key: 'ssm_r_scale', label: 'Obs. Noise (R)',    hint: 'Measurement noise',     step: '0.01',  min: '0.001',  max: '10' },
              ].map(p => (
                <div key={p.key}>
                  <label className="block text-sm text-gray-100 mb-1">{p.label}</label>
                  <input
                    type="number" step={p.step} min={p.min} max={p.max}
                    value={modelParams[p.key]}
                    disabled={isActive}
                    onChange={e => setModelParams(prev => ({ ...prev, [p.key]: e.target.value }))}
                    className="w-full bg-gray-700 border border-gray-600 rounded-lg px-3 py-2
                               text-white text-sm focus:border-teal-500 focus:outline-none
                               disabled:opacity-50 disabled:cursor-not-allowed"
                  />
                  <p className="text-gray-500 text-xs mt-1">{p.hint}</p>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* ── Hyperparameter Tuning Panel ── */}
      {selectedDataset && (
        <div className="bg-gray-800 rounded-xl mb-4 overflow-hidden">
          {/* Collapsible header */}
          <button
            onClick={() => setShowTunePanel(p => !p)}
            className="w-full flex items-center justify-between px-6 py-4 text-white hover:bg-gray-750 transition-colors"
          >
            <div className="flex items-center gap-3">
              <span className="text-lg">🔧</span>
              <span className="font-semibold">Tune Hyperparameters</span>
              <span className="text-xs bg-purple-900 text-purple-300 px-2 py-0.5 rounded-full">
                Grid / Random Search
              </span>
              {isTuning && (
                <span className="text-xs text-yellow-400 animate-pulse">● Running…</span>
              )}
              {tuningRun?.status === 'done' && (
                <span className="text-xs text-green-400">
                  ✓ Best {tuningRun.focus_y_cols?.length > 0 ? 'Focus' : ''} R² {((tuningRun.results?.best_overall_r2 ?? 0) * 100).toFixed(1)}%
                  {tuningRun.focus_y_cols?.length > 0 && (
                    <span className="text-purple-300 ml-1">({tuningRun.focus_y_cols.length} Y focused)</span>
                  )}
                </span>
              )}
            </div>
            <span className="text-gray-200 text-sm">{showTunePanel ? '▲' : '▼'}</span>
          </button>

          {showTunePanel && (
            <div className="px-6 pb-6 border-t border-gray-700">

              {/* ── Config row ── */}
              <div className="grid grid-cols-4 gap-4 mt-4 mb-4">
                {/* Strategy */}
                <div>
                  <label className="block text-sm text-gray-100 mb-1">Strategy</label>
                  <div className="flex rounded-lg overflow-hidden border border-gray-600">
                    {['random', 'grid'].map(s => (
                      <button
                        key={s}
                        onClick={() => setTuneCfg(p => ({ ...p, strategy: s }))}
                        disabled={isTuning}
                        className={`flex-1 py-2 text-sm font-medium transition-colors disabled:opacity-50 ${
                          tuneCfg.strategy === s
                            ? 'bg-purple-600 text-white'
                            : 'bg-gray-700 text-gray-200 hover:bg-gray-600'
                        }`}
                      >
                        {s === 'random' ? '🎲 Random' : '⊞ Grid'}
                      </button>
                    ))}
                  </div>
                  <p className="text-gray-500 text-xs mt-1">
                    {tuneCfg.strategy === 'random' ? 'Samples N random combos' : 'First N of all combos'}
                  </p>
                </div>

                {/* Iterations */}
                <div>
                  <label className="block text-sm text-gray-100 mb-1">Iterations</label>
                  <input
                    type="number" min="1" max="100" step="1"
                    value={tuneCfg.n_iterations}
                    disabled={isTuning}
                    onChange={e => setTuneCfg(p => ({ ...p, n_iterations: e.target.value }))}
                    className="w-full bg-gray-700 border border-gray-600 rounded-lg px-3 py-2
                               text-white text-sm focus:border-purple-500 focus:outline-none
                               disabled:opacity-50 disabled:cursor-not-allowed"
                  />
                  <p className="text-gray-500 text-xs mt-1">Number of trials to run</p>
                </div>

                {/* Overall R² threshold */}
                <div>
                  <label className="block text-sm text-gray-100 mb-1">
                    R² Flag Threshold
                    <span className="text-purple-400 ml-1 font-bold">{tuneCfg.r2_threshold}%</span>
                  </label>
                  <input
                    type="range" min="50" max="99" step="1"
                    value={tuneCfg.r2_threshold}
                    disabled={isTuning}
                    onChange={e => setTuneCfg(p => ({ ...p, r2_threshold: Number(e.target.value) }))}
                    className="w-full accent-purple-500 disabled:opacity-50"
                  />
                  <p className="text-gray-500 text-xs mt-1">Flag if overall R² below this</p>
                </div>

                {/* Per-Y threshold */}
                <div>
                  <label className="block text-sm text-gray-100 mb-1">
                    Per-Y Threshold
                    <span className="text-pink-400 ml-1 font-bold">{tuneCfg.per_y_threshold}%</span>
                  </label>
                  <input
                    type="range" min="50" max="99" step="1"
                    value={tuneCfg.per_y_threshold}
                    disabled={isTuning}
                    onChange={e => setTuneCfg(p => ({ ...p, per_y_threshold: Number(e.target.value) }))}
                    className="w-full accent-pink-500 disabled:opacity-50"
                  />
                  <p className="text-gray-500 text-xs mt-1">Optimize each Y below this</p>
                </div>
              </div>

              {/* ── Y-column focus selector ── */}
              <div className="mb-4">
                <div className="flex items-center justify-between mb-2">
                  <div>
                    <label className="block text-sm text-gray-100 font-medium">
                      Focus Tuning On Y Variables
                    </label>
                    <p className="text-gray-500 text-xs mt-0.5">
                      Select which outputs to optimise. Trials are scored on these columns only.
                      {focusYCols.length === 0 && <span className="text-yellow-400 ml-1">All Y columns used when none selected.</span>}
                    </p>
                  </div>
                  <div className="flex gap-2">
                    <button
                      onClick={() => setFocusYCols(yCols)}
                      disabled={isTuning}
                      className="text-xs text-gray-200 hover:text-white border border-gray-600 hover:border-gray-400
                                 px-2 py-1 rounded transition-colors disabled:opacity-40"
                    >All</button>
                    <button
                      onClick={() => {
                        const perYR2 = run?.metrics?.per_y_r2 || {}
                        const threshold = tuneCfg.per_y_threshold / 100
                        const poor = yCols.filter(c => {
                          const v = perYR2[c]
                          return v != null && v < threshold
                        })
                        setFocusYCols(poor)
                      }}
                      disabled={isTuning || !run?.metrics?.per_y_r2}
                      title={!run?.metrics?.per_y_r2 ? 'Train a model first to see per-Y metrics' : ''}
                      className="text-xs text-orange-400 hover:text-white border border-orange-700 hover:border-orange-400
                                 px-2 py-1 rounded transition-colors disabled:opacity-40"
                    >Poor Only</button>
                    <button
                      onClick={() => setFocusYCols([])}
                      disabled={isTuning}
                      className="text-xs text-gray-200 hover:text-white border border-gray-600 hover:border-gray-400
                                 px-2 py-1 rounded transition-colors disabled:opacity-40"
                    >Clear</button>
                  </div>
                </div>

                <div className="flex flex-wrap gap-2">
                  {yCols.map(col => {
                    const r2      = run?.metrics?.per_y_r2?.[col]
                    const mae     = run?.metrics?.per_y_mae?.[col]
                    const isPoor  = r2 != null && r2 < tuneCfg.per_y_threshold / 100
                    const focused = focusYCols.includes(col)
                    return (
                      <button
                        key={col}
                        onClick={() => !isTuning && setFocusYCols(p =>
                          p.includes(col) ? p.filter(c => c !== col) : [...p, col]
                        )}
                        disabled={isTuning}
                        className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg border text-sm
                                    transition-colors disabled:opacity-50 ${
                          focused
                            ? 'bg-purple-700 border-purple-500 text-white'
                            : isPoor
                            ? 'bg-orange-900/30 border-orange-700 text-orange-300 hover:border-purple-500'
                            : 'bg-gray-700 border-gray-600 text-gray-100 hover:border-purple-500'
                        }`}
                      >
                        <span className="font-medium truncate max-w-32" title={col}>{col}</span>
                        {r2 != null && (
                          <span className={`text-xs font-bold ml-1 ${isPoor ? 'text-orange-400' : 'text-green-400'}`}>
                            {(r2 * 100).toFixed(1)}%
                          </span>
                        )}
                        {mae != null && (
                          <span className="text-xs text-gray-200">MAE {mae.toFixed(3)}</span>
                        )}
                        {r2 == null && (
                          <span className="text-xs text-gray-500">no data</span>
                        )}
                      </button>
                    )
                  })}
                </div>

                {focusYCols.length > 0 && (
                  <p className="text-purple-300 text-xs mt-2">
                    🎯 Optimising for: <span className="font-semibold">{focusYCols.join(', ')}</span>
                  </p>
                )}
              </div>

              {/* Start tuning button */}
              {!isTuning && (
                <button
                  onClick={handleStartTuning}
                  disabled={isActive}
                  title={isActive ? 'Stop training before tuning' : ''}
                  className="bg-purple-600 hover:bg-purple-500 disabled:opacity-40 disabled:cursor-not-allowed
                             text-white px-8 py-2.5 rounded-lg font-semibold text-sm transition-colors mb-4"
                >
                  🚀 Start Tuning ({tuneCfg.n_iterations} trials
                  {focusYCols.length > 0 ? ` · ${focusYCols.length} Y focused` : ''})
                </button>
              )}

              {tuneError && (
                <div className="bg-red-900/40 border border-red-700 rounded-lg px-4 py-2 mb-4">
                  <p className="text-red-400 text-sm">{tuneError}</p>
                </div>
              )}

              {/* ── Progress ── */}
              {tuningRun && (isTuning || tuningRun.status === 'done' || tuningRun.status === 'error') && (() => {
                const trials   = tuningRun.results?.trials ?? []
                const nTotal   = tuningRun.n_iterations
                const nDone    = tuningRun.current_trial ?? trials.length
                const pct      = nTotal > 0 ? Math.round((nDone / nTotal) * 100) : 0
                const best     = tuningRun.results?.best_params
                const bestR2   = tuningRun.results?.best_overall_r2
                const perYBest = tuningRun.results?.per_y_best_params ?? {}
                const perYR2   = tuningRun.results?.per_y_best_r2    ?? {}
                const poorY    = tuningRun.results?.poor_y_columns   ?? []
                const meetsThreshold = tuningRun.results?.meets_threshold

                return (
                  <div>
                    {/* Progress bar */}
                    <div className="mb-4">
                      <div className="flex justify-between text-xs text-gray-200 mb-1">
                        <span>Trial {nDone} / {nTotal}</span>
                        <span className="font-medium">{pct}%{isTuning ? ' · running…' : ''}</span>
                      </div>
                      <div className="w-full bg-gray-700 rounded-full h-2.5">
                        <div
                          className={`h-2.5 rounded-full transition-all duration-500 ${
                            tuningRun.status === 'done' ? 'bg-green-500' :
                            tuningRun.status === 'error' ? 'bg-red-500' : 'bg-purple-500'
                          }`}
                          style={{ width: `${pct}%` }}
                        />
                      </div>
                    </div>

                    {/* Error state */}
                    {tuningRun.status === 'error' && (
                      <div className="bg-red-900/40 border border-red-700 rounded-lg px-4 py-3 mb-4">
                        <p className="text-red-400 text-sm">{tuningRun.results?.error}</p>
                      </div>
                    )}

                    {/* Trial results table */}
                    {trials.length > 0 && (
                      <div className="mb-4">
                        <h3 className="text-gray-100 text-sm font-medium mb-2">
                          Trial Results ({trials.length} completed)
                        </h3>
                        <div className="overflow-auto max-h-64 rounded-lg border border-gray-700">
                          <table className="w-full text-xs text-gray-100 border-collapse">
                            <thead className="sticky top-0 bg-gray-700 text-gray-200">
                              <tr>
                                <th className="px-3 py-2 text-left">#</th>
                                <th className="px-3 py-2 text-left">Noise</th>
                                <th className="px-3 py-2 text-left">Hidden</th>
                                <th className="px-3 py-2 text-left">Epochs</th>
                                {tuningRun.results?.focus_y_cols?.length > 0 && (
                                  <th className="px-3 py-2 text-left text-purple-300">Focus R²</th>
                                )}
                                <th className="px-3 py-2 text-left">All R²</th>
                                {yCols.map(y => (
                                  <th key={y} className="px-3 py-2 text-left truncate max-w-24" title={y}>
                                    {y.length > 10 ? y.slice(0, 10) + '…' : y}
                                  </th>
                                ))}
                              </tr>
                            </thead>
                            <tbody>
                              {trials.map((t, i) => {
                                const isBest = best &&
                                  t.params?.noise_factor === best.noise_factor &&
                                  t.params?.hidden_dim   === best.hidden_dim   &&
                                  t.params?.epochs       === best.epochs
                                return (
                                  <tr
                                    key={i}
                                    className={`border-t border-gray-700 ${
                                      isBest ? 'bg-green-900/30' : i % 2 === 0 ? 'bg-gray-800' : 'bg-gray-750'
                                    }`}
                                  >
                                    <td className="px-3 py-1.5">
                                      {i + 1}{isBest && <span className="text-purple-400 ml-1">★</span>}
                                    </td>
                                    <td className="px-3 py-1.5">{t.params?.noise_factor ?? t.noise_factor}</td>
                                    <td className="px-3 py-1.5">{t.params?.hidden_dim   ?? t.hidden_dim}</td>
                                    <td className="px-3 py-1.5">{t.params?.epochs       ?? t.epochs}</td>
                                    {tuningRun.results?.focus_y_cols?.length > 0 && (
                                      <td className={`px-3 py-1.5 font-bold ${
                                        (t.focus_score ?? 0) >= tuneCfg.r2_threshold / 100
                                          ? 'text-purple-400' : 'text-orange-400'
                                      }`}>
                                        {t.focus_score != null ? `${(t.focus_score * 100).toFixed(1)}%` : '—'}
                                      </td>
                                    )}
                                    <td className={`px-3 py-1.5 font-medium ${
                                      (t.overall_r2 ?? 0) >= tuneCfg.r2_threshold / 100
                                        ? 'text-green-400' : 'text-gray-200'
                                    }`}>
                                      {t.overall_r2 != null ? `${(t.overall_r2 * 100).toFixed(1)}%` : '—'}
                                    </td>
                                    {yCols.map(y => {
                                      const r2 = t.per_y_r2?.[y]
                                      return (
                                        <td key={y} className={`px-3 py-1.5 ${
                                          r2 == null ? 'text-gray-600' :
                                          r2 >= tuneCfg.per_y_threshold / 100 ? 'text-green-400' : 'text-orange-400'
                                        }`}>
                                          {r2 != null ? `${(r2 * 100).toFixed(1)}%` : '—'}
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
                    )}

                    {/* Best params card */}
                    {best && tuningRun.status === 'done' && (
                      <div className={`rounded-xl border p-4 mb-4 ${
                        meetsThreshold
                          ? 'bg-green-900/20 border-green-700'
                          : 'bg-orange-900/20 border-orange-700'
                      }`}>
                        <div className="flex items-start justify-between">
                          <div>
                            <div className="flex items-center gap-2 mb-3">
                              <span className="text-white font-semibold">
                                {meetsThreshold ? '✅ Best Parameters' : '⚠️ Best Parameters Found (below threshold)'}
                              </span>
                              <span className={`text-sm font-bold ${
                                meetsThreshold ? 'text-green-400' : 'text-orange-400'
                              }`}>
                                R² {((bestR2 ?? 0) * 100).toFixed(2)}%
                              </span>
                            </div>
                            <div className="flex gap-6 text-sm">
                              <div>
                                <span className="text-gray-200">Noise Factor</span>
                                <span className="text-white font-bold ml-2">{best.noise_factor}</span>
                              </div>
                              <div>
                                <span className="text-gray-200">Hidden Dim</span>
                                <span className="text-white font-bold ml-2">{best.hidden_dim}</span>
                              </div>
                              <div>
                                <span className="text-gray-200">Epochs</span>
                                <span className="text-white font-bold ml-2">{best.epochs}</span>
                              </div>
                            </div>
                          </div>
                          <button
                            onClick={handleApplyBest}
                            className="bg-purple-600 hover:bg-purple-500 text-white px-4 py-2
                                       rounded-lg text-sm font-semibold transition-colors whitespace-nowrap"
                          >
                            ↑ Apply to Training
                          </button>
                        </div>
                      </div>
                    )}

                    {/* Poor Y columns analysis */}
                    {poorY.length > 0 && tuningRun.status === 'done' && (
                      <div className="bg-orange-900/20 border border-orange-700 rounded-xl p-4">
                        <p className="text-orange-300 font-semibold text-sm mb-3">
                          ⚠ {poorY.length} output variable{poorY.length > 1 ? 's' : ''} below {tuneCfg.per_y_threshold}% R² threshold
                        </p>
                        <p className="text-gray-200 text-xs mb-3">
                          Best achievable R² per variable (individual optimal parameters shown):
                        </p>
                        <div className="space-y-2">
                          {poorY.map(y => {
                            const yr2   = perYR2[y]
                            const ybest = perYBest[y]
                            return (
                              <div key={y} className="bg-gray-800 rounded-lg px-4 py-2.5 flex items-center justify-between">
                                <div>
                                  <span className="text-white text-sm font-medium">{y}</span>
                                  {ybest && (
                                    <span className="text-gray-200 text-xs ml-3">
                                      noise={ybest.noise_factor} · dim={ybest.hidden_dim} · ep={ybest.epochs}
                                    </span>
                                  )}
                                </div>
                                <span className={`text-sm font-bold ${
                                  yr2 != null && yr2 >= 0.5 ? 'text-orange-400' : 'text-red-400'
                                }`}>
                                  {yr2 != null ? `${(yr2 * 100).toFixed(1)}%` : '—'}
                                </span>
                              </div>
                            )
                          })}
                        </div>
                        <p className="text-gray-500 text-xs mt-3">
                          💡 Consider adding more X variables, collecting more data, or reviewing
                          if these outputs are truly predictable from the selected inputs.
                        </p>
                      </div>
                    )}

                    {/* All Y good */}
                    {poorY.length === 0 && tuningRun.status === 'done' && yCols.length > 0 && (
                      <div className="bg-green-900/20 border border-green-700 rounded-lg px-4 py-2.5 text-sm">
                        <span className="text-green-400 font-medium">✅ All output variables exceed the per-Y R² threshold.</span>
                      </div>
                    )}
                  </div>
                )
              })()}
            </div>
          )}
        </div>
      )}

      {/* ── Action buttons ── */}
      {selectedDataset && (
        <div className="flex items-center gap-4 mb-6">
          {/* Start Training — disabled while any run is active */}
          <button
            onClick={handleTrain}
            disabled={isActive}
            title={isActive ? 'A training run is already in progress. Stop it first or wait.' : ''}
            className="bg-green-600 hover:bg-green-500 disabled:opacity-40 disabled:cursor-not-allowed
                       text-white px-10 py-3 rounded-xl font-bold text-sm transition-colors"
          >
            Start Training
          </button>

          {/* Stop Training — visible only while active */}
          {isActive && run && (
            <button
              onClick={handleStop}
              disabled={stopping}
              className="bg-red-600 hover:bg-red-500 disabled:opacity-50 disabled:cursor-not-allowed
                         text-white px-8 py-3 rounded-xl font-bold text-sm transition-colors
                         flex items-center gap-2"
            >
              {stopping
                ? <><span className="animate-spin">⏳</span> Stopping...</>
                : '⏹ Stop Training'
              }
            </button>
          )}

          {/* Force Kill — always visible when any run is active */}
          {isActive && (
            <button
              onClick={handleKillAll}
              disabled={killing}
              title="Immediately marks all active runs as stopped in the database. Use if Stop button is unresponsive."
              className="bg-gray-700 hover:bg-gray-600 disabled:opacity-50 disabled:cursor-not-allowed
                         text-red-400 border border-red-700 px-5 py-3 rounded-xl text-sm font-medium
                         transition-colors flex items-center gap-2"
            >
              {killing ? 'Killing...' : '☠ Force Kill'}
            </button>
          )}

          {isActive && (
            <span className="text-blue-400 text-sm animate-pulse">
              Training in progress...
            </span>
          )}
        </div>
      )}

      {error && (
        <div className="bg-red-900/40 border border-red-700 rounded-lg px-4 py-3 mb-4">
          <p className="text-red-400 text-sm">{error}</p>
        </div>
      )}

      {/* ── Training status card ── */}
      {run && (
        <div className={`rounded-xl p-6 border ${STATUS_BG[run.status] ?? 'bg-gray-800 border-gray-700'}`}>
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-3">
              <h2 className="text-white font-semibold">Training Run #{run.id}</h2>
              <span className={`text-sm font-bold ${STATUS_COLOR[run.status] ?? 'text-gray-200'}`}>
                {run.status.toUpperCase()}
              </span>
            </div>
            {run.status === 'stopped' && (
              <span className="text-xs bg-orange-900 text-orange-300 border border-orange-700 px-3 py-1 rounded-full">
                Stopped early — partial model saved
              </span>
            )}
          </div>

          {/* Live progress bar (shown while training) */}
          {(run.status === 'training' || run.status === 'pending') && (
            <div className="mb-5">
              <div className="flex justify-between text-xs text-gray-200 mb-1">
                <span>Epoch {currentEpoch} / {totalEpochs}</span>
                <span>{progressPct}%</span>
              </div>
              <div className="w-full bg-gray-700 rounded-full h-3">
                <div
                  className="bg-blue-500 h-3 rounded-full transition-all duration-500"
                  style={{ width: `${progressPct}%` }}
                />
              </div>
              {run.metrics?.live_train_loss != null && (
                <div className="flex gap-6 mt-2 text-xs text-gray-200">
                  <span>Train Loss: <span className="text-blue-300">{run.metrics.live_train_loss}</span></span>
                  <span>Val Loss: <span className="text-green-300">{run.metrics.live_val_loss}</span></span>
                </div>
              )}
            </div>
          )}

          {/* Final metrics (shown when done or stopped) */}
          {['done', 'stopped'].includes(run.status) && run.metrics && (
            <>
              <div className="grid grid-cols-4 gap-4 mb-6">
                {[
                  { label: 'Epochs Completed',  value: run.metrics.epochs_completed ?? '—' },
                  { label: 'Train Rows',         value: run.metrics.n_train_rows?.toLocaleString() ?? '—' },
                  { label: 'Test Rows',          value: run.metrics.n_test_rows?.toLocaleString() ?? '—' },
                  { label: 'Overall R²',         value: run.metrics.r2_score?.toFixed(4) ?? '—' },
                ].map(m => (
                  <div key={m.label} className="bg-gray-700/60 rounded-lg p-4 text-center">
                    <p className="text-blue-400 font-bold text-xl">{m.value}</p>
                    <p className="text-gray-200 text-xs mt-1">{m.label}</p>
                  </div>
                ))}
              </div>

              {/* Per-Y R² and MAE table */}
              {run.metrics.per_y_r2 && Object.keys(run.metrics.per_y_r2).length > 0 && (() => {
                const perYR2  = run.metrics.per_y_r2  || {}
                const perYMAE = run.metrics.per_y_mae || {}
                const threshold = tuneCfg.per_y_threshold / 100
                return (
                  <div className="mb-6">
                    <div className="flex items-center justify-between mb-2">
                      <h3 className="text-gray-100 text-sm font-medium">Per-Output Performance</h3>
                      <button
                        onClick={() => {
                          setShowTunePanel(true)
                          const poor = Object.entries(perYR2)
                            .filter(([, v]) => v != null && v < threshold)
                            .map(([k]) => k)
                          setFocusYCols(poor.length > 0 ? poor : Object.keys(perYR2))
                        }}
                        className="text-xs bg-purple-700 hover:bg-purple-600 text-white px-3 py-1
                                   rounded-lg transition-colors font-medium"
                      >
                        🔧 Tune underperforming Y
                      </button>
                    </div>
                    <div className="overflow-auto rounded-lg border border-gray-700">
                      <table className="w-full text-sm border-collapse">
                        <thead className="bg-gray-700 text-gray-200 text-xs">
                          <tr>
                            <th className="px-4 py-2 text-left">Output Variable</th>
                            <th className="px-4 py-2 text-right">R²</th>
                            <th className="px-4 py-2 text-right">MAE</th>
                            <th className="px-4 py-2 text-left">Status</th>
                            <th className="px-4 py-2 text-center">Focus Tune</th>
                          </tr>
                        </thead>
                        <tbody>
                          {Object.entries(perYR2).map(([col, r2], idx) => {
                            const mae      = perYMAE[col]
                            const isPoor   = r2 != null && r2 < threshold
                            const isFocused = focusYCols.includes(col)
                            return (
                              <tr
                                key={col}
                                className={`border-t border-gray-700 ${idx % 2 === 0 ? 'bg-gray-800' : 'bg-gray-750'}`}
                              >
                                <td className="px-4 py-2 text-white font-medium">{col}</td>
                                <td className={`px-4 py-2 text-right font-bold ${
                                  isPoor ? 'text-orange-400' : 'text-green-400'
                                }`}>
                                  {r2 != null ? `${(r2 * 100).toFixed(2)}%` : '—'}
                                </td>
                                <td className="px-4 py-2 text-right text-gray-100">
                                  {mae != null ? mae.toFixed(4) : '—'}
                                </td>
                                <td className="px-4 py-2">
                                  {isPoor
                                    ? <span className="text-xs bg-orange-900/60 text-orange-300 px-2 py-0.5 rounded-full">Below {tuneCfg.per_y_threshold}%</span>
                                    : <span className="text-xs bg-green-900/60 text-green-300 px-2 py-0.5 rounded-full">Good</span>
                                  }
                                </td>
                                <td className="px-4 py-2 text-center">
                                  <button
                                    onClick={() => setFocusYCols(p =>
                                      p.includes(col) ? p.filter(c => c !== col) : [...p, col]
                                    )}
                                    className={`w-5 h-5 rounded border-2 transition-colors ${
                                      isFocused
                                        ? 'bg-purple-600 border-purple-500'
                                        : 'bg-gray-700 border-gray-500 hover:border-purple-500'
                                    }`}
                                    title={isFocused ? 'Remove from tuning focus' : 'Add to tuning focus'}
                                  >
                                    {isFocused && <span className="text-white text-xs leading-none">✓</span>}
                                  </button>
                                </td>
                              </tr>
                            )
                          })}
                        </tbody>
                      </table>
                    </div>
                    {focusYCols.length > 0 && (
                      <p className="text-purple-300 text-xs mt-2">
                        🎯 {focusYCols.length} variable{focusYCols.length > 1 ? 's' : ''} selected for focused tuning:
                        <span className="font-medium ml-1">{focusYCols.join(', ')}</span>
                      </p>
                    )}
                  </div>
                )
              })()}

              {chartData.length > 0 && (
                <>
                  <h3 className="text-gray-100 text-sm font-medium mb-3">Loss Curve</h3>
                  <ResponsiveContainer width="100%" height={250}>
                    <LineChart data={chartData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                      <XAxis dataKey="epoch" stroke="#6b7280" tick={{ fontSize: 11 }} />
                      <YAxis stroke="#6b7280" tick={{ fontSize: 11 }} />
                      <Tooltip contentStyle={{ backgroundColor: '#1f2937', border: 'none', color: '#e5e7eb' }} />
                      <Legend />
                      <Line type="monotone" dataKey="Train Loss" stroke="#60a5fa" dot={false} strokeWidth={2} />
                      <Line type="monotone" dataKey="Val Loss"   stroke="#34d399" dot={false} strokeWidth={2} />
                    </LineChart>
                  </ResponsiveContainer>
                </>
              )}
            </>
          )}

          {run.status === 'error' && (
            <p className="text-red-400 text-sm">{run.metrics?.error}</p>
          )}
        </div>
      )}
    </div>
  )
}
