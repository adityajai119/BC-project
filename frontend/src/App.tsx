import { useEffect, useMemo, useState, useCallback } from 'react'
import { Box, Container, LinearProgress, Typography, Alert, Snackbar, Chip, Stack, IconButton, Tooltip, Skeleton } from '@mui/material'
import { ThemeProvider, createTheme } from '@mui/material/styles'
import CssBaseline from '@mui/material/CssBaseline'
import ScienceIcon from '@mui/icons-material/Science'
import LightModeIcon from '@mui/icons-material/LightMode'
import DarkModeIcon from '@mui/icons-material/DarkMode'
import HelpOutlineIcon from '@mui/icons-material/HelpOutline'
import FitnessCenterIcon from '@mui/icons-material/FitnessCenter'
import { motion, AnimatePresence } from 'framer-motion'
import Joyride, { type Step, type CallBackProps } from 'react-joyride'
import axios from 'axios'
import { API, NAV as BASE_NAV, MODEL_COLORS, C, type PageKey, type FeatureMeta, type MetricsData, type DatasetStatsResponse, type FeatureImportanceResponse, type ModelInfo, type HistoryEntry } from './shared'
import { OverviewPage } from './OverviewPage'
import { SinglePage } from './SinglePage'
import { BatchPage } from './BatchPage'
import { ComparePage } from './ComparePage'
import { InsightsPage } from './InsightsPage'
import { ExplorerPage } from './ExplorerPage'
import { WhatIfPage } from './WhatIfPage'
import { TrainingPage } from './TrainingPage'
import { Chatbot } from './Chatbot'
import './App.css'

const NAV = [...BASE_NAV, { key: 'training' as PageKey, label: 'Training', icon: <FitnessCenterIcon fontSize="small" /> }]

const mkTheme = (mode: 'dark' | 'light') => createTheme({
  palette: {
    mode,
    primary: { main: '#00d4aa' },
    secondary: { main: '#7c3aed' },
    ...(mode === 'dark' ? { background: { default: '#060918', paper: 'rgba(15,20,40,0.8)' }, text: { primary: '#e2e8f0', secondary: '#94a3b8' } }
      : { background: { default: '#f0f4f8', paper: '#ffffff' }, text: { primary: '#1e293b', secondary: '#64748b' } }),
  },
  typography: { fontFamily: '"Inter", system-ui, sans-serif', h3: { fontWeight: 800 }, h4: { fontWeight: 700 }, h5: { fontWeight: 600 }, h6: { fontWeight: 600 }, subtitle1: { fontWeight: 600 } },
  shape: { borderRadius: 16 },
  components: {
    MuiPaper: { styleOverrides: { root: { backgroundImage: 'none' } } },
    MuiButton: { styleOverrides: { contained: { background: 'linear-gradient(135deg, #00d4aa, #00b4d8)', color: '#060918', fontWeight: 700, textTransform: 'none' as const, borderRadius: '12px', boxShadow: '0 4px 20px rgba(0,212,170,0.3)' }, outlined: { borderColor: 'rgba(0,212,170,0.3)', color: '#00d4aa', textTransform: 'none' as const, borderRadius: '12px' }, text: { textTransform: 'none' as const } } },
    MuiTextField: { styleOverrides: { root: { '& .MuiOutlinedInput-root': { borderRadius: '12px' } } } },
    MuiLinearProgress: { styleOverrides: { root: { borderRadius: 999, height: 8 }, bar: { borderRadius: 999, background: 'linear-gradient(90deg, #00d4aa, #00b4d8)' } } },
  },
})

const TOUR_STEPS: Step[] = [
  { target: '.sidebar-nav', content: 'Navigate between pages using the sidebar. Each page offers a unique perspective on the data.', placement: 'right', disableBeacon: true },
  { target: '#nav-single', content: 'Make predictions with any model. Try the "All Models" button for consensus from all 3 models simultaneously.', placement: 'right' },
  { target: '#nav-compare', content: 'Compare all models with ROC curves, cross-validation scores, and confusion matrices side by side.', placement: 'right' },
  { target: '#nav-explorer', content: 'Explore the dataset with scatter plots, histograms, PCA visualization, and a correlation heatmap.', placement: 'right' },
  { target: '#nav-whatif', content: 'What-If Analysis: drag sliders and watch predictions change in real-time with SHAP explanations.', placement: 'right' },
  { target: '#nav-training', content: 'Retrain models with custom hyperparameters and compare training vs test performance.', placement: 'right' },
  { target: '#theme-toggle', content: 'Switch between dark and light mode. Your preference is remembered.', placement: 'top' },
]

type Toast = { id: number; message: string; severity: 'success' | 'info' | 'warning' | 'error' }

function App() {
  const [mode, setMode] = useState<'dark' | 'light'>(() => (localStorage.getItem('theme') as 'dark' | 'light') || 'dark')
  const [page, setPage] = useState<PageKey>('overview')
  const [features, setFeatures] = useState<FeatureMeta[]>([])
  const [loadingMeta, setLoadingMeta] = useState(true)
  const [metrics, setMetrics] = useState<MetricsData | null>(null)
  const [loadingMetrics, setLoadingMetrics] = useState(true)
  const [datasetStats, setDatasetStats] = useState<DatasetStatsResponse | null>(null)
  const [featureImps, setFeatureImps] = useState<FeatureImportanceResponse | null>(null)
  const [availableModels, setAvailableModels] = useState<ModelInfo[]>([])
  const [compareData, setCompareData] = useState<MetricsData[]>([])
  const [history, setHistory] = useState<HistoryEntry[]>([])
  const [toasts, setToasts] = useState<Toast[]>([])
  const [runTour, setRunTour] = useState(false)

  const theme = useMemo(() => mkTheme(mode), [mode])
  const toggleMode = () => { const n = mode === 'dark' ? 'light' : 'dark'; setMode(n); localStorage.setItem('theme', n) }
  const isDark = mode === 'dark'

  const addToast = useCallback((message: string, severity: Toast['severity'] = 'info') => {
    setToasts(prev => [...prev, { id: Date.now(), message, severity }])
  }, [])
  const removeToast = (id: number) => setToasts(prev => prev.filter(t => t.id !== id))

  useEffect(() => { if (!localStorage.getItem('tourDone2')) { setTimeout(() => setRunTour(true), 1500) } }, [])
  const handleTourCallback = (data: CallBackProps) => { if (data.status === 'finished' || data.status === 'skipped') { setRunTour(false); localStorage.setItem('tourDone2', '1') } }

  const addHistory = useCallback((entry: Omit<HistoryEntry, 'id' | 'time'>) => {
    setHistory(prev => [{ ...entry, id: Date.now(), time: new Date().toLocaleTimeString() }, ...prev].slice(0, 20))
    addToast(`Prediction: ${entry.label} (${entry.confidence}% confidence)`, entry.label === 'Malignant' ? 'warning' : 'success')
  }, [addToast])

  useEffect(() => {
    const go = async () => {
      setLoadingMeta(true); setLoadingMetrics(true)
      const [m, met, st, imp, mod, cmp] = await Promise.allSettled([
        axios.get(`${API}/api/feature-metadata`), axios.get(`${API}/api/metrics`),
        axios.get(`${API}/api/dataset-stats`), axios.get(`${API}/api/feature-importance`),
        axios.get(`${API}/api/models`), axios.get(`${API}/api/compare-models`),
      ])
      if (m.status === 'fulfilled') setFeatures(m.value.data.features)
      if (met.status === 'fulfilled') setMetrics(met.value.data)
      if (st.status === 'fulfilled') setDatasetStats(st.value.data)
      if (imp.status === 'fulfilled') setFeatureImps(imp.value.data)
      if (mod.status === 'fulfilled') setAvailableModels(mod.value.data.models)
      if (cmp.status === 'fulfilled') setCompareData(cmp.value.data.models)
      setLoadingMeta(false); setLoadingMetrics(false)
    }; go()
  }, [])

  const pageVariants = { initial: { opacity: 0, y: 20 }, animate: { opacity: 1, y: 0, transition: { duration: 0.35, ease: 'easeOut' } }, exit: { opacity: 0, y: -15, transition: { duration: 0.2 } } }
  const borderColor = isDark ? 'rgba(255,255,255,0.06)' : 'rgba(0,0,0,0.08)'
  const sidebarBg = isDark ? 'rgba(10,14,30,0.95)' : 'rgba(255,255,255,0.97)'

  const content = useMemo(() => {
    switch (page) {
      case 'single': return <SinglePage features={features} loadingMeta={loadingMeta} onError={(e) => addToast(e || 'Error', 'error')} models={availableModels} addHistory={addHistory} addToast={addToast} />
      case 'batch': return <BatchPage onError={(e) => addToast(e || 'Error', 'error')} models={availableModels} addToast={addToast} />
      case 'compare': return <ComparePage data={compareData} loading={loadingMetrics} />
      case 'insights': return <InsightsPage metrics={metrics} loading={loadingMetrics} featureImps={featureImps} models={availableModels} />
      case 'explorer': return <ExplorerPage features={features} />
      case 'whatif': return <WhatIfPage features={features} models={availableModels} />
      case 'training' as PageKey: return <TrainingPage models={availableModels} addToast={addToast} />
      default: return <OverviewPage metrics={metrics} loadingMetrics={loadingMetrics} datasetStats={datasetStats} featureImps={featureImps} history={history} />
    }
  }, [page, features, loadingMeta, metrics, loadingMetrics, datasetStats, featureImps, availableModels, compareData, history, addHistory, addToast])

  // Skeleton for loading state
  const LoadingSkeleton = () => (
    <Box sx={{ p: 3 }}>
      <Skeleton variant="text" width="40%" height={40} sx={{ mb: 2, bgcolor: isDark ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.06)' }} />
      <Stack direction="row" spacing={2} sx={{ mb: 3 }}>{[1, 2, 3, 4].map(i => <Skeleton key={i} variant="rounded" width="25%" height={100} sx={{ borderRadius: '16px', bgcolor: isDark ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.06)' }} />)}</Stack>
      <Stack direction="row" spacing={2}><Skeleton variant="rounded" width="40%" height={240} sx={{ borderRadius: '20px', bgcolor: isDark ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.06)' }} /><Skeleton variant="rounded" width="60%" height={240} sx={{ borderRadius: '20px', bgcolor: isDark ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.06)' }} /></Stack>
    </Box>
  )

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Joyride steps={TOUR_STEPS} run={runTour} continuous showSkipButton showProgress callback={handleTourCallback}
        styles={{
          options: { primaryColor: '#00d4aa', backgroundColor: isDark ? '#0f1428' : '#fff', textColor: isDark ? '#e2e8f0' : '#1e293b', arrowColor: isDark ? '#0f1428' : '#fff', overlayColor: 'rgba(0,0,0,0.75)' },
          tooltip: { borderRadius: 16, padding: 20 }, buttonNext: { borderRadius: 10, fontWeight: 700 }, buttonSkip: { color: '#94a3b8' }
        }} />
      <Box sx={{ minHeight: '100vh', background: isDark ? '#060918' : '#f0f4f8', display: 'flex' }}>
        {/* Sidebar */}
        <Box className="sidebar-nav" sx={{
          width: { xs: '100%', md: 240 }, position: { xs: 'fixed', md: 'sticky' },
          bottom: { xs: 0, md: 'auto' }, top: { md: 0 }, height: { md: '100vh' }, zIndex: 100,
          display: 'flex', flexDirection: { xs: 'row', md: 'column' },
          background: sidebarBg, backdropFilter: 'blur(20px)',
          borderRight: { md: `1px solid ${borderColor}` }, borderTop: { xs: `1px solid ${borderColor}`, md: 'none' },
          p: { xs: 1, md: 3 }, gap: { xs: 0.5, md: 0.5 },
          boxShadow: isDark ? 'none' : '4px 0 20px rgba(0,0,0,0.04)',
        }}>
          <Box sx={{ display: { xs: 'none', md: 'flex' }, alignItems: 'center', gap: 1.5, mb: 3, px: 1 }}>
            <Box sx={{ width: 36, height: 36, borderRadius: '10px', background: 'linear-gradient(135deg, #00d4aa, #00b4d8)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <ScienceIcon sx={{ color: '#060918', fontSize: 20 }} />
            </Box>
            <Box><Typography variant="subtitle1" sx={{ fontWeight: 800, color: 'text.primary', lineHeight: 1.2, fontSize: '0.75rem' }}>Breast Cancer AI</Typography><Typography variant="caption" sx={{ color: 'text.secondary', fontSize: '0.55rem' }}>Multi-Model Studio</Typography></Box>
          </Box>
          {NAV.map(n => <button key={n.key} id={`nav-${n.key}`} className={`nav-tab ${page === n.key ? 'active' : ''} ${isDark ? '' : 'light'}`} onClick={() => setPage(n.key as PageKey)}>{n.icon}<span>{n.label}</span></button>)}
          <Box sx={{ mt: 'auto', display: { xs: 'none', md: 'flex' }, flexDirection: 'column', gap: 1, px: 1, pt: 2 }}>
            <Stack direction="row" spacing={1} alignItems="center"><span className="glow-dot success" /><Typography variant="caption" sx={{ color: 'text.secondary' }}>{availableModels.length} models</Typography></Stack>
            <Stack direction="row" spacing={0.5}>
              <Tooltip title="Toggle theme"><IconButton id="theme-toggle" size="small" onClick={toggleMode} sx={{ color: 'text.secondary' }}>{isDark ? <LightModeIcon fontSize="small" /> : <DarkModeIcon fontSize="small" />}</IconButton></Tooltip>
              <Tooltip title="Tour"><IconButton size="small" onClick={() => setRunTour(true)} sx={{ color: 'text.secondary' }}><HelpOutlineIcon fontSize="small" /></IconButton></Tooltip>
            </Stack>
          </Box>
        </Box>
        {/* Main */}
        <Box sx={{ flex: 1, pb: { xs: 10, md: 0 }, overflow: 'hidden' }}>
          <Box sx={{ px: { xs: 2, md: 4 }, py: 2, borderBottom: `1px solid ${borderColor}`, display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <Typography variant="h6" sx={{ fontWeight: 700, color: 'text.primary' }}>{NAV.find(n => n.key === page)?.label}</Typography>
            <Stack direction="row" spacing={1}>
              {availableModels.map(m => <Chip key={m.key} size="small" label={m.name} sx={{ bgcolor: `${MODEL_COLORS[m.key] || C.teal}15`, color: MODEL_COLORS[m.key] || C.teal, border: `1px solid ${MODEL_COLORS[m.key] || C.teal}33`, fontWeight: 600, fontSize: '0.6rem' }} />)}
              <IconButton size="small" onClick={toggleMode} sx={{ display: { md: 'none' }, color: 'text.secondary' }}>{isDark ? <LightModeIcon fontSize="small" /> : <DarkModeIcon fontSize="small" />}</IconButton>
            </Stack>
          </Box>
          <Container maxWidth="xl" sx={{ py: 3 }}>
            <AnimatePresence mode="wait">
              <motion.div key={page} variants={pageVariants} initial="initial" animate="animate" exit="exit">
                {loadingMeta && page === 'overview' ? <LoadingSkeleton /> : content}
              </motion.div>
            </AnimatePresence>
          </Container>
        </Box>
      </Box>
      {/* Toast System */}
      {toasts.map((t, i) => (
        <Snackbar key={t.id} open anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }} autoHideDuration={4000} onClose={() => removeToast(t.id)}
          sx={{ bottom: `${(i * 60) + 24}px !important` }}>
          <Alert severity={t.severity} onClose={() => removeToast(t.id)} variant="filled" sx={{ width: '100%', borderRadius: '12px', fontWeight: 600, boxShadow: '0 8px 32px rgba(0,0,0,0.3)' }}>{t.message}</Alert>
        </Snackbar>
      ))}
      <Chatbot />
    </ThemeProvider>
  )
}

export default App
