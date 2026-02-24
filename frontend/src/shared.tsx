// Shared types and constants for the app
import React from 'react'
import DashboardIcon from '@mui/icons-material/Dashboard'
import PersonSearchIcon from '@mui/icons-material/PersonSearch'
import UploadFileIcon from '@mui/icons-material/UploadFile'
import CompareArrowsIcon from '@mui/icons-material/CompareArrows'
import InsightsIcon from '@mui/icons-material/Insights'
import ExploreIcon from '@mui/icons-material/Explore'
import TuneIcon from '@mui/icons-material/Tune'

export type FeatureMeta = { name: string; default: number; type: string }
export type MetricsData = { modelKey: string; modelName: string; accuracy: number; precision: number; recall: number; f1: number; confusionMatrix: number[][]; nSamples: number }
export type PredictionResponse = { prediction: number; label: string; probabilities: { benign: number; malignant: number }; modelKey: string; modelName: string; contributions?: { feature: string; contribution: number }[]; confidenceInterval?: { low: number; high: number } }
export type DatasetStatsResponse = { classDistribution: { benign: number; malignant: number }; totalSamples: number; totalFeatures: number; topCorrelations: { feature: string; correlation: number }[] }
export type FeatureImportanceResponse = { importances: { feature: string; importance: number }[]; modelName: string }
export type ModelInfo = { key: string; name: string }
export type HistoryEntry = { id: number; model: string; label: string; confidence: number; time: string }
export type ConsensusResult = { results: { modelKey: string; modelName: string; prediction: number; label: string; probabilities: { benign: number; malignant: number } }[]; consensus: string; agreement: number }
export type PageKey = 'overview' | 'single' | 'batch' | 'compare' | 'insights' | 'explorer' | 'whatif' | 'training'

export const API = import.meta.env.VITE_API_URL || 'http://localhost:5000'
export const C = { teal: '#00d4aa', cyan: '#00b4d8', violet: '#7c3aed', rose: '#f87171', amber: '#fbbf24', emerald: '#059669', blue: '#60a5fa', pink: '#f472b6' }
export const PIE_C = [C.teal, C.rose]
export const MODEL_COLORS: Record<string, string> = { rf: C.teal, gb: C.amber, lr: C.violet }
export const TT = { contentStyle: { background: '#111827', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 12, color: '#e2e8f0' } }

export const NAV: { key: PageKey; label: string; icon: React.ReactNode }[] = [
    { key: 'overview', label: 'Dashboard', icon: <DashboardIcon fontSize="small" /> },
    { key: 'single', label: 'Predict', icon: <PersonSearchIcon fontSize="small" /> },
    { key: 'batch', label: 'Batch', icon: <UploadFileIcon fontSize="small" /> },
    { key: 'compare', label: 'Compare', icon: <CompareArrowsIcon fontSize="small" /> },
    { key: 'insights', label: 'Insights', icon: <InsightsIcon fontSize="small" /> },
    { key: 'explorer', label: 'Explorer', icon: <ExploreIcon fontSize="small" /> },
    { key: 'whatif', label: 'What-If', icon: <TuneIcon fontSize="small" /> },
]

export const FGROUPS = [
    { title: 'Size & Shape (Mean)', keywords: ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean'] },
    { title: 'Smoothness & Compactness', keywords: ['smoothness', 'compactness', 'concavity', 'concave points'] },
    { title: 'Standard Error', keywords: ['_se'] },
    { title: 'Worst Case', keywords: ['_worst'] },
]
