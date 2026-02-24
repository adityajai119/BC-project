import { useState } from 'react'
import { Box, Typography, Grid, Stack, FormControl, InputLabel, Select, MenuItem, Slider, Button, Chip, type SelectChangeEvent } from '@mui/material'
import ScienceIcon from '@mui/icons-material/Science'
import axios from 'axios'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip as RTooltip, ResponsiveContainer, Cell } from 'recharts'
import { C, API, TT, MODEL_COLORS, type ModelInfo } from './shared'

type TrainResult = { status: string; modelKey: string; modelName: string; trainMetrics: { accuracy: number; f1: number }; testMetrics: { accuracy: number; f1: number; precision: number; recall: number }; confusionMatrix: number[][]; params: Record<string, any> }

const PARAM_CONFIG: Record<string, { label: string; params: { key: string; label: string; min: number; max: number; step: number; default: number }[] }> = {
    rf: {
        label: 'Random Forest', params: [
            { key: 'n_estimators', label: 'Trees', min: 10, max: 500, step: 10, default: 100 },
            { key: 'max_depth', label: 'Max Depth', min: 1, max: 30, step: 1, default: 10 },
            { key: 'min_samples_split', label: 'Min Samples Split', min: 2, max: 20, step: 1, default: 2 },
        ]
    },
    gb: {
        label: 'Gradient Boosting', params: [
            { key: 'n_estimators', label: 'Estimators', min: 10, max: 500, step: 10, default: 100 },
            { key: 'max_depth', label: 'Max Depth', min: 1, max: 15, step: 1, default: 3 },
            { key: 'learning_rate', label: 'Learning Rate', min: 0.01, max: 1, step: 0.01, default: 0.1 },
        ]
    },
    lr: {
        label: 'Logistic Regression', params: [
            { key: 'C', label: 'Regularization (C)', min: 0.01, max: 100, step: 0.1, default: 1.0 },
            { key: 'max_iter', label: 'Max Iterations', min: 100, max: 10000, step: 100, default: 5000 },
        ]
    },
}

export function TrainingPage({ models, addToast }: { models: ModelInfo[]; addToast: (m: string, s?: any) => void }) {
    const [modelKey, setModelKey] = useState('rf')
    const [params, setParams] = useState<Record<string, number>>({})
    const [loading, setLoading] = useState(false)
    const [result, setResult] = useState<TrainResult | null>(null)

    const cfg = PARAM_CONFIG[modelKey]
    const getParam = (key: string) => params[key] ?? cfg?.params.find(p => p.key === key)?.default ?? 0

    const handleRetrain = async () => {
        setLoading(true); setResult(null)
        try {
            const p: Record<string, any> = {}
            cfg?.params.forEach(param => { p[param.key] = getParam(param.key) })
            const res = await axios.post(`${API}/api/retrain`, { model: modelKey, params: p })
            setResult(res.data)
            addToast(`Retrained ${res.data.modelName} — Test accuracy: ${(res.data.testMetrics.accuracy * 100).toFixed(1)}%`, 'success')
        } catch (e: any) { addToast(e?.response?.data?.error || 'Retrain failed', 'error') } finally { setLoading(false) }
    }

    const metricBars = result ? [
        { name: 'Accuracy', train: +(result.trainMetrics.accuracy * 100).toFixed(1), test: +(result.testMetrics.accuracy * 100).toFixed(1) },
        { name: 'F1 Score', train: +(result.trainMetrics.f1 * 100).toFixed(1), test: +(result.testMetrics.f1 * 100).toFixed(1) },
        { name: 'Precision', train: 0, test: +(result.testMetrics.precision * 100).toFixed(1) },
        { name: 'Recall', train: 0, test: +(result.testMetrics.recall * 100).toFixed(1) },
    ] : []

    return (
        <Box className="animate-fade-in-up">
            <Typography variant="h5" gutterBottom sx={{ color: 'text.primary', fontWeight: 700 }}>Model Training</Typography>
            <Typography variant="body2" sx={{ color: 'text.secondary', mb: 3 }}>Retrain models with custom hyperparameters. Results are in-memory only (original models preserved).</Typography>
            <Grid container spacing={3}>
                <Grid size={{ xs: 12, md: 5 }}>
                    <Box className="glass-card" sx={{ p: 3 }}>
                        <Stack direction="row" spacing={2} alignItems="center" mb={3}>
                            <Typography variant="subtitle1" sx={{ color: 'text.primary', flex: 1 }}>Hyperparameters</Typography>
                            <FormControl size="small" sx={{ minWidth: 180 }}><InputLabel>Model</InputLabel><Select value={modelKey} label="Model" onChange={(e: SelectChangeEvent) => { setModelKey(e.target.value); setParams({}); setResult(null) }}>{models.map(m => <MenuItem key={m.key} value={m.key}><Stack direction="row" spacing={1} alignItems="center"><Box sx={{ width: 8, height: 8, borderRadius: '50%', bgcolor: MODEL_COLORS[m.key] || C.teal }} /><span>{m.name}</span></Stack></MenuItem>)}</Select></FormControl>
                        </Stack>
                        {cfg?.params.map(p => (
                            <Box key={p.key} sx={{ mb: 3 }}>
                                <Stack direction="row" justifyContent="space-between" alignItems="center" mb={0.5}>
                                    <Typography variant="caption" sx={{ color: 'text.secondary' }}>{p.label}</Typography>
                                    <Chip label={getParam(p.key).toFixed(p.step < 1 ? 2 : 0)} size="small" sx={{ bgcolor: `${MODEL_COLORS[modelKey]}20`, color: MODEL_COLORS[modelKey], fontWeight: 700 }} />
                                </Stack>
                                <Slider value={getParam(p.key)} min={p.min} max={p.max} step={p.step} onChange={(_, v) => setParams(prev => ({ ...prev, [p.key]: v as number }))}
                                    sx={{ color: MODEL_COLORS[modelKey] || C.teal, '& .MuiSlider-thumb': { width: 14, height: 14 } }} />
                            </Box>
                        ))}
                        <Button variant="contained" fullWidth size="large" onClick={handleRetrain} disabled={loading} sx={{ py: 1.5, mt: 1 }}>{loading ? 'Training...' : 'Retrain Model'}</Button>
                    </Box>
                </Grid>
                <Grid size={{ xs: 12, md: 7 }}>
                    {result ? (
                        <Box className="glass-card" sx={{ p: 3 }}>
                            <Stack direction="row" spacing={1} alignItems="center" mb={2}>
                                <ScienceIcon sx={{ color: MODEL_COLORS[modelKey], fontSize: 24 }} />
                                <Typography variant="subtitle1" sx={{ color: 'text.primary' }}>Training Results — {result.modelName}</Typography>
                            </Stack>
                            <ResponsiveContainer width="100%" height={220}>
                                <BarChart data={metricBars}><CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" /><XAxis dataKey="name" tick={{ fill: '#94a3b8', fontSize: 11 }} axisLine={false} /><YAxis domain={[80, 105]} tick={{ fill: '#64748b', fontSize: 11 }} axisLine={false} /><RTooltip {...TT} />
                                    <Bar dataKey="train" name="Train" fill={C.teal} radius={[4, 4, 0, 0]} opacity={0.6} /><Bar dataKey="test" name="Test" fill={MODEL_COLORS[modelKey] || C.amber} radius={[4, 4, 0, 0]} /></BarChart>
                            </ResponsiveContainer>
                            <Grid container spacing={2} sx={{ mt: 2 }}>{Object.entries(result.testMetrics).map(([k, v]) => (
                                <Grid size={{ xs: 6, sm: 3 }} key={k}><Box sx={{ textAlign: 'center', p: 1.5, borderRadius: '12px', background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.06)' }}>
                                    <Typography variant="caption" sx={{ color: 'text.secondary', textTransform: 'capitalize' }}>{k}</Typography>
                                    <Typography variant="h5" sx={{ color: MODEL_COLORS[modelKey], fontWeight: 800 }}>{(v * 100).toFixed(1)}%</Typography>
                                </Box></Grid>
                            ))}</Grid>
                            <Box sx={{ mt: 3 }}>
                                <Typography variant="subtitle1" sx={{ color: 'text.primary', mb: 1 }}>Confusion Matrix</Typography>
                                <table className="heatmap-table"><thead><tr><th></th><th>Pred B</th><th>Pred M</th></tr></thead><tbody>
                                    <tr><th>Act B</th><td className="heatmap-cell-high">{result.confusionMatrix[0]?.[0]}</td><td className="heatmap-cell-low">{result.confusionMatrix[0]?.[1]}</td></tr>
                                    <tr><th>Act M</th><td className="heatmap-cell-low">{result.confusionMatrix[1]?.[0]}</td><td className="heatmap-cell-high">{result.confusionMatrix[1]?.[1]}</td></tr>
                                </tbody></table>
                            </Box>
                        </Box>
                    ) : (
                        <Box className="glass-card" sx={{ p: 6, textAlign: 'center' }}>
                            <ScienceIcon sx={{ fontSize: 64, color: 'text.secondary', opacity: 0.2, mb: 2 }} />
                            <Typography variant="h6" sx={{ color: 'text.secondary', mb: 1 }}>Ready to Train</Typography>
                            <Typography variant="body2" sx={{ color: 'text.secondary' }}>Adjust hyperparameters and click "Retrain Model" to see results.</Typography>
                        </Box>
                    )}
                </Grid>
            </Grid>
        </Box>
    )
}
