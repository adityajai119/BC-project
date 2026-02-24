import { useEffect, useState } from 'react'
import { Box, Typography, Grid, LinearProgress, Stack, FormControl, InputLabel, Select, MenuItem, type SelectChangeEvent } from '@mui/material'
import axios from 'axios'
import { ResponsiveContainer, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, Tooltip as RTooltip, BarChart, Bar, XAxis, YAxis, CartesianGrid, LineChart, Line } from 'recharts'
import { C, API, TT, MODEL_COLORS, type MetricsData, type FeatureImportanceResponse, type ModelInfo } from './shared'

type LCPoint = { size: number; trainMean: number; valMean: number }

export function InsightsPage({ metrics, loading, featureImps, models }: { metrics: MetricsData | null; loading: boolean; featureImps: FeatureImportanceResponse | null; models: ModelInfo[] }) {
    const [selModel, setSelModel] = useState('rf')
    const [localMetrics, setLocalMetrics] = useState<MetricsData | null>(metrics)
    const [localImps, setLocalImps] = useState(featureImps)
    const [lcData, setLcData] = useState<LCPoint[]>([])

    useEffect(() => { setLocalMetrics(metrics); setLocalImps(featureImps) }, [metrics, featureImps])

    useEffect(() => {
        const go = async () => {
            const [mR, iR, lR] = await Promise.allSettled([
                axios.get(`${API}/api/metrics?model=${selModel}`),
                axios.get(`${API}/api/feature-importance?model=${selModel}`),
                axios.get(`${API}/api/learning-curve?model=${selModel}`),
            ])
            if (mR.status === 'fulfilled') setLocalMetrics(mR.value.data)
            if (iR.status === 'fulfilled') setLocalImps(iR.value.data)
            else setLocalImps(null)
            if (lR.status === 'fulfilled') setLcData(lR.value.data.points)
        }; go()
    }, [selModel])

    const radar = localMetrics ? ['accuracy', 'precision', 'recall', 'f1'].map(k => ({ metric: k === 'f1' ? 'F1' : k.charAt(0).toUpperCase() + k.slice(1), value: +((localMetrics as any)[k] * 100).toFixed(1), fullMark: 100 })) : []
    const imps = localImps ? localImps.importances.slice(0, 12).map(f => ({ name: f.feature.length > 16 ? f.feature.slice(0, 14) + '..' : f.feature, value: +(f.importance * 100).toFixed(2) })).reverse() : []
    const mc: Record<string, string> = { accuracy: C.teal, precision: C.cyan, recall: C.violet, f1: C.amber }

    return (
        <Box className="animate-fade-in-up">
            <Stack direction="row" spacing={2} alignItems="center" mb={3}>
                <Typography variant="h5" sx={{ color: 'text.primary', fontWeight: 700, flex: 1 }}>Model Insights</Typography>
                <FormControl size="small" sx={{ minWidth: 180 }}><InputLabel>Model</InputLabel><Select value={selModel} label="Model" onChange={(e: SelectChangeEvent) => setSelModel(e.target.value)}>{models.map(m => <MenuItem key={m.key} value={m.key}><Stack direction="row" spacing={1} alignItems="center"><Box sx={{ width: 8, height: 8, borderRadius: '50%', bgcolor: MODEL_COLORS[m.key] || C.teal }} /><span>{m.name}</span></Stack></MenuItem>)}</Select></FormControl>
            </Stack>
            {loading && <LinearProgress sx={{ mb: 3 }} />}
            {localMetrics && (<Grid container spacing={3}>
                <Grid size={{ xs: 12, md: 5 }}>
                    <Box className="glass-card" sx={{ p: 3 }}>
                        <Typography variant="subtitle1" sx={{ color: 'text.primary', mb: 2 }}>Performance Radar</Typography>
                        <ResponsiveContainer width="100%" height={240}><RadarChart data={radar}><PolarGrid stroke="rgba(255,255,255,0.06)" /><PolarAngleAxis dataKey="metric" tick={{ fill: '#94a3b8', fontSize: 12 }} /><PolarRadiusAxis angle={30} domain={[0, 100]} tick={{ fill: '#64748b', fontSize: 10 }} /><Radar dataKey="value" stroke={MODEL_COLORS[selModel] || C.teal} fill={MODEL_COLORS[selModel] || C.teal} fillOpacity={0.15} strokeWidth={2} /><RTooltip {...TT} /></RadarChart></ResponsiveContainer>
                        <Stack spacing={1.5} mt={2}>{(['accuracy', 'precision', 'recall', 'f1'] as const).map(k => (
                            <Box key={k}><Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}><Typography variant="caption" sx={{ color: '#94a3b8', textTransform: 'capitalize' }}>{k === 'f1' ? 'F1 Score' : k}</Typography><Typography variant="caption" sx={{ color: mc[k], fontWeight: 700 }}>{((localMetrics as any)[k] * 100).toFixed(1)}%</Typography></Box>
                                <LinearProgress variant="determinate" value={(localMetrics as any)[k] * 100} sx={{ height: 6, borderRadius: 999, bgcolor: 'rgba(255,255,255,0.04)', '& .MuiLinearProgress-bar': { background: `linear-gradient(90deg, ${mc[k]}, ${mc[k]}88)` } }} /></Box>
                        ))}</Stack>
                    </Box>
                </Grid>
                <Grid size={{ xs: 12, md: 7 }}>
                    <Box className="glass-card" sx={{ p: 3, mb: 3 }}>
                        <Typography variant="subtitle1" sx={{ color: 'text.primary', mb: 2 }}>Confusion Matrix</Typography>
                        <Box sx={{ display: 'flex', justifyContent: 'center' }}>
                            <table className="heatmap-table"><thead><tr><th></th><th>Pred Benign</th><th>Pred Malignant</th></tr></thead><tbody>
                                <tr><th>Act B</th><td className="heatmap-cell-high">{localMetrics.confusionMatrix[0]?.[0]}</td><td className="heatmap-cell-low">{localMetrics.confusionMatrix[0]?.[1]}</td></tr>
                                <tr><th>Act M</th><td className="heatmap-cell-low">{localMetrics.confusionMatrix[1]?.[0]}</td><td className="heatmap-cell-high">{localMetrics.confusionMatrix[1]?.[1]}</td></tr>
                            </tbody></table>
                        </Box>
                    </Box>
                    {imps.length > 0 && <Box className="glass-card" sx={{ p: 3, mb: 3 }}><Typography variant="subtitle1" sx={{ color: 'text.primary', mb: 2 }}>Feature Importances (Top 12)</Typography>
                        <ResponsiveContainer width="100%" height={300}><BarChart data={imps} layout="vertical" margin={{ left: 10, right: 20 }}><CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" /><XAxis type="number" tick={{ fill: '#64748b', fontSize: 11 }} axisLine={false} /><YAxis type="category" dataKey="name" tick={{ fill: '#94a3b8', fontSize: 10 }} width={120} axisLine={false} /><RTooltip {...TT} /><Bar dataKey="value" radius={[0, 6, 6, 0]} fill={MODEL_COLORS[selModel] || C.teal} /></BarChart></ResponsiveContainer>
                    </Box>}
                    {lcData.length > 0 && <Box className="glass-card" sx={{ p: 3 }}><Typography variant="subtitle1" sx={{ color: 'text.primary', mb: 2 }}>Learning Curves</Typography>
                        <ResponsiveContainer width="100%" height={220}><LineChart data={lcData}><CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" /><XAxis dataKey="size" tick={{ fill: '#64748b', fontSize: 10 }} label={{ value: 'Training Size', fill: '#64748b', fontSize: 11, position: 'bottom' }} /><YAxis tick={{ fill: '#64748b', fontSize: 10 }} domain={[0.8, 1.05]} /><RTooltip {...TT} /><Line type="monotone" dataKey="trainMean" name="Training" stroke={C.teal} strokeWidth={2} /><Line type="monotone" dataKey="valMean" name="Validation" stroke={C.amber} strokeWidth={2} /></LineChart></ResponsiveContainer>
                    </Box>}
                </Grid>
                <Grid size={12}><Box className="glass-card" sx={{ p: 3, display: 'flex', flexWrap: 'wrap', gap: 4 }}>
                    {[['Model', localMetrics.modelName], ['Dataset', `${localMetrics.nSamples} samples`], ['Framework', 'scikit-learn + Flask'], ['Frontend', 'React + Recharts']].map(([l, v]) => <Box key={l}><Typography variant="caption" sx={{ color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.1em' }}>{l}</Typography><Typography variant="subtitle1" sx={{ color: 'text.primary' }}>{v}</Typography></Box>)}
                </Box></Grid>
            </Grid>)}
        </Box>
    )
}
