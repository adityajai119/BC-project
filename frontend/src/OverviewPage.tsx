import { Box, Typography, Grid, LinearProgress, Chip, Stack, Button } from '@mui/material'
import HistoryIcon from '@mui/icons-material/History'
import DownloadIcon from '@mui/icons-material/Download'
import { PieChart, Pie, Cell, ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip as RTooltip } from 'recharts'
import { C, PIE_C, TT, type MetricsData, type DatasetStatsResponse, type FeatureImportanceResponse, type HistoryEntry } from './shared'

export function OverviewPage({ metrics, loadingMetrics, datasetStats, featureImps, history }: {
    metrics: MetricsData | null; loadingMetrics: boolean; datasetStats: DatasetStatsResponse | null
    featureImps: FeatureImportanceResponse | null; history: HistoryEntry[]
}) {
    const pie = datasetStats ? [{ name: 'Benign', value: datasetStats.classDistribution.benign }, { name: 'Malignant', value: datasetStats.classDistribution.malignant }] : []
    const bars = featureImps ? featureImps.importances.slice(0, 8).map(f => ({ name: f.feature.length > 14 ? f.feature.slice(0, 12) + '...' : f.feature, value: +(f.importance * 100).toFixed(2) })).reverse() : []

    const exportHistory = () => {
        if (!history.length) return
        const csv = 'Time,Model,Label,Confidence\n' + history.map(h => `${h.time},${h.model},${h.label},${h.confidence}%`).join('\n')
        const blob = new Blob([csv], { type: 'text/csv' }); const a = document.createElement('a')
        a.href = URL.createObjectURL(blob); a.download = 'prediction_history.csv'; a.click()
    }

    return (
        <Box>
            <Box className="animate-fade-in-up" sx={{ mb: 4 }}>
                <Typography variant="h3" sx={{ fontWeight: 900, mb: 1 }}><span className="gradient-text">Breast Cancer</span><br />AI Studio</Typography>
                <Typography variant="body1" sx={{ color: '#64748b', maxWidth: 600, lineHeight: 1.7 }}>Multi-model ML platform with What-If analysis, Data Explorer, and model comparison.</Typography>
            </Box>
            {loadingMetrics && <LinearProgress sx={{ mb: 3 }} />}
            <Grid container spacing={2.5} sx={{ mb: 4 }}>
                {[{ l: 'Accuracy', v: metrics?.accuracy, c: C.teal }, { l: 'Precision', v: metrics?.precision, c: C.cyan }, { l: 'Recall', v: metrics?.recall, c: C.violet }, { l: 'F1 Score', v: metrics?.f1, c: C.amber }].map((s, i) => (
                    <Grid size={{ xs: 6, md: 3 }} key={s.l}><Box className={`stat-card animate-fade-in-up stagger-${i + 1}`}>
                        <Typography variant="caption" sx={{ color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.08em', fontSize: '0.65rem' }}>{s.l}</Typography>
                        <Typography variant="h4" sx={{ fontWeight: 800, color: s.c, mt: 0.5 }}>{s.v ? `${(s.v * 100).toFixed(1)}%` : '...'}</Typography>
                    </Box></Grid>
                ))}
            </Grid>
            <Grid container spacing={3}>
                <Grid size={{ xs: 12, md: 5 }}>
                    <Box className="glass-card animate-fade-in-up stagger-3" sx={{ p: 3 }}>
                        <Typography variant="subtitle1" sx={{ color: 'text.primary', mb: 2 }}>Class Distribution</Typography>
                        {pie.length > 0 ? (
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 3 }}>
                                <ResponsiveContainer width="50%" height={200}><PieChart><Pie data={pie} cx="50%" cy="50%" outerRadius={80} innerRadius={50} dataKey="value" stroke="none">{pie.map((_, i) => <Cell key={i} fill={PIE_C[i]} />)}</Pie><RTooltip {...TT} /></PieChart></ResponsiveContainer>
                                <Box>{pie.map((d, i) => <Box key={d.name} sx={{ display: 'flex', alignItems: 'center', gap: 1.5, mb: 1.5 }}><Box sx={{ width: 12, height: 12, borderRadius: '4px', bgcolor: PIE_C[i] }} /><Box><Typography variant="body2" sx={{ color: '#94a3b8', fontSize: '0.75rem' }}>{d.name}</Typography><Typography variant="subtitle1" sx={{ color: 'text.primary', fontWeight: 700 }}>{d.value}</Typography></Box></Box>)}</Box>
                            </Box>
                        ) : <Typography variant="body2" sx={{ color: '#64748b' }}>Loading...</Typography>}
                    </Box>
                </Grid>
                <Grid size={{ xs: 12, md: 7 }}>
                    <Box className="glass-card animate-fade-in-up stagger-4" sx={{ p: 3 }}>
                        <Typography variant="subtitle1" sx={{ color: 'text.primary', mb: 2 }}>Top Feature Importances</Typography>
                        {bars.length > 0 ? (
                            <ResponsiveContainer width="100%" height={220}>
                                <BarChart data={bars} layout="vertical" margin={{ left: 10, right: 20 }}>
                                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" /><XAxis type="number" tick={{ fill: '#64748b', fontSize: 11 }} axisLine={false} /><YAxis type="category" dataKey="name" tick={{ fill: '#94a3b8', fontSize: 11 }} width={110} axisLine={false} />
                                    <RTooltip {...TT} formatter={(v: number) => [`${v.toFixed(2)}%`, 'Importance']} /><Bar dataKey="value" radius={[0, 6, 6, 0]} fill="url(#bG)" />
                                    <defs><linearGradient id="bG" x1="0" y1="0" x2="1" y2="0"><stop offset="0%" stopColor={C.teal} /><stop offset="100%" stopColor={C.cyan} /></linearGradient></defs>
                                </BarChart>
                            </ResponsiveContainer>
                        ) : <Typography variant="body2" sx={{ color: '#64748b' }}>Loading...</Typography>}
                    </Box>
                </Grid>
            </Grid>
            {history.length > 0 && (
                <Box className="glass-card animate-fade-in-up" sx={{ p: 3, mt: 3 }}>
                    <Stack direction="row" spacing={1} alignItems="center" mb={2}>
                        <HistoryIcon sx={{ color: C.teal, fontSize: 20 }} /><Typography variant="subtitle1" sx={{ color: 'text.primary', flex: 1 }}>Recent Predictions</Typography>
                        <Button size="small" variant="outlined" startIcon={<DownloadIcon />} onClick={exportHistory} sx={{ fontSize: '0.7rem' }}>Export CSV</Button>
                    </Stack>
                    <Stack spacing={1}>{history.map(h => (
                        <Box key={h.id} sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', p: 1.5, borderRadius: '12px', background: h.label === 'Malignant' ? 'rgba(248,113,113,0.06)' : 'rgba(0,212,170,0.04)', border: `1px solid ${h.label === 'Malignant' ? 'rgba(248,113,113,0.1)' : 'rgba(0,212,170,0.08)'}` }}>
                            <Stack direction="row" spacing={2} alignItems="center"><Typography variant="caption" sx={{ color: '#64748b' }}>{h.time}</Typography><Chip label={h.model} size="small" sx={{ bgcolor: 'rgba(255,255,255,0.05)', color: '#94a3b8', fontSize: '0.65rem' }} /></Stack>
                            <Stack direction="row" spacing={1} alignItems="center"><Typography variant="body2" sx={{ color: '#94a3b8' }}>{h.confidence}%</Typography><Chip label={h.label} size="small" sx={{ bgcolor: h.label === 'Malignant' ? 'rgba(248,113,113,0.15)' : 'rgba(0,212,170,0.15)', color: h.label === 'Malignant' ? C.rose : C.teal, fontWeight: 700 }} /></Stack>
                        </Box>
                    ))}</Stack>
                </Box>
            )}
        </Box>
    )
}
