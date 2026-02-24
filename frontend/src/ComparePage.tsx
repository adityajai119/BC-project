import { useEffect, useState } from 'react'
import { Box, Typography, Grid, LinearProgress, Stack } from '@mui/material'
import axios from 'axios'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip as RTooltip, Legend, ResponsiveContainer, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, LineChart, Line, Cell } from 'recharts'
import { C, API, TT, MODEL_COLORS, type MetricsData } from './shared'

type CVResult = { modelKey: string; modelName: string; scores: number[]; mean: number; std: number }
type ROCCurve = { modelKey: string; modelName: string; fpr: number[]; tpr: number[]; auc: number }

export function ComparePage({ data, loading }: { data: MetricsData[]; loading: boolean }) {
    const [cvData, setCvData] = useState<CVResult[]>([])
    const [rocData, setRocData] = useState<ROCCurve[]>([])
    const [loadingExtra, setLoadingExtra] = useState(false)

    useEffect(() => {
        const go = async () => {
            setLoadingExtra(true)
            const [cv, roc] = await Promise.allSettled([axios.get(`${API}/api/cross-validation`), axios.get(`${API}/api/roc-data`)])
            if (cv.status === 'fulfilled') setCvData(cv.value.data.results)
            if (roc.status === 'fulfilled') setRocData(roc.value.data.curves)
            setLoadingExtra(false)
        }; go()
    }, [])

    const barData = ['accuracy', 'precision', 'recall', 'f1'].map(m => {
        const row: any = { metric: m === 'f1' ? 'F1 Score' : m.charAt(0).toUpperCase() + m.slice(1) }
        data.forEach(d => { row[d.modelKey] = +((d as any)[m] * 100).toFixed(1) }); return row
    })
    const radarData = ['accuracy', 'precision', 'recall', 'f1'].map(m => {
        const row: any = { metric: m === 'f1' ? 'F1' : m.charAt(0).toUpperCase() + m.slice(1), fullMark: 100 }
        data.forEach(d => { row[d.modelKey] = +((d as any)[m] * 100).toFixed(1) }); return row
    })
    // ROC as line chart points
    const rocLines = rocData.map(r => ({ ...r, points: r.fpr.map((f, i) => ({ fpr: f, tpr: r.tpr[i] })) }))
    // CV box data
    const cvBarData = cvData.map(cv => ({ name: cv.modelName, mean: +(cv.mean * 100).toFixed(1), std: +(cv.std * 100).toFixed(2), color: MODEL_COLORS[cv.modelKey] || C.teal }))

    return (
        <Box className="animate-fade-in-up">
            <Typography variant="h5" gutterBottom sx={{ color: 'text.primary', fontWeight: 700 }}>Model Comparison</Typography>
            {(loading || loadingExtra) && <LinearProgress sx={{ mb: 3 }} />}
            {data.length > 0 && (<Grid container spacing={3}>
                {/* Metrics Table */}
                <Grid size={12}>
                    <Box className="glass-card" sx={{ p: 3, overflowX: 'auto' }}>
                        <Typography variant="subtitle1" sx={{ color: 'text.primary', mb: 2 }}>Performance Metrics</Typography>
                        <table style={{ width: '100%', borderCollapse: 'separate', borderSpacing: '0 8px' }}>
                            <thead><tr>{['Model', 'Accuracy', 'Precision', 'Recall', 'F1'].map(h => <th key={h} style={{ textAlign: 'left', padding: '8px 16px', color: '#64748b', fontSize: '0.75rem', fontWeight: 600, textTransform: 'uppercase' as const, border: 'none', background: 'transparent' }}>{h}</th>)}</tr></thead>
                            <tbody>{data.map(m => (<tr key={m.modelKey} style={{ background: 'rgba(255,255,255,0.02)' }}>
                                <td style={{ padding: '12px 16px', borderRadius: '12px 0 0 12px', border: 'none' }}><Stack direction="row" spacing={1} alignItems="center"><Box sx={{ width: 10, height: 10, borderRadius: '50%', bgcolor: MODEL_COLORS[m.modelKey] }} /><Typography variant="body2" sx={{ color: 'text.primary', fontWeight: 600 }}>{m.modelName}</Typography></Stack></td>
                                {['accuracy', 'precision', 'recall', 'f1'].map((k, i) => <td key={k} style={{ padding: '12px 16px', border: 'none', borderRadius: i === 3 ? '0 12px 12px 0' : undefined }}><Typography variant="body2" sx={{ color: MODEL_COLORS[m.modelKey], fontWeight: 700 }}>{((m as any)[k] * 100).toFixed(1)}%</Typography></td>)}
                            </tr>))}</tbody>
                        </table>
                    </Box>
                </Grid>
                {/* Bar + Radar */}
                <Grid size={{ xs: 12, md: 6 }}>
                    <Box className="glass-card" sx={{ p: 3 }}><Typography variant="subtitle1" sx={{ color: 'text.primary', mb: 2 }}>Metric Comparison</Typography>
                        <ResponsiveContainer width="100%" height={260}><BarChart data={barData}><CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" /><XAxis dataKey="metric" tick={{ fill: '#94a3b8', fontSize: 12 }} axisLine={false} /><YAxis domain={[80, 100]} tick={{ fill: '#64748b', fontSize: 11 }} axisLine={false} /><RTooltip {...TT} /><Legend wrapperStyle={{ color: '#94a3b8', fontSize: 12 }} />{data.map(m => <Bar key={m.modelKey} dataKey={m.modelKey} name={m.modelName} fill={MODEL_COLORS[m.modelKey]} radius={[4, 4, 0, 0]} />)}</BarChart></ResponsiveContainer>
                    </Box>
                </Grid>
                <Grid size={{ xs: 12, md: 6 }}>
                    <Box className="glass-card" sx={{ p: 3 }}><Typography variant="subtitle1" sx={{ color: 'text.primary', mb: 2 }}>Performance Radar</Typography>
                        <ResponsiveContainer width="100%" height={260}><RadarChart data={radarData}><PolarGrid stroke="rgba(255,255,255,0.06)" /><PolarAngleAxis dataKey="metric" tick={{ fill: '#94a3b8', fontSize: 12 }} /><PolarRadiusAxis angle={30} domain={[80, 100]} tick={{ fill: '#64748b', fontSize: 10 }} />{data.map(m => <Radar key={m.modelKey} name={m.modelName} dataKey={m.modelKey} stroke={MODEL_COLORS[m.modelKey]} fill={MODEL_COLORS[m.modelKey]} fillOpacity={0.1} strokeWidth={2} />)}<Legend wrapperStyle={{ color: '#94a3b8', fontSize: 12 }} /><RTooltip {...TT} /></RadarChart></ResponsiveContainer>
                    </Box>
                </Grid>
                {/* ROC Curves */}
                {rocLines.length > 0 && <Grid size={{ xs: 12, md: 6 }}>
                    <Box className="glass-card" sx={{ p: 3 }}><Typography variant="subtitle1" sx={{ color: 'text.primary', mb: 2 }}>ROC Curves</Typography>
                        <ResponsiveContainer width="100%" height={260}>
                            <LineChart margin={{ left: 0, right: 10, bottom: 5 }}>
                                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" /><XAxis dataKey="fpr" type="number" domain={[0, 1]} tick={{ fill: '#64748b', fontSize: 10 }} label={{ value: 'FPR', fill: '#64748b', fontSize: 11, position: 'bottom' }} /><YAxis type="number" domain={[0, 1]} tick={{ fill: '#64748b', fontSize: 10 }} label={{ value: 'TPR', fill: '#64748b', fontSize: 11, angle: -90, position: 'insideLeft' }} />
                                <RTooltip {...TT} /><Legend wrapperStyle={{ color: '#94a3b8', fontSize: 11 }} />
                                {rocLines.map(r => <Line key={r.modelKey} data={r.points} type="monotone" dataKey="tpr" name={`${r.modelName} (AUC: ${r.auc})`} stroke={MODEL_COLORS[r.modelKey]} strokeWidth={2} dot={false} />)}
                                <Line data={[{ fpr: 0, tpr: 0 }, { fpr: 1, tpr: 1 }]} type="linear" dataKey="tpr" name="Random" stroke="#475569" strokeDasharray="5 5" strokeWidth={1} dot={false} />
                            </LineChart>
                        </ResponsiveContainer>
                    </Box>
                </Grid>}
                {/* CV Scores */}
                {cvBarData.length > 0 && <Grid size={{ xs: 12, md: 6 }}>
                    <Box className="glass-card" sx={{ p: 3 }}><Typography variant="subtitle1" sx={{ color: 'text.primary', mb: 2 }}>Cross-Validation (5-Fold)</Typography>
                        <ResponsiveContainer width="100%" height={260}><BarChart data={cvBarData}><CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" /><XAxis dataKey="name" tick={{ fill: '#94a3b8', fontSize: 11 }} axisLine={false} /><YAxis domain={[90, 100]} tick={{ fill: '#64748b', fontSize: 11 }} axisLine={false} /><RTooltip {...TT} formatter={(v: any, n: any) => [n === 'std' ? `+/-${v}` : `${v}%`, n === 'std' ? 'Std Dev' : 'Mean Accuracy']} /><Bar dataKey="mean" name="Mean" radius={[6, 6, 0, 0]}>{cvBarData.map((d, i) => <Cell key={i} fill={d.color} />)}</Bar></BarChart></ResponsiveContainer>
                        <Stack spacing={0.5} mt={2}>{cvData.map(cv => <Typography key={cv.modelKey} variant="caption" sx={{ color: '#94a3b8' }}>{cv.modelName}: {(cv.mean * 100).toFixed(1)}% (+/- {(cv.std * 100).toFixed(1)}%)</Typography>)}</Stack>
                    </Box>
                </Grid>}
                {/* Confusion Matrices */}
                <Grid size={12}><Typography variant="subtitle1" sx={{ color: 'text.primary', mb: 2 }}>Confusion Matrices</Typography>
                    <Grid container spacing={2}>{data.map(m => (<Grid size={{ xs: 12, md: 4 }} key={m.modelKey}><Box className="glass-card" sx={{ p: 3 }}>
                        <Stack direction="row" spacing={1} alignItems="center" mb={2}><Box sx={{ width: 10, height: 10, borderRadius: '50%', bgcolor: MODEL_COLORS[m.modelKey] }} /><Typography variant="subtitle1" sx={{ color: 'text.primary', fontSize: '0.85rem' }}>{m.modelName}</Typography></Stack>
                        <table className="heatmap-table"><thead><tr><th></th><th>Pred B</th><th>Pred M</th></tr></thead><tbody>
                            <tr><th>Act B</th><td className="heatmap-cell-high">{m.confusionMatrix[0]?.[0]}</td><td className="heatmap-cell-low">{m.confusionMatrix[0]?.[1]}</td></tr>
                            <tr><th>Act M</th><td className="heatmap-cell-low">{m.confusionMatrix[1]?.[0]}</td><td className="heatmap-cell-high">{m.confusionMatrix[1]?.[1]}</td></tr>
                        </tbody></table></Box></Grid>))}</Grid>
                </Grid>
            </Grid>)}
        </Box>
    )
}
