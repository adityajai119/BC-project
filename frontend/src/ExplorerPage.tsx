import { useEffect, useState } from 'react'
import { Box, Typography, Grid, Stack, FormControl, InputLabel, Select, MenuItem, LinearProgress, type SelectChangeEvent, Button, Chip, Tabs, Tab } from '@mui/material'
import axios from 'axios'
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip as RTooltip, ResponsiveContainer, BarChart, Bar, Cell, Legend } from 'recharts'
import { C, API, TT, MODEL_COLORS, type FeatureMeta } from './shared'

type CorrMatrix = { features: string[]; matrix: { x: string; y: string; value: number }[] }
type PCAPoint = { x: number; y: number; label: number }
type HistBin = { center: number; benign: number; malignant: number }

export function ExplorerPage({ features }: { features: FeatureMeta[] }) {
    const fNames = features.map(f => f.name)
    const [f1, setF1] = useState(fNames[0] || '')
    const [f2, setF2] = useState(fNames[1] || '')
    const [scatter, setScatter] = useState<{ benign: any[]; malignant: any[] }>({ benign: [], malignant: [] })
    const [corrMatrix, setCorrMatrix] = useState<CorrMatrix>({ features: [], matrix: [] })
    const [pcaData, setPcaData] = useState<{ points: PCAPoint[]; explained: number[] }>({ points: [], explained: [] })
    const [histData, setHistData] = useState<HistBin[]>([])
    const [histFeature, setHistFeature] = useState(fNames[0] || '')
    const [dbPoints, setDbPoints] = useState<any[]>([])
    const [dbGrid, setDbGrid] = useState<any[]>([])
    const [dbF1, setDbF1] = useState(fNames[0] || '')
    const [dbF2, setDbF2] = useState(fNames[1] || '')
    const [dbModel, setDbModel] = useState('rf')
    const [loading, setLoading] = useState(false)
    const [tab, setTab] = useState(0)

    useEffect(() => { if (fNames.length) { if (!f1) setF1(fNames[0]); if (!f2) setF2(fNames[1]); if (!histFeature) setHistFeature(fNames[0]); if (!dbF1) setDbF1(fNames[0]); if (!dbF2) setDbF2(fNames[1]) } }, [fNames])

    useEffect(() => {
        if (!f1 || !f2) return
        setLoading(true)
        Promise.allSettled([
            axios.get(`${API}/api/scatter-data?f1=${f1}&f2=${f2}`),
            axios.get(`${API}/api/correlation-matrix?n=12`),
            axios.get(`${API}/api/pca-data`),
        ]).then(([sc, cm, pca]) => {
            if (sc.status === 'fulfilled') { const pts = sc.value.data.points; setScatter({ benign: pts.filter((p: any) => p.label === 0), malignant: pts.filter((p: any) => p.label === 1) }) }
            if (cm.status === 'fulfilled') setCorrMatrix(cm.value.data)
            if (pca.status === 'fulfilled') setPcaData(pca.value.data)
            setLoading(false)
        })
    }, [f1, f2])

    useEffect(() => { if (histFeature) axios.get(`${API}/api/histogram-data?feature=${histFeature}`).then(r => setHistData(r.data.bins)).catch(() => { }) }, [histFeature])

    const fetchDecisionBoundary = () => {
        if (!dbF1 || !dbF2) return
        setLoading(true)
        axios.post(`${API}/api/decision-boundary`, { f1: dbF1, f2: dbF2, model: dbModel }).then(r => {
            setDbPoints(r.data.points); setDbGrid(r.data.grid)
        }).catch(() => { }).finally(() => setLoading(false))
    }

    const corrColor = (v: number) => {
        const abs = Math.abs(v)
        if (abs > 0.8) return '#00d4aa'
        if (abs > 0.5) return '#00b4d8'
        if (abs > 0.3) return '#7c3aed'
        return '#94a3b8'
    }

    return (
        <Box className="animate-fade-in-up">
            <Typography variant="h5" gutterBottom sx={{ color: 'text.primary', fontWeight: 700 }}>Data Explorer</Typography>
            <Typography variant="body2" sx={{ color: 'text.secondary', mb: 3 }}>Visualize feature distributions, correlations, and model decision boundaries.</Typography>
            {loading && <LinearProgress sx={{ mb: 3 }} />}
            <Tabs value={tab} onChange={(_, v) => setTab(v)} sx={{ mb: 3, '& .MuiTab-root': { textTransform: 'none', fontWeight: 600 } }}>
                <Tab label="Scatter & Correlation" /><Tab label="Histograms" /><Tab label="PCA" /><Tab label="Decision Boundary" />
            </Tabs>
            {tab === 0 && (<Grid container spacing={3}>
                <Grid size={{ xs: 12, md: 7 }}>
                    <Box className="glass-card" sx={{ p: 3 }}>
                        <Stack direction="row" spacing={2} mb={2} flexWrap="wrap">
                            <Typography variant="subtitle1" sx={{ color: 'text.primary', flex: 1 }}>Feature Scatter Plot</Typography>
                            <FormControl size="small" sx={{ minWidth: 140 }}><InputLabel>X Axis</InputLabel><Select value={f1} label="X Axis" onChange={(e: SelectChangeEvent) => setF1(e.target.value)}>{fNames.map(n => <MenuItem key={n} value={n}>{n}</MenuItem>)}</Select></FormControl>
                            <FormControl size="small" sx={{ minWidth: 140 }}><InputLabel>Y Axis</InputLabel><Select value={f2} label="Y Axis" onChange={(e: SelectChangeEvent) => setF2(e.target.value)}>{fNames.map(n => <MenuItem key={n} value={n}>{n}</MenuItem>)}</Select></FormControl>
                        </Stack>
                        <ResponsiveContainer width="100%" height={360}>
                            <ScatterChart margin={{ left: 0, right: 10, bottom: 20 }}>
                                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" /><XAxis dataKey="x" type="number" name={f1} tick={{ fill: '#64748b', fontSize: 10 }} label={{ value: f1, fill: '#64748b', fontSize: 11, position: 'bottom' }} /><YAxis dataKey="y" type="number" name={f2} tick={{ fill: '#64748b', fontSize: 10 }} label={{ value: f2, fill: '#64748b', fontSize: 11, angle: -90, position: 'insideLeft' }} /><RTooltip {...TT} />
                                <Scatter name={`Benign (${scatter.benign.length})`} data={scatter.benign} fill={C.teal} opacity={0.6} />
                                <Scatter name={`Malignant (${scatter.malignant.length})`} data={scatter.malignant} fill={C.rose} opacity={0.6} />
                                <Legend wrapperStyle={{ fontSize: 12 }} />
                            </ScatterChart>
                        </ResponsiveContainer>
                    </Box>
                </Grid>
                <Grid size={{ xs: 12, md: 5 }}>
                    <Box className="glass-card" sx={{ p: 3 }}>
                        <Typography variant="subtitle1" sx={{ color: 'text.primary', mb: 2 }}>Feature Correlations</Typography>
                        {corrMatrix.features.length > 0 ? (
                            <Box sx={{ overflowX: 'auto' }}>
                                <Box sx={{ display: 'grid', gridTemplateColumns: `40px repeat(${corrMatrix.features.length}, 1fr)`, gap: '2px', fontSize: '0.55rem' }}>
                                    <Box />{corrMatrix.features.map(f => <Box key={f} sx={{ textAlign: 'center', color: 'text.secondary', transform: 'rotate(-45deg)', transformOrigin: 'center', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis', height: 40, display: 'flex', alignItems: 'end', justifyContent: 'center' }}>{f.slice(0, 10)}</Box>)}
                                    {corrMatrix.features.map((f1r) => [
                                        <Box key={f1r} sx={{ display: 'flex', alignItems: 'center', color: 'text.secondary', fontSize: '0.5rem', pr: 0.5, justifyContent: 'flex-end' }}>{f1r.slice(0, 8)}</Box>,
                                        ...corrMatrix.features.map(f2r => { const entry = corrMatrix.matrix.find(m => m.x === f1r && m.y === f2r); const v = entry?.value ?? 0; return <Box key={`${f1r}-${f2r}`} sx={{ bgcolor: `${corrColor(v)}${Math.round(Math.abs(v) * 60 + 20).toString(16)}`, borderRadius: '3px', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'text.primary', fontWeight: Math.abs(v) > 0.7 ? 700 : 400, p: 0.3, minHeight: 22 }}>{v.toFixed(1)}</Box> })
                                    ])}
                                </Box>
                            </Box>
                        ) : <Typography variant="body2" sx={{ color: 'text.secondary' }}>Loading...</Typography>}
                    </Box>
                </Grid>
            </Grid>)}
            {tab === 1 && (<Box className="glass-card" sx={{ p: 3 }}>
                <Stack direction="row" spacing={2} mb={2} alignItems="center">
                    <Typography variant="subtitle1" sx={{ color: 'text.primary', flex: 1 }}>Feature Distribution</Typography>
                    <FormControl size="small" sx={{ minWidth: 200 }}><InputLabel>Feature</InputLabel><Select value={histFeature} label="Feature" onChange={(e: SelectChangeEvent) => setHistFeature(e.target.value)}>{fNames.map(n => <MenuItem key={n} value={n}>{n}</MenuItem>)}</Select></FormControl>
                </Stack>
                <ResponsiveContainer width="100%" height={360}>
                    <BarChart data={histData}><CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
                        <XAxis dataKey="center" tick={{ fill: '#64748b', fontSize: 10 }} tickFormatter={(v: number) => v.toFixed(1)} />
                        <YAxis tick={{ fill: '#64748b', fontSize: 10 }} />
                        <RTooltip {...TT} /><Legend />
                        <Bar dataKey="benign" name="Benign" fill={C.teal} opacity={0.7} radius={[2, 2, 0, 0]} />
                        <Bar dataKey="malignant" name="Malignant" fill={C.rose} opacity={0.7} radius={[2, 2, 0, 0]} />
                    </BarChart>
                </ResponsiveContainer>
            </Box>)}
            {tab === 2 && (<Box className="glass-card" sx={{ p: 3 }}>
                <Stack direction="row" spacing={2} mb={2} alignItems="center">
                    <Typography variant="subtitle1" sx={{ color: 'text.primary', flex: 1 }}>PCA Projection (2D)</Typography>
                    {pcaData.explained.length > 0 && <Stack direction="row" spacing={1}>{pcaData.explained.map((v, i) => <Chip key={i} label={`PC${i + 1}: ${(v * 100).toFixed(1)}%`} size="small" sx={{ bgcolor: 'rgba(0,212,170,0.1)', color: C.teal, fontWeight: 600 }} />)}</Stack>}
                </Stack>
                <ResponsiveContainer width="100%" height={400}>
                    <ScatterChart margin={{ left: 0, right: 10, bottom: 20 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
                        <XAxis dataKey="x" type="number" name="PC1" tick={{ fill: '#64748b', fontSize: 10 }} label={{ value: 'Principal Component 1', fill: '#64748b', fontSize: 11, position: 'bottom' }} />
                        <YAxis dataKey="y" type="number" name="PC2" tick={{ fill: '#64748b', fontSize: 10 }} label={{ value: 'Principal Component 2', fill: '#64748b', fontSize: 11, angle: -90, position: 'insideLeft' }} />
                        <RTooltip {...TT} />
                        <Scatter name="Benign" data={pcaData.points.filter(p => p.label === 0)} fill={C.teal} opacity={0.5} />
                        <Scatter name="Malignant" data={pcaData.points.filter(p => p.label === 1)} fill={C.rose} opacity={0.5} />
                        <Legend />
                    </ScatterChart>
                </ResponsiveContainer>
            </Box>)}
            {tab === 3 && (<Box className="glass-card" sx={{ p: 3 }}>
                <Stack direction="row" spacing={2} mb={2} alignItems="center" flexWrap="wrap">
                    <Typography variant="subtitle1" sx={{ color: 'text.primary', flex: 1 }}>Decision Boundary</Typography>
                    <FormControl size="small" sx={{ minWidth: 140 }}><InputLabel>X Feature</InputLabel><Select value={dbF1} label="X Feature" onChange={(e: SelectChangeEvent) => setDbF1(e.target.value)}>{fNames.map(n => <MenuItem key={n} value={n}>{n}</MenuItem>)}</Select></FormControl>
                    <FormControl size="small" sx={{ minWidth: 140 }}><InputLabel>Y Feature</InputLabel><Select value={dbF2} label="Y Feature" onChange={(e: SelectChangeEvent) => setDbF2(e.target.value)}>{fNames.map(n => <MenuItem key={n} value={n}>{n}</MenuItem>)}</Select></FormControl>
                    <FormControl size="small" sx={{ minWidth: 140 }}><InputLabel>Model</InputLabel><Select value={dbModel} label="Model" onChange={(e: SelectChangeEvent) => setDbModel(e.target.value)}>
                        <MenuItem value="rf">Random Forest</MenuItem><MenuItem value="gb">Gradient Boosting</MenuItem><MenuItem value="lr">Logistic Regression</MenuItem>
                    </Select></FormControl>
                    <Button variant="contained" size="small" onClick={fetchDecisionBoundary}>Generate</Button>
                </Stack>
                {dbGrid.length > 0 ? (
                    <ResponsiveContainer width="100%" height={400}>
                        <ScatterChart margin={{ left: 0, right: 10, bottom: 20 }}>
                            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
                            <XAxis dataKey="x" type="number" name={dbF1} tick={{ fill: '#64748b', fontSize: 10 }} label={{ value: `${dbF1} (scaled)`, fill: '#64748b', fontSize: 11, position: 'bottom' }} />
                            <YAxis dataKey="y" type="number" name={dbF2} tick={{ fill: '#64748b', fontSize: 10 }} label={{ value: `${dbF2} (scaled)`, fill: '#64748b', fontSize: 11, angle: -90, position: 'insideLeft' }} />
                            <RTooltip {...TT} />
                            <Scatter name="Boundary" data={dbGrid} fill="transparent">{dbGrid.map((d, i) => <Cell key={i} fill={d.p > 0.5 ? `rgba(248,113,113,${Math.min(d.p * 0.4, 0.3)})` : `rgba(0,212,170,${Math.min((1 - d.p) * 0.4, 0.3)})`} />)}</Scatter>
                            <Scatter name="Benign" data={dbPoints.filter(p => p.label === 0)} fill={C.teal} opacity={0.8} />
                            <Scatter name="Malignant" data={dbPoints.filter(p => p.label === 1)} fill={C.rose} opacity={0.8} />
                            <Legend />
                        </ScatterChart>
                    </ResponsiveContainer>
                ) : <Box sx={{ textAlign: 'center', py: 8 }}><Typography variant="body2" sx={{ color: 'text.secondary' }}>Select features and model, then click "Generate" to visualize the decision boundary.</Typography></Box>}
            </Box>)}
        </Box>
    )
}
