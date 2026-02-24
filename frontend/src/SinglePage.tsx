import { useEffect, useState } from 'react'
import { Box, Typography, Grid, TextField, Button, Stack, Chip, FormControl, InputLabel, Select, MenuItem, type SelectChangeEvent } from '@mui/material'
import ScienceIcon from '@mui/icons-material/Science'
import LocalHospitalIcon from '@mui/icons-material/LocalHospital'
import DownloadIcon from '@mui/icons-material/Download'
import axios from 'axios'
import { RadialBarChart, RadialBar, ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip as RTooltip, Cell } from 'recharts'
import { C, API, TT, FGROUPS, MODEL_COLORS, type FeatureMeta, type PredictionResponse, type ModelInfo, type HistoryEntry, type ConsensusResult } from './shared'
import jsPDF from 'jspdf'

function ModelSelector({ models, value, onChange }: { models: ModelInfo[]; value: string; onChange: (v: string) => void }) {
    return (<FormControl size="small" sx={{ minWidth: 180 }}><InputLabel>Model</InputLabel>
        <Select value={value} label="Model" onChange={(e: SelectChangeEvent) => onChange(e.target.value)}>
            {models.map(m => <MenuItem key={m.key} value={m.key}><Stack direction="row" spacing={1} alignItems="center"><Box sx={{ width: 8, height: 8, borderRadius: '50%', bgcolor: MODEL_COLORS[m.key] || C.teal }} /><span>{m.name}</span></Stack></MenuItem>)}
        </Select></FormControl>)
}

export function SinglePage({ features, loadingMeta, onError, models, addHistory, addToast }: {
    features: FeatureMeta[]; loadingMeta: boolean; onError: (m: string | null) => void; models: ModelInfo[]
    addHistory: (e: Omit<HistoryEntry, 'id' | 'time'>) => void; addToast: (m: string, s?: any) => void
}) {
    const [values, setValues] = useState<Record<string, string>>({})
    const [loading, setLoading] = useState(false)
    const [result, setResult] = useState<PredictionResponse | null>(null)
    const [consensus, setConsensus] = useState<ConsensusResult | null>(null)
    const [shapValues, setShapValues] = useState<{ feature: string; value: number }[]>([])
    const [modelKey, setModelKey] = useState('rf')

    useEffect(() => { if (features.length) { const init: Record<string, string> = {}; features.forEach(f => { init[f.name] = String(Number(f.default.toFixed(3))) }); setValues(init) } }, [features])

    const getNumeric = () => {
        const numeric: Record<string, number> = {}
        for (const f of features) { const p = Number(values[f.name]); if (Number.isNaN(p)) throw new Error(`Invalid: ${f.name}`); numeric[f.name] = p }
        return numeric
    }

    const fetchShap = async (numeric: Record<string, number>, mk: string) => {
        try {
            const res = await axios.post(`${API}/api/shap-values`, { features: numeric, model: mk })
            setShapValues(res.data.shapValues || [])
        } catch { setShapValues([]) }
    }

    const handleSubmit = async () => {
        setLoading(true); setResult(null); setConsensus(null); setShapValues([])
        try {
            const numeric = getNumeric()
            const res = await axios.post<PredictionResponse>(`${API}/api/predict`, { features: numeric, model: modelKey })
            setResult(res.data)
            const conf = Math.round((res.data.label === 'Malignant' ? res.data.probabilities.malignant : res.data.probabilities.benign) * 100)
            addHistory({ model: res.data.modelName, label: res.data.label, confidence: conf })
            fetchShap(numeric, modelKey)
        } catch (e: any) { onError(e?.response?.data?.error || e?.message || 'Failed.') } finally { setLoading(false) }
    }

    const handleConsensus = async () => {
        setLoading(true); setResult(null); setConsensus(null); setShapValues([])
        try {
            const numeric = getNumeric()
            const res = await axios.post<ConsensusResult>(`${API}/api/predict-all`, { features: numeric })
            setConsensus(res.data)
            addHistory({ model: 'Consensus', label: res.data.consensus, confidence: res.data.agreement })
        } catch (e: any) { onError(e?.response?.data?.error || e?.message || 'Failed.') } finally { setLoading(false) }
    }

    const downloadPDF = () => {
        if (!result && !consensus) return
        const doc = new jsPDF()
        doc.setFontSize(20); doc.setTextColor(0, 180, 140)
        doc.text('Breast Cancer AI Studio', 20, 20)
        doc.setFontSize(10); doc.setTextColor(100)
        doc.text(`Report generated: ${new Date().toLocaleString()}`, 20, 28)
        doc.setDrawColor(0, 212, 170); doc.line(20, 32, 190, 32)
        let y = 40
        if (result) {
            doc.setFontSize(14); doc.setTextColor(40)
            doc.text(`Prediction: ${result.label}`, 20, y); y += 8
            doc.setFontSize(10); doc.setTextColor(80)
            doc.text(`Model: ${result.modelName}`, 20, y); y += 6
            doc.text(`Confidence: Benign ${(result.probabilities.benign * 100).toFixed(1)}% | Malignant ${(result.probabilities.malignant * 100).toFixed(1)}%`, 20, y); y += 6
            if (result.confidenceInterval) { doc.text(`90% CI: [${(result.confidenceInterval.low * 100).toFixed(1)}%, ${(result.confidenceInterval.high * 100).toFixed(1)}%]`, 20, y); y += 8 }
            if (shapValues.length) {
                doc.setFontSize(12); doc.setTextColor(40); doc.text('SHAP Feature Contributions:', 20, y); y += 6
                doc.setFontSize(9); doc.setTextColor(60)
                shapValues.slice(0, 10).forEach(sv => { doc.text(`  ${sv.feature}: ${sv.value > 0 ? '+' : ''}${sv.value.toFixed(4)}`, 20, y); y += 5 })
            }
        }
        if (consensus) {
            doc.setFontSize(14); doc.setTextColor(40)
            doc.text(`Consensus: ${consensus.consensus} (${consensus.agreement}% agreement)`, 20, y); y += 10
            consensus.results.forEach(r => { doc.setFontSize(10); doc.text(`${r.modelName}: ${r.label} (${(Math.max(r.probabilities.benign, r.probabilities.malignant) * 100).toFixed(1)}%)`, 24, y); y += 6 })
        }
        y += 6; doc.setFontSize(12); doc.setTextColor(40); doc.text('Patient Features:', 20, y); y += 6
        doc.setFontSize(8); doc.setTextColor(80)
        const cols = 3; const fList = Object.entries(values); const perCol = Math.ceil(fList.length / cols)
        for (let c = 0; c < cols; c++) {
            const slice = fList.slice(c * perCol, (c + 1) * perCol)
            let yy = y
            slice.forEach(([k, v]) => { doc.text(`${k}: ${v}`, 20 + c * 60, yy); yy += 4 })
        }
        doc.save('breast_cancer_report.pdf')
        addToast('PDF report downloaded', 'success')
    }

    const grouped = FGROUPS.map(g => ({ ...g, features: features.filter(f => g.keywords.some(kw => f.name.includes(kw))) }))
    const gNames = new Set(grouped.flatMap(g => g.features.map(f => f.name)))
    const other = features.filter(f => !gNames.has(f.name))
    const gauge = result ? [{ name: result.label, value: Math.round((result.label === 'Malignant' ? result.probabilities.malignant : result.probabilities.benign) * 100), fill: result.label === 'Malignant' ? C.rose : C.teal }] : []

    return (
        <Grid container spacing={3}>
            <Grid size={{ xs: 12, md: 7 }}>
                <Box className="animate-fade-in-up">
                    <Stack direction="row" spacing={2} alignItems="center" mb={2}>
                        <Typography variant="h5" sx={{ color: 'text.primary', fontWeight: 700, flex: 1 }}>Patient Features</Typography>
                        <ModelSelector models={models} value={modelKey} onChange={setModelKey} />
                    </Stack>
                    <Stack direction="row" spacing={1} mb={2}>
                        {[{ l: 'Benign-like', s: 0.9 }, { l: 'Malignant-like', s: 1.2 }].map(p => <Button key={p.l} variant="text" size="small" onClick={() => { const n: Record<string, string> = {}; features.forEach(f => { n[f.name] = (f.default * p.s).toFixed(3) }); setValues(n) }} sx={{ fontSize: '0.75rem' }}>{p.l}</Button>)}
                    </Stack>
                    <Box sx={{ maxHeight: 400, overflowY: 'auto', pr: 1 }}>
                        {grouped.map(g => g.features.length > 0 && (<Box key={g.title} sx={{ mb: 3 }}>
                            <Typography variant="subtitle1" sx={{ color: 'text.secondary', mb: 1.5, fontSize: '0.8rem', textTransform: 'uppercase', letterSpacing: '0.05em' }}>{g.title}</Typography>
                            <Grid container spacing={1.5}>{g.features.map(f => <Grid size={{ xs: 12, sm: 6, md: 4 }} key={f.name}><TextField fullWidth size="small" label={f.name} value={values[f.name] || ''} type="number" onChange={e => setValues(prev => ({ ...prev, [f.name]: e.target.value }))} InputLabelProps={{ shrink: true }} /></Grid>)}</Grid>
                        </Box>))}
                        {other.length > 0 && <Box sx={{ mb: 2 }}><Typography variant="subtitle1" sx={{ color: 'text.secondary', mb: 1.5, fontSize: '0.8rem', textTransform: 'uppercase' }}>Other</Typography><Grid container spacing={1.5}>{other.map(f => <Grid size={{ xs: 12, sm: 6, md: 4 }} key={f.name}><TextField fullWidth size="small" label={f.name} value={values[f.name] || ''} type="number" onChange={e => setValues(prev => ({ ...prev, [f.name]: e.target.value }))} InputLabelProps={{ shrink: true }} /></Grid>)}</Grid></Box>}
                    </Box>
                    <Stack direction="row" spacing={2} sx={{ mt: 3 }}>
                        <Button variant="contained" size="large" fullWidth onClick={handleSubmit} disabled={loading || loadingMeta || !features.length} sx={{ py: 1.5 }}>{loading ? 'Analyzing...' : 'Run Prediction'}</Button>
                        <Button variant="outlined" size="large" onClick={handleConsensus} disabled={loading || !features.length} sx={{ py: 1.5, whiteSpace: 'nowrap' }}>All Models</Button>
                    </Stack>
                </Box>
            </Grid>
            <Grid size={{ xs: 12, md: 5 }}>
                <Box className="animate-fade-in-up stagger-2">
                    <Stack direction="row" spacing={1} alignItems="center" mb={1}>
                        <Typography variant="h5" sx={{ color: 'text.primary', fontWeight: 700, flex: 1 }}>Result</Typography>
                        {(result || consensus) && <Button size="small" variant="outlined" startIcon={<DownloadIcon />} onClick={downloadPDF} sx={{ fontSize: '0.7rem' }}>PDF Report</Button>}
                    </Stack>
                    <Box className={`glass-card ${result ? (result.label === 'Malignant' ? 'result-malignant' : 'result-benign') : consensus ? (consensus.consensus === 'Malignant' ? 'result-malignant' : 'result-benign') : ''}`} sx={{ p: 3, mb: 3 }}>
                        {result ? (<>
                            <Stack direction="row" spacing={1.5} alignItems="center" mb={1}><LocalHospitalIcon sx={{ color: result.label === 'Malignant' ? C.rose : C.teal, fontSize: 28 }} /><Typography variant="h5" sx={{ fontWeight: 800, color: result.label === 'Malignant' ? C.rose : C.teal }}>{result.label}</Typography></Stack>
                            <Chip label={result.modelName} size="small" sx={{ bgcolor: 'rgba(255,255,255,0.05)', color: 'text.secondary', mb: 2 }} />
                            <Box sx={{ display: 'flex', justifyContent: 'center' }}><ResponsiveContainer width="100%" height={140}><RadialBarChart cx="50%" cy="50%" innerRadius="60%" outerRadius="90%" startAngle={180} endAngle={0} data={gauge} barSize={16}><RadialBar dataKey="value" cornerRadius={10} background={{ fill: 'rgba(255,255,255,0.04)' }} /></RadialBarChart></ResponsiveContainer></Box>
                            <Typography variant="h4" align="center" sx={{ fontWeight: 800, color: gauge[0]?.fill, mt: -5, mb: 1 }}>{gauge[0]?.value}%</Typography>
                            <Typography variant="caption" align="center" display="block" sx={{ color: 'text.secondary', mb: 1 }}>Confidence</Typography>
                            {result.confidenceInterval && <Typography variant="caption" align="center" display="block" sx={{ color: 'text.secondary', mb: 2 }}>90% CI: [{(result.confidenceInterval.low * 100).toFixed(1)}% — {(result.confidenceInterval.high * 100).toFixed(1)}%]</Typography>}
                            <Stack direction="row" spacing={1} justifyContent="center">
                                <Chip label={`B: ${(result.probabilities.benign * 100).toFixed(1)}%`} sx={{ bgcolor: 'rgba(0,212,170,0.1)', color: C.teal, fontWeight: 700 }} />
                                <Chip label={`M: ${(result.probabilities.malignant * 100).toFixed(1)}%`} sx={{ bgcolor: 'rgba(248,113,113,0.1)', color: C.rose, fontWeight: 700 }} />
                            </Stack>
                            {shapValues.length > 0 && (<Box sx={{ mt: 3 }}>
                                <Typography variant="subtitle1" sx={{ color: 'text.primary', mb: 1, fontSize: '0.85rem' }}>SHAP Explanations</Typography>
                                <ResponsiveContainer width="100%" height={180}><BarChart data={shapValues.slice(0, 8).map(s => ({ name: s.feature.length > 12 ? s.feature.slice(0, 10) + '..' : s.feature, value: s.value }))} layout="vertical" margin={{ left: 5, right: 10 }}>
                                    <XAxis type="number" tick={{ fill: '#64748b', fontSize: 10 }} axisLine={false} /><YAxis type="category" dataKey="name" tick={{ fill: '#94a3b8', fontSize: 9 }} width={80} axisLine={false} />
                                    <RTooltip {...TT} /><Bar dataKey="value" radius={[0, 4, 4, 0]}>{shapValues.slice(0, 8).map((s, i) => <Cell key={i} fill={s.value > 0 ? C.rose : C.teal} />)}</Bar>
                                </BarChart></ResponsiveContainer>
                                <Typography variant="caption" sx={{ color: 'text.secondary' }}>SHAP: Red = pushes towards malignant, Teal = towards benign</Typography>
                            </Box>)}
                        </>) : consensus ? (<>
                            <Typography variant="h5" align="center" sx={{ fontWeight: 800, color: consensus.consensus === 'Malignant' ? C.rose : C.teal, mb: 1 }}>Consensus: {consensus.consensus}</Typography>
                            <Typography variant="body2" align="center" sx={{ color: 'text.secondary', mb: 2 }}>{consensus.agreement}% model agreement</Typography>
                            <Stack spacing={1}>{consensus.results.map(r => (
                                <Box key={r.modelKey} sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', p: 1.5, borderRadius: '12px', background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.06)' }}>
                                    <Stack direction="row" spacing={1} alignItems="center"><Box sx={{ width: 8, height: 8, borderRadius: '50%', bgcolor: MODEL_COLORS[r.modelKey] }} /><Typography variant="body2" sx={{ color: 'text.primary', fontWeight: 600 }}>{r.modelName}</Typography></Stack>
                                    <Stack direction="row" spacing={1}><Chip label={r.label} size="small" sx={{ bgcolor: r.label === 'Malignant' ? 'rgba(248,113,113,0.15)' : 'rgba(0,212,170,0.15)', color: r.label === 'Malignant' ? C.rose : C.teal, fontWeight: 700 }} /><Typography variant="caption" sx={{ color: 'text.secondary', alignSelf: 'center' }}>{(Math.max(r.probabilities.benign, r.probabilities.malignant) * 100).toFixed(0)}%</Typography></Stack>
                                </Box>
                            ))}</Stack>
                        </>) : (<Box sx={{ textAlign: 'center', py: 4 }}><ScienceIcon sx={{ fontSize: 48, color: 'text.secondary', opacity: 0.2, mb: 1 }} /><Typography variant="body2" sx={{ color: 'text.secondary' }}>Run a prediction or try All Models consensus.</Typography></Box>)}
                    </Box>
                </Box>
            </Grid>
        </Grid>
    )
}
