import { useState } from 'react'
import { Box, Typography, Grid, LinearProgress, Stack, Chip, FormControl, InputLabel, Select, MenuItem, type SelectChangeEvent } from '@mui/material'
import DatasetIcon from '@mui/icons-material/Dataset'
import axios from 'axios'
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip as RTooltip } from 'recharts'
import { C, API, PIE_C, TT, MODEL_COLORS, type ModelInfo } from './shared'

export function BatchPage({ onError, models }: { onError: (m: string | null) => void; models: ModelInfo[] }) {
    const [fileName, setFileName] = useState<string | null>(null)
    const [loading, setLoading] = useState(false)
    const [results, setResults] = useState<{ index: number; label: string }[]>([])
    const [modelKey, setModelKey] = useState('rf')

    const handleFile = async (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0]; if (!file) return
        setFileName(file.name); setLoading(true); setResults([])
        try {
            const fd = new FormData(); fd.append('file', file)
            const res = await axios.post(`${API}/api/predict-batch?model=${modelKey}`, fd, { headers: { 'Content-Type': 'multipart/form-data' } })
            setResults(res.data.results)
        } catch (e: any) { onError(e?.response?.data?.error || e?.message || 'Failed.') } finally { setLoading(false) }
    }

    const bc = results.filter(r => r.label === 'Benign').length, mc = results.filter(r => r.label === 'Malignant').length
    const sp = results.length > 0 ? [{ name: 'Benign', value: bc }, { name: 'Malignant', value: mc }] : []

    return (
        <Box className="animate-fade-in-up">
            <Stack direction="row" spacing={2} alignItems="center" mb={3}>
                <Typography variant="h5" sx={{ color: 'text.primary', fontWeight: 700, flex: 1 }}>Batch Prediction</Typography>
                <FormControl size="small" sx={{ minWidth: 180 }}><InputLabel>Model</InputLabel><Select value={modelKey} label="Model" onChange={(e: SelectChangeEvent) => setModelKey(e.target.value)}>{models.map(m => <MenuItem key={m.key} value={m.key}><Stack direction="row" spacing={1} alignItems="center"><Box sx={{ width: 8, height: 8, borderRadius: '50%', bgcolor: MODEL_COLORS[m.key] || C.teal }} /><span>{m.name}</span></Stack></MenuItem>)}</Select></FormControl>
            </Stack>
            <label><Box className="upload-area" sx={{ mb: 3 }}><DatasetIcon sx={{ fontSize: 48, color: '#00d4aa', mb: 1, opacity: 0.6 }} /><Typography variant="subtitle1" sx={{ color: 'text.primary' }}>{fileName || 'Drop CSV or click to browse'}</Typography><input type="file" hidden accept=".csv" onChange={handleFile} disabled={loading} /></Box></label>
            {loading && <LinearProgress sx={{ mb: 3 }} />}
            {results.length > 0 && (
                <Grid container spacing={3}>
                    <Grid size={{ xs: 12, md: 4 }}>
                        <Box className="glass-card" sx={{ p: 3 }}>
                            <Typography variant="subtitle1" sx={{ color: 'text.primary', mb: 2 }}>Summary</Typography>
                            <ResponsiveContainer width="100%" height={180}><PieChart><Pie data={sp} cx="50%" cy="50%" outerRadius={70} innerRadius={45} dataKey="value" stroke="none">{sp.map((_, i) => <Cell key={i} fill={PIE_C[i]} />)}</Pie><RTooltip {...TT} /></PieChart></ResponsiveContainer>
                            <Stack spacing={1} mt={1}><Box sx={{ display: 'flex', justifyContent: 'space-between' }}><Typography variant="body2" sx={{ color: C.teal }}>Benign</Typography><Typography variant="body2" sx={{ color: 'text.primary', fontWeight: 700 }}>{bc}</Typography></Box><Box sx={{ display: 'flex', justifyContent: 'space-between' }}><Typography variant="body2" sx={{ color: C.rose }}>Malignant</Typography><Typography variant="body2" sx={{ color: 'text.primary', fontWeight: 700 }}>{mc}</Typography></Box></Stack>
                        </Box>
                    </Grid>
                    <Grid size={{ xs: 12, md: 8 }}>
                        <Box className="glass-card" sx={{ p: 3, maxHeight: 460, overflowY: 'auto' }}>
                            <Typography variant="subtitle1" sx={{ color: 'text.primary', mb: 2 }}>Results ({results.length})</Typography>
                            <Stack spacing={1}>{results.map((r: any) => <Box key={r.index} sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', p: 1.5, borderRadius: '12px', background: r.label === 'Malignant' ? 'rgba(248,113,113,0.06)' : 'rgba(0,212,170,0.04)' }}><Typography variant="body2" sx={{ color: '#94a3b8' }}>Row {r.index + 1}</Typography><Chip label={r.label} size="small" sx={{ bgcolor: r.label === 'Malignant' ? 'rgba(248,113,113,0.15)' : 'rgba(0,212,170,0.15)', color: r.label === 'Malignant' ? C.rose : C.teal, fontWeight: 700 }} /></Box>)}</Stack>
                        </Box>
                    </Grid>
                </Grid>
            )}
        </Box>
    )
}
