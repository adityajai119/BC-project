import { useEffect, useState, useCallback } from 'react'
import { Box, Typography, Grid, Slider, Stack, Chip, FormControl, InputLabel, Select, MenuItem, type SelectChangeEvent } from '@mui/material'
import LocalHospitalIcon from '@mui/icons-material/LocalHospital'
import axios from 'axios'
import { RadialBarChart, RadialBar, ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip as RTooltip, Cell } from 'recharts'
import { C, API, TT, MODEL_COLORS, type FeatureMeta, type ModelInfo, type PredictionResponse } from './shared'

export function WhatIfPage({ features, models }: { features: FeatureMeta[]; models: ModelInfo[] }) {
    const [values, setValues] = useState<Record<string, number>>({})
    const [modelKey, setModelKey] = useState('rf')
    const [result, setResult] = useState<PredictionResponse | null>(null)
    const [debounceTimer, setDebounceTimer] = useState<number | null>(null)

    useEffect(() => {
        if (features.length) {
            const init: Record<string, number> = {}
            features.forEach(f => { init[f.name] = Number(f.default.toFixed(3)) })
            setValues(init)
        }
    }, [features])

    const predict = useCallback(async (vals: Record<string, number>, mk: string) => {
        try {
            const res = await axios.post<PredictionResponse>(`${API}/api/predict`, { features: vals, model: mk })
            setResult(res.data)
        } catch { /* silent */ }
    }, [])

    useEffect(() => {
        if (Object.keys(values).length === 0) return
        if (debounceTimer) clearTimeout(debounceTimer)
        const t = window.setTimeout(() => predict(values, modelKey), 300)
        setDebounceTimer(t)
        return () => clearTimeout(t)
    }, [values, modelKey])

    const handleSlider = (name: string, value: number) => {
        setValues(prev => ({ ...prev, [name]: value }))
    }

    // Get range for slider: 0 to 3x mean (or at least a reasonable range)
    const getRange = (f: FeatureMeta): [number, number] => {
        const mean = f.default
        return [0, Math.max(mean * 3, mean + 1)]
    }

    const gauge = result ? [{ name: result.label, value: Math.round((result.label === 'Malignant' ? result.probabilities.malignant : result.probabilities.benign) * 100), fill: result.label === 'Malignant' ? C.rose : C.teal }] : []

    // Top features to show as sliders (most important first)
    const topFeatures = features.slice(0, 15)

    return (
        <Box className="animate-fade-in-up">
            <Stack direction="row" spacing={2} alignItems="center" mb={3}>
                <Typography variant="h5" sx={{ color: 'text.primary', fontWeight: 700, flex: 1 }}>What-If Analysis</Typography>
                <FormControl size="small" sx={{ minWidth: 180 }}><InputLabel>Model</InputLabel><Select value={modelKey} label="Model" onChange={(e: SelectChangeEvent) => setModelKey(e.target.value)}>{models.map(m => <MenuItem key={m.key} value={m.key}><Stack direction="row" spacing={1} alignItems="center"><Box sx={{ width: 8, height: 8, borderRadius: '50%', bgcolor: MODEL_COLORS[m.key] || C.teal }} /><span>{m.name}</span></Stack></MenuItem>)}</Select></FormControl>
            </Stack>
            <Typography variant="body2" sx={{ color: '#64748b', mb: 3 }}>Adjust feature values with sliders and watch the prediction change in real-time.</Typography>

            <Grid container spacing={3}>
                {/* Sliders */}
                <Grid size={{ xs: 12, md: 7 }}>
                    <Box className="glass-card" sx={{ p: 3, maxHeight: 540, overflowY: 'auto' }}>
                        <Typography variant="subtitle1" sx={{ color: 'text.primary', mb: 2 }}>Feature Controls</Typography>
                        {topFeatures.map(f => {
                            const [min, max] = getRange(f)
                            const step = (max - min) / 100
                            return (
                                <Box key={f.name} sx={{ mb: 2 }}>
                                    <Stack direction="row" justifyContent="space-between" alignItems="center">
                                        <Typography variant="caption" sx={{ color: '#94a3b8', fontSize: '0.7rem' }}>{f.name}</Typography>
                                        <Typography variant="caption" sx={{ color: C.teal, fontWeight: 700 }}>{(values[f.name] || 0).toFixed(2)}</Typography>
                                    </Stack>
                                    <Slider
                                        value={values[f.name] || 0}
                                        min={min} max={max} step={step}
                                        onChange={(_, v) => handleSlider(f.name, v as number)}
                                        sx={{
                                            color: C.teal, height: 4,
                                            '& .MuiSlider-thumb': { width: 14, height: 14, bgcolor: C.teal },
                                            '& .MuiSlider-track': { background: `linear-gradient(90deg, ${C.teal}, ${C.cyan})` },
                                        }}
                                    />
                                </Box>
                            )
                        })}
                    </Box>
                </Grid>

                {/* Live Result */}
                <Grid size={{ xs: 12, md: 5 }}>
                    <Box className={`glass-card ${result ? (result.label === 'Malignant' ? 'result-malignant' : 'result-benign') : ''}`} sx={{ p: 3, position: 'sticky', top: 80 }}>
                        <Typography variant="subtitle1" sx={{ color: 'text.primary', mb: 2, textAlign: 'center' }}>Live Prediction</Typography>
                        {result ? (<>
                            <Stack direction="row" spacing={1} justifyContent="center" alignItems="center" mb={1}>
                                <LocalHospitalIcon sx={{ color: result.label === 'Malignant' ? C.rose : C.teal, fontSize: 28 }} />
                                <Typography variant="h4" sx={{ fontWeight: 800, color: result.label === 'Malignant' ? C.rose : C.teal }}>{result.label}</Typography>
                            </Stack>
                            <Chip label={result.modelName} size="small" sx={{ bgcolor: 'rgba(255,255,255,0.05)', color: '#94a3b8', display: 'block', mx: 'auto', mb: 2, width: 'fit-content' }} />
                            <Box sx={{ display: 'flex', justifyContent: 'center' }}>
                                <ResponsiveContainer width="100%" height={150}><RadialBarChart cx="50%" cy="50%" innerRadius="60%" outerRadius="90%" startAngle={180} endAngle={0} data={gauge} barSize={16}><RadialBar dataKey="value" cornerRadius={10} background={{ fill: 'rgba(255,255,255,0.04)' }} /></RadialBarChart></ResponsiveContainer>
                            </Box>
                            <Typography variant="h3" align="center" sx={{ fontWeight: 800, color: gauge[0]?.fill, mt: -5, mb: 1 }}>{gauge[0]?.value}%</Typography>
                            <Typography variant="caption" align="center" display="block" sx={{ color: '#64748b', mb: 2 }}>Confidence</Typography>
                            <Stack direction="row" spacing={1} justifyContent="center">
                                <Chip label={`B: ${(result.probabilities.benign * 100).toFixed(1)}%`} sx={{ bgcolor: 'rgba(0,212,170,0.1)', color: C.teal, fontWeight: 700 }} />
                                <Chip label={`M: ${(result.probabilities.malignant * 100).toFixed(1)}%`} sx={{ bgcolor: 'rgba(248,113,113,0.1)', color: C.rose, fontWeight: 700 }} />
                            </Stack>
                            {result.contributions && result.contributions.length > 0 && (<Box sx={{ mt: 3 }}>
                                <Typography variant="subtitle1" sx={{ color: 'text.primary', mb: 1, fontSize: '0.8rem' }}>Key Factors</Typography>
                                <ResponsiveContainer width="100%" height={140}><BarChart data={result.contributions.slice(0, 5).map(c => ({ name: c.feature.slice(0, 10), value: c.contribution }))} layout="vertical"><XAxis type="number" tick={{ fill: '#64748b', fontSize: 9 }} axisLine={false} /><YAxis type="category" dataKey="name" tick={{ fill: '#94a3b8', fontSize: 9 }} width={70} axisLine={false} /><RTooltip {...TT} /><Bar dataKey="value" radius={[0, 4, 4, 0]}>{result.contributions.slice(0, 5).map((c, i) => <Cell key={i} fill={c.contribution > 0 ? C.rose : C.teal} />)}</Bar></BarChart></ResponsiveContainer>
                            </Box>)}
                        </>) : <Typography variant="body2" sx={{ color: '#475569', textAlign: 'center' }}>Adjust sliders to see prediction...</Typography>}
                    </Box>
                </Grid>
            </Grid>
        </Box>
    )
}
