import { useState, useRef, useEffect } from 'react'
import { Box, Typography, IconButton, TextField, Stack, Avatar, Fab, Zoom, Tooltip } from '@mui/material'
import SmartToyIcon from '@mui/icons-material/SmartToy'
import CloseIcon from '@mui/icons-material/Close'
import SendIcon from '@mui/icons-material/Send'
import axios from 'axios'
import { API, C } from './shared'

type Msg = { role: 'user' | 'model'; text: string }

export function Chatbot() {
    const [open, setOpen] = useState(false)
    const [input, setInput] = useState('')
    const [messages, setMessages] = useState<Msg[]>([{ role: 'model', text: "Hi! I'm POTTI 🤖, your breast cancer AI assistant. Ask me anything about the models, features, or predictions!" }])
    const [loading, setLoading] = useState(false)
    const bottomRef = useRef<HTMLDivElement>(null)

    useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: 'smooth' }) }, [messages])

    const send = async () => {
        const msg = input.trim()
        if (!msg || loading) return
        setInput('')
        const userMsg: Msg = { role: 'user', text: msg }
        setMessages(prev => [...prev, userMsg])
        setLoading(true)
        try {
            const history = messages.map(m => ({ role: m.role, text: m.text }))
            const res = await axios.post(`${API}/api/chat`, { message: msg, history })
            setMessages(prev => [...prev, { role: 'model', text: res.data.reply }])
        } catch (e: any) {
            const err = e?.response?.data?.error || 'Failed to get response. Check if GEMINI_API_KEY is set.'
            setMessages(prev => [...prev, { role: 'model', text: `⚠️ ${err}` }])
        } finally { setLoading(false) }
    }

    return (
        <>
            {/* FAB */}
            <Zoom in={!open}>
                <Tooltip title="Chat with POTTI" placement="left">
                    <Fab onClick={() => setOpen(true)} sx={{
                        position: 'fixed', bottom: 24, right: 24, zIndex: 9999,
                        background: 'linear-gradient(135deg, #00d4aa, #00b4d8)',
                        color: '#060918', width: 56, height: 56,
                        boxShadow: '0 6px 24px rgba(0,212,170,0.4)',
                        '&:hover': { background: 'linear-gradient(135deg, #00e8bb, #00c4e8)', transform: 'scale(1.1)' },
                        transition: 'all 0.3s ease',
                    }}>
                        <SmartToyIcon />
                    </Fab>
                </Tooltip>
            </Zoom>
            {/* Chat Window */}
            <Zoom in={open}>
                <Box sx={{
                    position: 'fixed', bottom: 24, right: 24, zIndex: 9999,
                    width: 380, height: 520, borderRadius: '20px',
                    display: 'flex', flexDirection: 'column', overflow: 'hidden',
                    boxShadow: '0 16px 64px rgba(0,0,0,0.4), 0 0 40px rgba(0,212,170,0.08)',
                    border: '1px solid rgba(0,212,170,0.15)',
                    bgcolor: 'background.paper',
                }}>
                    {/* Header */}
                    <Box sx={{
                        background: 'linear-gradient(135deg, #00d4aa, #00b4d8)',
                        px: 2, py: 1.5, display: 'flex', alignItems: 'center', gap: 1.5,
                    }}>
                        <Avatar sx={{ bgcolor: 'rgba(6,9,24,0.3)', width: 36, height: 36 }}><SmartToyIcon sx={{ fontSize: 20 }} /></Avatar>
                        <Box sx={{ flex: 1 }}>
                            <Typography variant="subtitle1" sx={{ color: '#060918', fontWeight: 800, lineHeight: 1.2 }}>POTTI</Typography>
                            <Typography variant="caption" sx={{ color: 'rgba(6,9,24,0.6)', fontSize: '0.6rem' }}>Breast Cancer AI Assistant</Typography>
                        </Box>
                        <IconButton size="small" onClick={() => setOpen(false)} sx={{ color: '#060918' }}><CloseIcon /></IconButton>
                    </Box>
                    {/* Messages */}
                    <Box sx={{ flex: 1, overflowY: 'auto', p: 2, display: 'flex', flexDirection: 'column', gap: 1.5 }}>
                        {messages.map((m, i) => (
                            <Box key={i} sx={{ display: 'flex', justifyContent: m.role === 'user' ? 'flex-end' : 'flex-start', gap: 1 }}>
                                {m.role === 'model' && <Avatar sx={{ bgcolor: C.teal, width: 28, height: 28, fontSize: '0.65rem', mt: 0.5 }}>P</Avatar>}
                                <Box sx={{
                                    maxWidth: '80%', py: 1, px: 1.5, borderRadius: m.role === 'user' ? '16px 16px 4px 16px' : '16px 16px 16px 4px',
                                    background: m.role === 'user' ? 'linear-gradient(135deg, #00d4aa, #00b4d8)' : 'rgba(255,255,255,0.05)',
                                    border: m.role === 'user' ? 'none' : '1px solid rgba(255,255,255,0.08)',
                                    color: m.role === 'user' ? '#060918' : 'text.primary',
                                }}>
                                    <Typography variant="body2" sx={{ fontSize: '0.82rem', lineHeight: 1.5, whiteSpace: 'pre-wrap' }}>{m.text}</Typography>
                                </Box>
                            </Box>
                        ))}
                        {loading && (
                            <Box sx={{ display: 'flex', gap: 1 }}>
                                <Avatar sx={{ bgcolor: C.teal, width: 28, height: 28, fontSize: '0.65rem' }}>P</Avatar>
                                <Box sx={{ py: 1, px: 2, borderRadius: '16px 16px 16px 4px', background: 'rgba(255,255,255,0.05)', border: '1px solid rgba(255,255,255,0.08)' }}>
                                    <Stack direction="row" spacing={0.5}>{[0, 1, 2].map(i => <Box key={i} sx={{ width: 8, height: 8, borderRadius: '50%', bgcolor: C.teal, animation: `pulse 1.2s ease-in-out ${i * 0.15}s infinite`, '@keyframes pulse': { '0%, 100%': { opacity: 0.3, transform: 'scale(0.8)' }, '50%': { opacity: 1, transform: 'scale(1.2)' } } }} />)}</Stack>
                                </Box>
                            </Box>
                        )}
                        <div ref={bottomRef} />
                    </Box>
                    {/* Input */}
                    <Box sx={{ p: 1.5, borderTop: '1px solid', borderColor: 'divider' }}>
                        <Stack direction="row" spacing={1}>
                            <TextField fullWidth size="small" placeholder="Ask POTTI anything..." value={input} onChange={e => setInput(e.target.value)}
                                onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send() } }}
                                sx={{ '& .MuiOutlinedInput-root': { borderRadius: '12px', fontSize: '0.85rem' } }} disabled={loading} />
                            <IconButton onClick={send} disabled={loading || !input.trim()} sx={{ bgcolor: C.teal, color: '#060918', '&:hover': { bgcolor: '#00e8bb' }, borderRadius: '12px', width: 40, height: 40 }}>
                                <SendIcon fontSize="small" />
                            </IconButton>
                        </Stack>
                    </Box>
                </Box>
            </Zoom>
        </>
    )
}
