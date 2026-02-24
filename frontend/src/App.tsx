import { useEffect, useMemo, useState } from 'react'
import {
  AppBar,
  Box,
  Button,
  Container,
  CssBaseline,
  LinearProgress,
  Tab,
  Tabs,
  Toolbar,
  Typography,
  Paper,
  Grid,
  TextField,
  Alert,
  Snackbar,
  Chip,
  Stack,
  Card,
  CardContent,
} from '@mui/material'
import AnalyticsIcon from '@mui/icons-material/Analytics'
import ScienceIcon from '@mui/icons-material/Science'
import DatasetIcon from '@mui/icons-material/Dataset'
import axios from 'axios'
import './App.css'

type FeatureMeta = {
  name: string
  default: number
  type: string
}

type MetricsResponse = {
  accuracy: number
  precision: number
  recall: number
  f1: number
  confusionMatrix: number[][]
  modelName: string
  nSamples: number
}

type PredictionResponse = {
  prediction: number
  label: string
  probabilities: {
    benign: number
    malignant: number
  }
  modelName: string
}

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000'

type PageKey = 'overview' | 'single' | 'batch' | 'insights'

function App() {
  const [page, setPage] = useState<PageKey>('overview')
  const [features, setFeatures] = useState<FeatureMeta[]>([])
  const [loadingMeta, setLoadingMeta] = useState(false)
  const [metrics, setMetrics] = useState<MetricsResponse | null>(null)
  const [loadingMetrics, setLoadingMetrics] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const fetchMeta = async () => {
      try {
        setLoadingMeta(true)
        const res = await axios.get<{ features: FeatureMeta[] }>(
          `${API_BASE_URL}/api/feature-metadata`,
        )
        setFeatures(res.data.features)
      } catch (e: any) {
        setError(e?.message || 'Failed to load feature metadata.')
      } finally {
        setLoadingMeta(false)
      }
    }
    fetchMeta()
  }, [])

  useEffect(() => {
    const fetchMetrics = async () => {
      try {
        setLoadingMetrics(true)
        const res = await axios.get<MetricsResponse>(
          `${API_BASE_URL}/api/metrics`,
        )
        setMetrics(res.data)
      } catch {
        // metrics are nice-to-have; don't block the UI
      } finally {
        setLoadingMetrics(false)
      }
    }
    fetchMetrics()
  }, [])

  const handleChangePage = (_: React.SyntheticEvent, value: PageKey) => {
    setPage(value)
  }

  const content = useMemo(() => {
    if (page === 'single') {
      return (
        <SinglePredictionPage
          features={features}
          loadingMeta={loadingMeta}
          onError={setError}
        />
      )
    }
    if (page === 'batch') {
      return <BatchPredictionPage onError={setError} />
    }
    if (page === 'insights') {
      return (
        <InsightsPage
          metrics={metrics}
          loading={loadingMetrics}
        />
      )
    }
    return (
      <OverviewPage
        metrics={metrics}
        loadingMetrics={loadingMetrics}
      />
    )
  }, [page, features, loadingMeta, metrics, loadingMetrics])

  return (
    <>
      <CssBaseline />
      <Box
        sx={{
          minHeight: '100vh',
          background: 'linear-gradient(135deg, #e3f2fd 0%, #e0f7fa 50%, #f3e5f5 100%)',
        }}
      >
        <AppBar
          position="static"
          elevation={0}
          sx={{
            background: 'rgba(13, 71, 161, 0.95)',
            backdropFilter: 'blur(8px)',
          }}
        >
          <Toolbar>
            <ScienceIcon sx={{ mr: 1 }} />
            <Typography variant="h6" sx={{ flexGrow: 1, fontWeight: 600 }}>
              Breast Cancer AI Studio
            </Typography>
            <Typography variant="body2" sx={{ opacity: 0.8 }}>
              React · Flask · ML
            </Typography>
          </Toolbar>
        </AppBar>

        <Container maxWidth="lg" sx={{ py: 4 }}>
          <Paper
            elevation={3}
            sx={{
              borderRadius: 4,
              overflow: 'hidden',
            }}
          >
            <Box
              sx={{
                px: 3,
                pt: 2,
                borderBottom: 1,
                borderColor: 'divider',
                background:
                  'linear-gradient(90deg, rgba(227,242,253,0.9), rgba(232,245,233,0.9))',
              }}
            >
              <Tabs
                value={page}
                onChange={handleChangePage}
                textColor="primary"
                indicatorColor="primary"
                variant="scrollable"
                scrollButtons="auto"
              >
                <Tab label="Overview" value="overview" />
                <Tab label="Single Prediction" value="single" />
                <Tab label="Batch Prediction" value="batch" />
                <Tab label="Model Insights" value="insights" />
              </Tabs>
            </Box>

            {loadingMeta && <LinearProgress />}

            <Box sx={{ p: 3 }}>{content}</Box>
          </Paper>
        </Container>
      </Box>

      <Snackbar
        open={Boolean(error)}
        autoHideDuration={6000}
        onClose={() => setError(null)}
      >
        <Alert
          severity="error"
          onClose={() => setError(null)}
          sx={{ width: '100%' }}
        >
          {error}
        </Alert>
      </Snackbar>
    </>
  )
}

type SinglePredictionPageProps = {
  features: FeatureMeta[]
  loadingMeta: boolean
  onError: (msg: string | null) => void
}

function SinglePredictionPage({
  features,
  loadingMeta,
  onError,
}: SinglePredictionPageProps) {
  const [values, setValues] = useState<Record<string, string>>({})
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<PredictionResponse | null>(null)

  useEffect(() => {
    if (features.length) {
      const initial: Record<string, string> = {}
      features.forEach((f) => {
        initial[f.name] = String(Number(f.default.toFixed(3)))
      })
      setValues(initial)
    }
  }, [features])

  const handleChange = (name: string, value: string) => {
    setValues((prev) => ({ ...prev, [name]: value }))
  }

  const handleSubmit = async () => {
    setLoading(true)
    setResult(null)
    try {
      const numeric: Record<string, number> = {}
      for (const f of features) {
        const raw = values[f.name]
        const parsed = Number(raw)
        if (Number.isNaN(parsed)) {
          throw new Error(`Invalid value for ${f.name}`)
        }
        numeric[f.name] = parsed
      }

      const res = await axios.post<PredictionResponse>(
        `${API_BASE_URL}/api/predict`,
        {
          features: numeric,
        },
      )
      setResult(res.data)
    } catch (e: any) {
      const msg = e?.response?.data?.error || e?.message || 'Prediction failed.'
      onError(msg)
    } finally {
      setLoading(false)
    }
  }

  const riskColor =
    result?.label === 'Malignant' ? 'error' : result?.label === 'Benign' ? 'success' : 'default'

  return (
    <Grid container spacing={3}>
      <Grid item xs={12} md={7}>
        <Typography variant="h5" gutterBottom fontWeight={600}>
          Patient Features
        </Typography>
        <Typography variant="body2" color="text.secondary" gutterBottom>
          Adjust the measurements below to match the patient. Values start from dataset means.
        </Typography>
        <Box
          sx={{
            maxHeight: 520,
            overflowY: 'auto',
            pr: 1,
          }}
        >
          <Grid container spacing={2}>
            {features.map((f) => (
              <Grid item xs={12} sm={6} md={4} key={f.name}>
                <TextField
                  fullWidth
                  size="small"
                  label={f.name}
                  value={values[f.name] || ''}
                  type="number"
                  onChange={(e) => handleChange(f.name, e.target.value)}
                  InputLabelProps={{ shrink: true }}
                />
              </Grid>
            ))}
          </Grid>
        </Box>
        <Box mt={3}>
          <Button
            variant="contained"
            size="large"
            onClick={handleSubmit}
            disabled={loading || loadingMeta || !features.length}
          >
            {loading ? 'Predicting…' : 'Predict'}
          </Button>
        </Box>
      </Grid>

      <Grid item xs={12} md={5}>
        <Typography variant="h5" gutterBottom fontWeight={600}>
          Result
        </Typography>
        <Card
          elevation={4}
          sx={{
            borderRadius: 3,
            background:
              result?.label === 'Malignant'
                ? 'linear-gradient(135deg, #ffebee, #ffcdd2)'
                : result?.label === 'Benign'
                  ? 'linear-gradient(135deg, #e8f5e9, #c8e6c9)'
                  : 'linear-gradient(135deg, #e3f2fd, #bbdefb)',
          }}
        >
          <CardContent>
            {result ? (
              <>
                <Stack direction="row" spacing={1} alignItems="center" mb={1}>
                  <AnalyticsIcon color={riskColor === 'error' ? 'error' : 'primary'} />
                  <Typography variant="h6" fontWeight={700}>
                    {result.label}
                  </Typography>
                </Stack>
                <Typography variant="body2" color="text.secondary" gutterBottom>
                  Model: {result.modelName}
                </Typography>
                <Box mt={2}>
                  <Typography variant="subtitle2" gutterBottom>
                    Confidence
                  </Typography>
                  <Stack direction="row" spacing={1} flexWrap="wrap">
                    <Chip
                      label={`Benign: ${(result.probabilities.benign * 100).toFixed(1)}%`}
                      color="success"
                      variant={
                        result.label === 'Benign' ? 'filled' : 'outlined'
                      }
                    />
                    <Chip
                      label={`Malignant: ${(result.probabilities.malignant * 100).toFixed(1)}%`}
                      color="error"
                      variant={
                        result.label === 'Malignant' ? 'filled' : 'outlined'
                      }
                    />
                  </Stack>
                </Box>
              </>
            ) : (
              <Typography variant="body2" color="text.secondary">
                Run a prediction to see the AI&apos;s assessment.
              </Typography>
            )}
          </CardContent>
        </Card>

        <Box mt={3}>
          <Typography variant="subtitle2" gutterBottom>
            Clinical Disclaimer
          </Typography>
          <Typography variant="body2" color="text.secondary">
            This tool is for educational and exploratory purposes only and is
            not a substitute for professional medical diagnosis or treatment.
          </Typography>
        </Box>
      </Grid>
    </Grid>
  )
}

type BatchPredictionPageProps = {
  onError: (msg: string | null) => void
}

function BatchPredictionPage({ onError }: BatchPredictionPageProps) {
  const [fileName, setFileName] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [results, setResults] = useState<
    { index: number; label: string; prediction: number }[]
  >([])

  const handleFileChange = async (
    event: React.ChangeEvent<HTMLInputElement>,
  ) => {
    const file = event.target.files?.[0]
    if (!file) return
    setFileName(file.name)
    setLoading(true)
    setResults([])
    try {
      const formData = new FormData()
      formData.append('file', file)
      const res = await axios.post<{ results: any[] }>(
        `${API_BASE_URL}/api/predict-batch`,
        formData,
        {
          headers: { 'Content-Type': 'multipart/form-data' },
        },
      )
      setResults(
        res.data.results.map((r) => ({
          index: r.index,
          label: r.label,
          prediction: r.prediction,
        })),
      )
    } catch (e: any) {
      const msg = e?.response?.data?.error || e?.message || 'Batch prediction failed.'
      onError(msg)
    } finally {
      setLoading(false)
    }
  }

  return (
    <Box>
      <Typography variant="h5" gutterBottom fontWeight={600}>
        Batch Prediction
      </Typography>
      <Typography variant="body2" color="text.secondary" gutterBottom>
        Upload a CSV file with the same feature columns as the training dataset
        (without the diagnosis column). The API will return a label for each row.
      </Typography>

      <Box mt={3}>
        <Button
          variant="outlined"
          component="label"
          startIcon={<DatasetIcon />}
          disabled={loading}
        >
          {fileName ? 'Change file' : 'Choose CSV file'}
          <input
            type="file"
            hidden
            accept=".csv,text/csv"
            onChange={handleFileChange}
          />
        </Button>
        {fileName && (
          <Typography
            variant="body2"
            color="text.secondary"
            sx={{ mt: 1 }}
          >
            Selected: {fileName}
          </Typography>
        )}
      </Box>

      <Box mt={4}>
        {loading ? (
          <LinearProgress />
        ) : results.length ? (
          <Box>
            <Typography variant="subtitle1" gutterBottom>
              Predictions
            </Typography>
            <Stack spacing={1}>
              {results.map((r) => (
                <Paper
                  key={r.index}
                  variant="outlined"
                  sx={{
                    p: 1.5,
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                  }}
                >
                  <Typography variant="body2">
                    Row {r.index + 1}
                  </Typography>
                  <Chip
                    label={r.label}
                    color={r.label === 'Malignant' ? 'error' : 'success'}
                    size="small"
                  />
                </Paper>
              ))}
            </Stack>
          </Box>
        ) : (
          <Typography variant="body2" color="text.secondary">
            No predictions yet. Upload a CSV file to see results.
          </Typography>
        )}
      </Box>
    </Box>
  )
}

type OverviewPageProps = {
  metrics: MetricsResponse | null
  loadingMetrics: boolean
}

function OverviewPage({ metrics, loadingMetrics }: OverviewPageProps) {
  return (
    <Box>
      <Typography variant="h4" gutterBottom fontWeight={700}>
        Breast Cancer AI Studio
      </Typography>
      <Typography variant="body1" color="text.secondary" gutterBottom>
        Explore a machine learning model trained on the Wisconsin breast cancer
        dataset. Run single-patient predictions, batch analyses, and view
        high-level model performance metrics.
      </Typography>

      <Grid container spacing={3} mt={1}>
        <Grid item xs={12} md={4}>
          <InfoCard
            title="Single Prediction"
            description="Enter measurements for one patient and get an instant benign vs malignant assessment with confidence."
          />
        </Grid>
        <Grid item xs={12} md={4}>
          <InfoCard
            title="Batch Analysis"
            description="Upload CSV files to score multiple patients at once and review predictions row by row."
          />
        </Grid>
        <Grid item xs={12} md={4}>
          <InfoCard
            title="Model Insights"
            description="Review metrics such as accuracy, precision, recall, and F1-score to understand overall model quality."
          />
        </Grid>
      </Grid>

      <Box mt={4}>
        <Typography variant="h6" gutterBottom>
          Quick Metrics
        </Typography>
        {loadingMetrics && <LinearProgress />}
        {metrics ? (
          <Stack direction={{ xs: 'column', sm: 'row' }} spacing={2}>
            <MetricChip label="Accuracy" value={metrics.accuracy} />
            <MetricChip label="Precision" value={metrics.precision} />
            <MetricChip label="Recall" value={metrics.recall} />
            <MetricChip label="F1 Score" value={metrics.f1} />
          </Stack>
        ) : !loadingMetrics ? (
          <Typography variant="body2" color="text.secondary">
            Metrics are not available yet. Ensure the Flask backend is running.
          </Typography>
        ) : null}
      </Box>
    </Box>
  )
}

type InsightsPageProps = {
  metrics: MetricsResponse | null
  loading: boolean
}

function InsightsPage({ metrics, loading }: InsightsPageProps) {
  return (
    <Box>
      <Typography variant="h5" gutterBottom fontWeight={600}>
        Model Insights
      </Typography>
      <Typography variant="body2" color="text.secondary" gutterBottom>
        High-level quality indicators of the trained Random Forest classifier.
        These metrics are computed on the full dataset using the saved scaler
        and model.
      </Typography>
      {loading && <LinearProgress />}
      {metrics ? (
        <Box mt={3}>
          <Typography variant="subtitle1" gutterBottom>
            Metrics
          </Typography>
          <Stack direction={{ xs: 'column', sm: 'row' }} spacing={2} mb={3}>
            <MetricChip label="Accuracy" value={metrics.accuracy} />
            <MetricChip label="Precision" value={metrics.precision} />
            <MetricChip label="Recall" value={metrics.recall} />
            <MetricChip label="F1 Score" value={metrics.f1} />
          </Stack>
          <Typography variant="subtitle1" gutterBottom>
            Confusion Matrix
          </Typography>
          {metrics.confusionMatrix && (
            <Paper
              variant="outlined"
              sx={{
                display: 'inline-block',
                p: 2,
                backgroundColor: '#fafafa',
              }}
            >
              <table>
                <thead>
                  <tr>
                    <th></th>
                    <th>Pred 0 (Benign)</th>
                    <th>Pred 1 (Malignant)</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td>True 0</td>
                    <td>{metrics.confusionMatrix[0]?.[0]}</td>
                    <td>{metrics.confusionMatrix[0]?.[1]}</td>
                  </tr>
                  <tr>
                    <td>True 1</td>
                    <td>{metrics.confusionMatrix[1]?.[0]}</td>
                    <td>{metrics.confusionMatrix[1]?.[1]}</td>
                  </tr>
                </tbody>
              </table>
            </Paper>
          )}
        </Box>
      ) : !loading ? (
        <Typography variant="body2" color="text.secondary">
          Metrics not available. Start the Flask backend and refresh.
        </Typography>
      ) : null}
    </Box>
  )
}

type InfoCardProps = {
  title: string
  description: string
}

function InfoCard({ title, description }: InfoCardProps) {
  return (
    <Paper
      elevation={2}
      sx={{
        p: 2,
        borderRadius: 3,
        height: '100%',
        background:
          'linear-gradient(135deg, rgba(255,255,255,0.9), rgba(227,242,253,0.9))',
      }}
    >
      <Typography variant="subtitle1" fontWeight={600} gutterBottom>
        {title}
      </Typography>
      <Typography variant="body2" color="text.secondary">
        {description}
      </Typography>
    </Paper>
  )
}

type MetricChipProps = {
  label: string
  value: number
}

function MetricChip({ label, value }: MetricChipProps) {
  return (
    <Chip
      label={`${label}: ${(value * 100).toFixed(1)}%`}
      color="primary"
      variant="outlined"
    />
  )
}

export default App
