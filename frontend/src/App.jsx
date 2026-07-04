import React, { useState, useEffect } from 'react';
import axios from 'axios';
import ReactECharts from 'echarts-for-react';
import { 
  Database, 
  BarChart3, 
  Activity, 
  Sun, 
  Moon, 
  Search, 
  ChevronLeft, 
  ChevronRight, 
  RefreshCw, 
  AlertCircle, 
  CheckCircle2, 
  Sliders, 
  HelpCircle,
  Play,
  Home
} from 'lucide-react';

const API_BASE = "http://localhost:8000";

// Top 10 discriminative features for primary form inputs
const TOP_10_FEATURES = [
  'worst concave points', 
  'worst perimeter', 
  'worst radius', 
  'mean concave points', 
  'worst area', 
  'mean perimeter', 
  'mean radius', 
  'mean area', 
  'mean concavity', 
  'worst concavity'
];

function App() {
  // Global States
  const [activeTab, setActiveTab] = useState('home'); // 'home' | 'data' | 'models' | 'inference'
  const [isDarkMode, setIsDarkMode] = useState(true);

  // Tab 1: Data & EDA States
  const [rawTableData, setRawTableData] = useState([]);
  const [columns, setColumns] = useState([]);
  const [tablePage, setTablePage] = useState(1);
  const [tableTotalPages, setTableTotalPages] = useState(1);
  const [tableTotalRecords, setTableTotalRecords] = useState(0);
  const [tableSearch, setTableSearch] = useState('');
  const [tableTargetFilter, setTableTargetFilter] = useState('');
  const [edaPlots, setEdaPlots] = useState([]);
  const [edaStats, setEdaStats] = useState([]);
  const [selectedPlot, setSelectedPlot] = useState(null);
  const [loadingData, setLoadingData] = useState(false);

  // Tab 2: Model Evaluation States
  const [modelDetails, setModelDetails] = useState([]);
  const [overallPlots, setOverallPlots] = useState({});
  const [selectedModel, setSelectedModel] = useState(null);
  const [loadingModels, setLoadingModels] = useState(false);

  // Tab 3: Inference States
  const [predictorModel, setPredictorModel] = useState('KNN');
  const [features, setFeatures] = useState({});
  const [isExpanderOpen, setIsExpanderOpen] = useState(false);
  const [predictionResult, setPredictionResult] = useState(null);
  const [predicting, setPredicting] = useState(false);
  const [predictionError, setPredictionError] = useState(null);

  // Init theme
  useEffect(() => {
    const root = window.document.documentElement;
    if (isDarkMode) {
      root.classList.add('dark');
    } else {
      root.classList.remove('dark');
    }
  }, [isDarkMode]);

  // Fetch Data & EDA details
  useEffect(() => {
    fetchTableData(tablePage);
    fetchEdaDetails();
    fetchModelDetails();
  }, []);

  const fetchTableData = async (page = 1) => {
    setLoadingData(true);
    try {
      let url = `${API_BASE}/api/data?page=${page}&size=12`;
      if (tableSearch) url += `&search=${encodeURIComponent(tableSearch)}`;
      if (tableTargetFilter !== '') url += `&target=${tableTargetFilter}`;
      
      const response = await axios.get(url);
      setRawTableData(response.data.data);
      setColumns(response.data.columns);
      setTablePage(response.data.page);
      setTableTotalPages(response.data.total_pages);
      setTableTotalRecords(response.data.total_records);
    } catch (err) {
      console.error("Error fetching raw data table:", err);
    } finally {
      setLoadingData(false);
    }
  };

  const fetchEdaDetails = async () => {
    try {
      const response = await axios.get(`${API_BASE}/api/eda`);
      setEdaStats(response.data.statistical_summary);
      setEdaPlots(response.data.plots);
      if (response.data.plots.length > 0) {
        setSelectedPlot(response.data.plots[0]);
      }
    } catch (err) {
      console.error("Error fetching EDA details:", err);
    }
  };

  const fetchModelDetails = async () => {
    setLoadingModels(true);
    try {
      const response = await axios.get(`${API_BASE}/api/models`);
      setModelDetails(response.data.models);
      setOverallPlots(response.data.overall_plots);
      if (response.data.models.length > 0) {
        setSelectedModel(response.data.models[0]);
      }
    } catch (err) {
      console.error("Error fetching model details:", err);
    } finally {
      setLoadingModels(false);
    }
  };

  // Triggers search on table
  const handleSearchSubmit = (e) => {
    e.preventDefault();
    setTablePage(1);
    fetchTableData(1);
  };

  // Triggers target filter on table
  const handleTargetFilterChange = (val) => {
    setTableTargetFilter(val);
    setTablePage(1);
    // Use timeout to let state update
    setTimeout(() => {
      fetchTableData(1);
    }, 50);
  };

  // Prefill prediction form with random sample from backend
  const loadRandomSample = async (targetType = null) => {
    try {
      let url = `${API_BASE}/api/data/random`;
      if (targetType !== null) {
        url += `?target=${targetType}`;
      }
      const response = await axios.get(url);
      const sample = response.data;
      
      // Clean target/ID fields
      delete sample.target;
      delete sample.id;
      
      setFeatures(sample);
      setPredictionResult(null);
      setPredictionError(null);
    } catch (err) {
      console.error("Error loading random sample:", err);
      setPredictionError("Failed to fetch a random sample from the backend API.");
    }
  };

  // Lazy prefill default feature means on init if not loaded
  useEffect(() => {
    if (edaStats.length > 0 && Object.keys(features).length === 0) {
      const defaults = {};
      edaStats.forEach(item => {
        defaults[item.feature] = item.mean;
      });
      setFeatures(defaults);
    }
  }, [edaStats]);

  const handlePredict = async (e) => {
    e.preventDefault();
    setPredicting(true);
    setPredictionError(null);
    setPredictionResult(null);
    try {
      const response = await axios.post(`${API_BASE}/api/predict`, {
        model_name: predictorModel,
        features: features
      });
      setPredictionResult(response.data);
    } catch (err) {
      console.error("Prediction failed:", err);
      setPredictionError(err.response?.data?.detail || "An unexpected error occurred during prediction.");
    } finally {
      setPredicting(false);
    }
  };

  // ECharts visualizer setup for model comparisons
  const getComparisonChartOption = () => {
    if (modelDetails.length === 0) return {};

    const modelNames = modelDetails.map(m => m.name);
    const accuracy = modelDetails.map(m => m.metrics.accuracy);
    const precision = modelDetails.map(m => m.metrics.precision);
    const recall = modelDetails.map(m => m.metrics.recall);
    const f1 = modelDetails.map(m => m.metrics.f1);

    return {
      tooltip: {
        trigger: 'axis',
        axisPointer: { type: 'shadow' },
        backgroundColor: isDarkMode ? '#18181b' : '#ffffff',
        borderColor: isDarkMode ? '#27272a' : '#e4e4e7',
        textStyle: { color: isDarkMode ? '#fafafa' : '#09090b' }
      },
      legend: {
        data: ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
        textStyle: { color: isDarkMode ? '#a1a1aa' : '#71717a' }
      },
      grid: {
        left: '3%',
        right: '4%',
        bottom: '3%',
        containLabel: true
      },
      xAxis: {
        type: 'value',
        boundaryGap: [0, 0.01],
        splitLine: { lineStyle: { color: isDarkMode ? '#27272a' : '#e4e4e7' } },
        axisLabel: { color: isDarkMode ? '#a1a1aa' : '#71717a' }
      },
      yAxis: {
        type: 'category',
        data: modelNames,
        axisLabel: { color: isDarkMode ? '#a1a1aa' : '#71717a' },
        axisLine: { lineStyle: { color: isDarkMode ? '#27272a' : '#e4e4e7' } }
      },
      series: [
        { name: 'Accuracy', type: 'bar', data: accuracy, color: '#3b82f6' },
        { name: 'Precision', type: 'bar', data: precision, color: '#10b981' },
        { name: 'Recall', type: 'bar', data: recall, color: '#f59e0b' },
        { name: 'F1-Score', type: 'bar', data: f1, color: '#8b5cf6' }
      ]
    };
  };

  // Find the top-performing model (highest F1)
  const getBestModel = () => {
    if (modelDetails.length === 0) return null;
    return modelDetails.reduce((best, current) => {
      return (current.metrics.f1 > best.metrics.f1) ? current : best;
    }, modelDetails[0]);
  };

  const bestModel = getBestModel();

  return (
    <div className="min-h-screen bg-zinc-50 dark:bg-[#09090b] text-zinc-900 dark:text-zinc-50 transition-colors duration-300 font-sans flex flex-col">
      {/* ─── Header ──────────────────────────────────────────────────────────── */}
      <header className="border-b border-zinc-200 dark:border-zinc-800 bg-white dark:bg-[#0c0c0f] sticky top-0 z-30">
        <div className="max-w-[1600px] mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-blue-600 text-white rounded-lg flex items-center justify-center">
              <Activity className="w-6 h-6 animate-pulse" />
            </div>
            <div>
              <h1 className="text-xl font-extrabold tracking-tight">OncoSense</h1>
              <p className="text-xs text-zinc-500 dark:text-zinc-400 font-mono">Breast Cancer Classification Pipeline</p>
            </div>
          </div>

          {/* Navigation Tabs */}
          <nav className="hidden md:flex items-center gap-1 bg-zinc-100 dark:bg-[#16161c] p-1 rounded-lg">
            <button
              onClick={() => setActiveTab('home')}
              className={`flex items-center gap-2 px-4 py-2 text-sm font-medium rounded-md transition-all ${
                activeTab === 'home' 
                  ? 'bg-white dark:bg-zinc-800 text-blue-600 dark:text-blue-400 shadow-sm' 
                  : 'text-zinc-600 dark:text-zinc-400 hover:text-zinc-900 dark:hover:text-zinc-200'
              }`}
            >
              <Home className="w-4 h-4" />
              Home
            </button>
            <button
              onClick={() => setActiveTab('data')}
              className={`flex items-center gap-2 px-4 py-2 text-sm font-medium rounded-md transition-all ${
                activeTab === 'data' 
                  ? 'bg-white dark:bg-zinc-800 text-blue-600 dark:text-blue-400 shadow-sm' 
                  : 'text-zinc-600 dark:text-zinc-400 hover:text-zinc-900 dark:hover:text-zinc-200'
              }`}
            >
              <Database className="w-4 h-4" />
              Data & EDA Explorer
            </button>
            <button
              onClick={() => setActiveTab('models')}
              className={`flex items-center gap-2 px-4 py-2 text-sm font-medium rounded-md transition-all ${
                activeTab === 'models' 
                  ? 'bg-white dark:bg-zinc-800 text-blue-600 dark:text-blue-400 shadow-sm' 
                  : 'text-zinc-600 dark:text-zinc-400 hover:text-zinc-900 dark:hover:text-zinc-200'
              }`}
            >
              <BarChart3 className="w-4 h-4" />
              Models Dashboard
            </button>
            <button
              onClick={() => setActiveTab('inference')}
              className={`flex items-center gap-2 px-4 py-2 text-sm font-medium rounded-md transition-all ${
                activeTab === 'inference' 
                  ? 'bg-white dark:bg-zinc-800 text-blue-600 dark:text-blue-400 shadow-sm' 
                  : 'text-zinc-600 dark:text-zinc-400 hover:text-zinc-900 dark:hover:text-zinc-200'
              }`}
            >
              <Sliders className="w-4 h-4" />
              Live Diagnosis
            </button>
          </nav>

          <div className="flex items-center gap-2">
            {/* Theme Toggle */}
            <button
              onClick={() => setIsDarkMode(!isDarkMode)}
              className="p-2 border border-zinc-200 dark:border-zinc-800 rounded-lg bg-zinc-50 dark:bg-zinc-900/50 hover:bg-zinc-100 dark:hover:bg-zinc-800 transition-colors"
              title="Toggle Light/Dark Mode"
            >
              {isDarkMode ? <Sun className="w-4 h-4 text-amber-400" /> : <Moon className="w-4 h-4 text-blue-600" />}
            </button>
          </div>
        </div>
      </header>

      {/* Mobile Nav Header */}
      <div className="md:hidden flex border-b border-zinc-200 dark:border-zinc-800 bg-zinc-100 dark:bg-[#16161c] p-1 sticky top-[73px] z-20">
        <button
          onClick={() => setActiveTab('home')}
          className={`flex-1 flex justify-center items-center gap-1.5 py-2 text-xs font-semibold rounded-md transition-all ${
            activeTab === 'home' ? 'bg-white dark:bg-zinc-800 text-blue-600' : 'text-zinc-500'
          }`}
        >
          <Home className="w-3.5 h-3.5" />
          Home
        </button>
        <button
          onClick={() => setActiveTab('data')}
          className={`flex-1 flex justify-center items-center gap-1.5 py-2 text-xs font-semibold rounded-md transition-all ${
            activeTab === 'data' ? 'bg-white dark:bg-zinc-800 text-blue-600' : 'text-zinc-500'
          }`}
        >
          <Database className="w-3.5 h-3.5" />
          Data
        </button>
        <button
          onClick={() => setActiveTab('models')}
          className={`flex-1 flex justify-center items-center gap-1.5 py-2 text-xs font-semibold rounded-md transition-all ${
            activeTab === 'models' ? 'bg-white dark:bg-zinc-800 text-blue-600' : 'text-zinc-500'
          }`}
        >
          <BarChart3 className="w-3.5 h-3.5" />
          Models
        </button>
        <button
          onClick={() => setActiveTab('inference')}
          className={`flex-1 flex justify-center items-center gap-1.5 py-2 text-xs font-semibold rounded-md transition-all ${
            activeTab === 'inference' ? 'bg-white dark:bg-zinc-800 text-blue-600' : 'text-zinc-500'
          }`}
        >
          <Sliders className="w-3.5 h-3.5" />
          Diagnosis
        </button>
      </div>

      {/* ─── Main Content ─────────────────────────────────────────────────────── */}
      <main className="flex-grow max-w-[1600px] w-full mx-auto p-4 md:p-6">
        
        {/* ====================================================================== */}
        {/* TAB: HOME / EXPLANATION */}
        {/* ====================================================================== */}
        {activeTab === 'home' && (
          <div className="space-y-6 animate-fadeIn max-w-[1000px] mx-auto text-left">
            {/* Hero Banner */}
            <div className="bg-gradient-to-br from-blue-600 to-indigo-700 text-white rounded-2xl p-6 md:p-10 shadow-lg relative overflow-hidden">
              <div className="relative z-10 space-y-4">
                <span className="bg-blue-500/30 text-blue-200 border border-blue-400/30 text-[10px] uppercase font-bold tracking-widest px-2.5 py-1 rounded-full">
                  Clinical Diagnosis Portal
                </span>
                <h2 className="text-3xl md:text-4xl font-extrabold tracking-tight">OncoSense Diagnosis Explorer</h2>
                <p className="text-sm md:text-base text-blue-100 max-w-2xl leading-relaxed">
                  A high-fidelity machine learning pipeline analyzing fine-needle aspirates (FNA) of breast masses to accurately distinguish between malignant and benign tumor cells.
                </p>
              </div>
              <div className="absolute right-0 bottom-0 opacity-10 translate-x-12 translate-y-12 select-none pointer-events-none">
                <Activity className="w-80 h-80" />
              </div>
            </div>

            {/* Grid Layout: Context & Design Guide */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Medical Explanation */}
              <div className="bg-white dark:bg-[#0c0c0f] border border-zinc-200 dark:border-zinc-800 rounded-xl p-6 shadow-sm space-y-4">
                <h3 className="text-lg font-bold tracking-tight text-zinc-950 dark:text-zinc-50 border-b border-zinc-100 dark:border-zinc-800 pb-2">
                  Benign vs. Malignant Cell Diagnosis
                </h3>
                <div className="space-y-3 text-xs leading-relaxed text-zinc-600 dark:text-zinc-400">
                  <p>
                    Breast cancer diagnosis relies on identifying cell abnormalities in biopsies. Cell growths are classified as:
                  </p>
                  <ul className="list-disc pl-4 space-y-2">
                    <li>
                      <strong className="text-rose-600 dark:text-rose-400 font-bold">Malignant (Class 0):</strong> Cancerous cellular structures. These grow quickly, can invade adjacent breast tissue, and carry the risk of spreading (metastasis) if not treated immediately.
                    </li>
                    <li>
                      <strong className="text-emerald-600 dark:text-emerald-400 font-bold">Benign (Class 1):</strong> Non-cancerous masses. They are localized, slow-growing, and do not threaten surrounding healthy tissues.
                    </li>
                  </ul>
                  <p>
                    To predict these categories, our system evaluates cytological attributes extracted from biopsies, including cell nucleus shape regularity, perimeter roughness, and area size.
                  </p>
                </div>
              </div>

              {/* Guide for Non-Tech Users */}
              <div className="bg-white dark:bg-[#0c0c0f] border border-zinc-200 dark:border-zinc-800 rounded-xl p-6 shadow-sm space-y-4">
                <h3 className="text-lg font-bold tracking-tight text-zinc-950 dark:text-zinc-50 border-b border-zinc-100 dark:border-zinc-800 pb-2">
                  System Portal Guide
                </h3>
                <div className="space-y-3 text-xs leading-relaxed text-zinc-600 dark:text-zinc-400">
                  <p>
                    OncoSense utilizes a zinc-inspired structured design system. You can explore the data pipeline through three environments:
                  </p>
                  <ol className="list-decimal pl-4 space-y-3">
                    <li>
                      <strong>Data & EDA Explorer:</strong> View the raw patient records. Check cytological stats (minimums, maximums, standard averages) and visual distributions.
                    </li>
                    <li>
                      <strong>Models Dashboard:</strong> Review performance scores. Non-technical users should look at **Recall** (catching tumors) and **Precision** (avoiding false alarms).
                    </li>
                    <li>
                      <strong>Live Diagnosis:</strong> Use preset patient templates or input values manually to observe real-time AI classification scores.
                    </li>
                  </ol>
                </div>
              </div>
            </div>

            {/* Metrics Explanation Cards */}
            <div className="bg-white dark:bg-[#0c0c0f] border border-zinc-200 dark:border-zinc-800 rounded-xl p-6 shadow-sm space-y-4">
              <h3 className="text-base font-bold tracking-tight">Understanding Diagnosis Metrics (Layperson's Guide)</h3>
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                <div className="p-4 bg-zinc-50 dark:bg-zinc-900/40 rounded-xl border border-zinc-100 dark:border-zinc-800 space-y-2">
                  <div className="text-xs font-bold text-blue-600 dark:text-blue-400 font-mono">1. RECALL (SENSITIVITY)</div>
                  <p className="text-[11px] text-zinc-500 dark:text-zinc-400 leading-relaxed">
                    <strong>The Safety Metric:</strong> "How many of the actual malignant tumors did the model catch?" A 98% recall means we successfully diagnosed 98 out of 100 cancer cases. In clinical oncology, high recall is vital to prevent untreated tumors.
                  </p>
                </div>
                <div className="p-4 bg-zinc-50 dark:bg-zinc-900/40 rounded-xl border border-zinc-100 dark:border-zinc-800 space-y-2">
                  <div className="text-xs font-bold text-emerald-600 dark:text-emerald-400 font-mono">2. PRECISION</div>
                  <p className="text-[11px] text-zinc-500 dark:text-zinc-400 leading-relaxed">
                    <strong>The Accuracy Guarantee:</strong> "When the model flags a case as malignant, how often is it right?" High precision minimizes false alarms (benign masses mistaken for cancer), sparing patients from unnecessary anxiety.
                  </p>
                </div>
                <div className="p-4 bg-zinc-50 dark:bg-zinc-900/40 rounded-xl border border-zinc-100 dark:border-zinc-800 space-y-2">
                  <div className="text-xs font-bold text-purple-600 dark:text-purple-400 font-mono">3. CONFUSION MATRIX</div>
                  <p className="text-[11px] text-zinc-500 dark:text-zinc-400 leading-relaxed">
                    <strong>The Truth Grid:</strong> A grid showing correct predictions (True Positives/Negatives) versus mistakes (False Positives/Negatives) so we can locate where errors happen.
                  </p>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ====================================================================== */}
        {/* TAB 1: DATA & EDA EXPLORER */}
        {/* ====================================================================== */}
        {activeTab === 'data' && (
          <div className="space-y-6 animate-fadeIn">
            {/* Page Header */}
            <div>
              <h2 className="text-2xl font-bold tracking-tight">Data & EDA Explorer</h2>
              <p className="text-sm text-zinc-500 dark:text-zinc-400">
                Explore the Wisconsin Breast Cancer Dataset and examine the exploratory data analysis plots.
              </p>
            </div>

            {/* Split Layout: Data Table & Stats summary */}
            <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
              
              {/* Left Column (Table) */}
              <div className="xl:col-span-2 space-y-4">
                <div className="bg-white dark:bg-[#0c0c0f] border border-zinc-200 dark:border-zinc-800 rounded-xl overflow-hidden shadow-sm">
                  {/* Table Toolbar */}
                  <div className="p-4 border-b border-zinc-200 dark:border-zinc-800 flex flex-col sm:flex-row gap-3 items-center justify-between bg-zinc-50/50 dark:bg-zinc-900/10">
                    <form onSubmit={handleSearchSubmit} className="relative w-full sm:max-w-xs">
                      <span className="absolute inset-y-0 left-0 flex items-center pl-3 pointer-events-none text-zinc-400">
                        <Search className="w-4 h-4" />
                      </span>
                      <input
                        type="text"
                        placeholder="Search values..."
                        value={tableSearch}
                        onChange={(e) => setTableSearch(e.target.value)}
                        className="w-full pl-9 pr-4 py-2 text-xs border border-zinc-200 dark:border-zinc-800 rounded-lg bg-white dark:bg-zinc-950 text-zinc-900 dark:text-zinc-50 focus:outline-none focus:ring-1 focus:ring-blue-500 focus:border-blue-500 shadow-sm"
                      />
                    </form>

                    <div className="flex items-center gap-2 w-full sm:w-auto justify-end">
                      <select
                        value={tableTargetFilter}
                        onChange={(e) => handleTargetFilterChange(e.target.value)}
                        className="px-3 py-2 text-xs border border-zinc-200 dark:border-zinc-800 rounded-lg bg-white dark:bg-zinc-950 text-zinc-900 dark:text-zinc-50 focus:outline-none focus:ring-1 focus:ring-blue-500 shadow-sm"
                      >
                        <option value="">All Diagnoses</option>
                        <option value="0">Malignant (Class 0)</option>
                        <option value="1">Benign (Class 1)</option>
                      </select>
                      
                      <button
                        onClick={() => fetchTableData(tablePage)}
                        className="p-2 border border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-950 rounded-lg text-zinc-500 hover:text-zinc-900 dark:hover:text-zinc-100 transition-colors shadow-sm"
                        title="Reload Table"
                      >
                        <RefreshCw className="w-3.5 h-3.5" />
                      </button>
                    </div>
                  </div>

                  {/* Table Element */}
                  <div className="overflow-x-auto h-[480px] relative">
                    {loadingData ? (
                      <div className="absolute inset-0 bg-white/50 dark:bg-black/50 backdrop-blur-sm flex items-center justify-center z-10">
                        <div className="flex flex-col items-center gap-2">
                          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
                          <span className="text-xs text-zinc-500">Loading samples...</span>
                        </div>
                      </div>
                    ) : null}

                    <table className="w-full text-left border-collapse text-xs">
                      <thead className="bg-zinc-50 dark:bg-zinc-900/60 text-zinc-500 dark:text-zinc-400 sticky top-0 z-10 border-b border-zinc-200 dark:border-zinc-800">
                        <tr>
                          <th className="p-3 font-semibold">ID</th>
                          <th className="p-3 font-semibold">Diagnosis</th>
                          {columns.filter(c => c !== 'target' && c !== 'id').map((col) => (
                            <th key={col} className="p-3 font-semibold capitalize whitespace-nowrap">{col}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-zinc-100 dark:divide-zinc-800/50">
                        {rawTableData.map((row) => (
                          <tr key={row.id} className="hover:bg-zinc-50 dark:hover:bg-zinc-900/30 transition-colors">
                            <td className="p-3 font-mono font-medium text-zinc-500">{row.id}</td>
                            <td className="p-3">
                              <span className={`inline-flex items-center px-2 py-0.5 rounded-full text-[10px] font-bold ${
                                row.target === 0 
                                  ? 'bg-rose-50 dark:bg-rose-950/30 text-rose-600 dark:text-rose-400 border border-rose-200 dark:border-rose-900/30'
                                  : 'bg-emerald-50 dark:bg-emerald-950/30 text-emerald-600 dark:text-emerald-400 border border-emerald-200 dark:border-emerald-900/30'
                              }`}>
                                {row.target === 0 ? 'Malignant' : 'Benign'}
                              </span>
                            </td>
                            {columns.filter(c => c !== 'target' && c !== 'id').map((col) => (
                              <td key={col} className="p-3 font-mono whitespace-nowrap">{row[col]?.toFixed(4)}</td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>

                  {/* Pagination Footer */}
                  <div className="p-4 border-t border-zinc-200 dark:border-zinc-800 flex items-center justify-between bg-zinc-50/50 dark:bg-zinc-900/10 text-xs">
                    <span className="text-zinc-500">
                      Showing <strong>{rawTableData.length}</strong> of <strong>{tableTotalRecords}</strong> samples
                    </span>

                    <div className="flex items-center gap-2">
                      <button
                        onClick={() => {
                          if (tablePage > 1) {
                            fetchTableData(tablePage - 1);
                          }
                        }}
                        disabled={tablePage === 1}
                        className="px-3 py-1.5 border border-zinc-200 dark:border-zinc-800 rounded-lg hover:bg-zinc-50 dark:hover:bg-zinc-900 disabled:opacity-40 disabled:cursor-not-allowed flex items-center gap-1 font-medium transition-colors shadow-sm"
                      >
                        <ChevronLeft className="w-3.5 h-3.5" />
                        Previous
                      </button>
                      <span className="font-mono text-zinc-500">Page {tablePage} of {tableTotalPages}</span>
                      <button
                        onClick={() => {
                          if (tablePage < tableTotalPages) {
                            fetchTableData(tablePage + 1);
                          }
                        }}
                        disabled={tablePage === tableTotalPages}
                        className="px-3 py-1.5 border border-zinc-200 dark:border-zinc-800 rounded-lg hover:bg-zinc-50 dark:hover:bg-zinc-900 disabled:opacity-40 disabled:cursor-not-allowed flex items-center gap-1 font-medium transition-colors shadow-sm"
                      >
                        Next
                        <ChevronRight className="w-3.5 h-3.5" />
                      </button>
                    </div>
                  </div>
                </div>
              </div>

              {/* Right Column (Descriptive Stats summary) */}
              <div className="space-y-4">
                <div className="bg-white dark:bg-[#0c0c0f] border border-zinc-200 dark:border-zinc-800 rounded-xl p-5 shadow-sm h-[578px] flex flex-col">
                  <h3 className="text-sm font-bold uppercase tracking-wider text-zinc-400 mb-3 font-mono">Dataset Summaries</h3>
                  <div className="overflow-y-auto flex-grow divide-y divide-zinc-100 dark:divide-zinc-800/40 pr-2">
                    {edaStats.map((item) => (
                      <div key={item.feature} className="py-2.5 flex flex-col gap-1 text-xs">
                        <div className="flex items-center justify-between">
                          <span className="font-medium capitalize text-zinc-800 dark:text-zinc-200">{item.feature}</span>
                          <span className="font-mono text-zinc-400 bg-zinc-100 dark:bg-zinc-900 px-1.5 py-0.5 rounded text-[10px]">Mean: {item.mean?.toFixed(2)}</span>
                        </div>
                        <div className="grid grid-cols-3 gap-2 font-mono text-[10px] text-zinc-500 mt-1">
                          <div>Min: {item.min?.toFixed(2)}</div>
                          <div>Max: {item.max?.toFixed(2)}</div>
                          <div>Skew: {item.skew?.toFixed(2)}</div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

            </div>

            {/* EDA Visualization Panel */}
            <div className="bg-white dark:bg-[#0c0c0f] border border-zinc-200 dark:border-zinc-800 rounded-xl p-5 shadow-sm">
              <h3 className="text-base font-bold tracking-tight mb-4">Pipeline Visualizations</h3>
              
              <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
                {/* Visualizations List */}
                <div className="space-y-2 flex flex-row lg:flex-col overflow-x-auto lg:overflow-x-visible pb-3 lg:pb-0 gap-2 lg:gap-0 lg:divide-y lg:divide-zinc-100 lg:dark:divide-zinc-800/40">
                  {edaPlots.map((plot) => (
                    <button
                      key={plot.filename}
                      onClick={() => setSelectedPlot(plot)}
                      className={`w-full text-left px-4 py-3 rounded-lg text-xs font-semibold flex items-center justify-between transition-all shrink-0 lg:shrink ${
                        selectedPlot?.filename === plot.filename
                          ? 'bg-blue-50 dark:bg-blue-900/20 text-blue-600 dark:text-blue-400 border border-blue-200 dark:border-blue-900/30'
                          : 'hover:bg-zinc-50 dark:hover:bg-zinc-900/40 text-zinc-600 dark:text-zinc-400 border border-transparent'
                      }`}
                    >
                      <span>{plot.name}</span>
                      <span className="hidden lg:inline text-[10px] font-mono text-zinc-400 px-1.5 py-0.5 bg-zinc-100 dark:bg-zinc-800 rounded capitalize">{plot.type}</span>
                    </button>
                  ))}
                </div>

                {/* Selected Plot Image Viewer */}
                <div className="lg:col-span-3 border border-zinc-200 dark:border-zinc-800 rounded-xl overflow-hidden bg-zinc-50 dark:bg-zinc-900/20 p-4 flex items-center justify-center min-h-[400px]">
                  {selectedPlot ? (
                    <div className="space-y-3 w-full text-center">
                      <img 
                        src={`${API_BASE}/static/figures/${selectedPlot.filename}`} 
                        alt={selectedPlot.name}
                        className="max-h-[500px] object-contain mx-auto rounded-lg shadow-sm border border-zinc-200 dark:border-zinc-800"
                      />
                      <p className="text-xs text-zinc-500 font-mono italic">{selectedPlot.name} (Saved figure: {selectedPlot.filename})</p>
                    </div>
                  ) : (
                    <span className="text-sm text-zinc-400">Select a plot to view.</span>
                  )}
                </div>
              </div>

            </div>

          </div>
        )}

        {/* ====================================================================== */}
        {/* TAB 2: MODEL DASHBOARD */}
        {/* ====================================================================== */}
        {activeTab === 'models' && (
          <div className="space-y-6 animate-fadeIn">
            {/* Page Header */}
            <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
              <div>
                <h2 className="text-2xl font-bold tracking-tight">Models Dashboard</h2>
                <p className="text-sm text-zinc-500 dark:text-zinc-400">
                  Compare cross-validation metrics, confusion matrices, and parameter configurations.
                </p>
              </div>

              {/* Display Best Model KPI */}
              {bestModel && (
                <div className="bg-emerald-50 dark:bg-emerald-950/20 border border-emerald-200 dark:border-emerald-900/30 rounded-xl p-3 flex items-center gap-3">
                  <div className="p-2 bg-emerald-600 text-white rounded-lg">
                    <CheckCircle2 className="w-5 h-5" />
                  </div>
                  <div>
                    <div className="text-[10px] uppercase font-bold text-emerald-600 tracking-wider">Top Performing Model</div>
                    <div className="text-sm font-extrabold text-zinc-800 dark:text-zinc-200">{bestModel.name} <span className="text-xs font-mono text-zinc-500">(F1: {(bestModel.metrics.f1 * 100).toFixed(1)}%)</span></div>
                  </div>
                </div>
              )}
            </div>

            {/* KPI Cards Grid */}
            {bestModel && (
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="bg-white dark:bg-[#0c0c0f] border border-zinc-200 dark:border-zinc-800 rounded-xl p-4 shadow-sm">
                  <div className="text-xs text-zinc-500 dark:text-zinc-400 mb-1">Classifier Model Accuracy</div>
                  <div className="text-2xl font-black font-mono text-blue-600">{(bestModel.metrics.accuracy * 100).toFixed(2)}%</div>
                </div>
                <div className="bg-white dark:bg-[#0c0c0f] border border-zinc-200 dark:border-zinc-800 rounded-xl p-4 shadow-sm">
                  <div className="text-xs text-zinc-500 dark:text-zinc-400 mb-1">Precision (Benign)</div>
                  <div className="text-2xl font-black font-mono text-emerald-600">{(bestModel.metrics.precision * 100).toFixed(2)}%</div>
                </div>
                <div className="bg-white dark:bg-[#0c0c0f] border border-zinc-200 dark:border-zinc-800 rounded-xl p-4 shadow-sm">
                  <div className="text-xs text-zinc-500 dark:text-zinc-400 mb-1">Recall (Benign)</div>
                  <div className="text-2xl font-black font-mono text-orange-500">{(bestModel.metrics.recall * 100).toFixed(2)}%</div>
                </div>
                <div className="bg-white dark:bg-[#0c0c0f] border border-zinc-200 dark:border-zinc-800 rounded-xl p-4 shadow-sm">
                  <div className="text-xs text-zinc-500 dark:text-zinc-400 mb-1">ROC-AUC Margin</div>
                  <div className="text-2xl font-black font-mono text-purple-600">{(bestModel.metrics.roc_auc * 100).toFixed(2)}%</div>
                </div>
              </div>
            )}

            {/* Metrics ECharts Comparison Bar Graph */}
            <div className="bg-white dark:bg-[#0c0c0f] border border-zinc-200 dark:border-zinc-800 rounded-xl p-5 shadow-sm">
              <h3 className="text-base font-bold tracking-tight mb-4">Model Performance Comparisons</h3>
              {loadingModels ? (
                <div className="h-[350px] flex items-center justify-center">
                  <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
                </div>
              ) : (
                <ReactECharts 
                  option={getComparisonChartOption()} 
                  style={{ height: '350px' }}
                  theme={isDarkMode ? 'dark' : 'light'}
                />
              )}
            </div>

            {/* Split Screen: Model Inspector details & Confusion Matrix */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              
              {/* Select Model List */}
              <div className="space-y-4">
                <div className="bg-white dark:bg-[#0c0c0f] border border-zinc-200 dark:border-zinc-800 rounded-xl p-4 shadow-sm">
                  <h3 className="text-sm font-bold uppercase tracking-wider text-zinc-400 mb-3 font-mono">Trained Classifiers</h3>
                  <div className="space-y-2">
                    {modelDetails.map((model) => (
                      <button
                        key={model.name}
                        onClick={() => setSelectedModel(model)}
                        className={`w-full text-left px-4 py-3 rounded-lg text-xs font-semibold flex flex-col gap-1.5 transition-all ${
                          selectedModel?.name === model.name
                            ? 'bg-blue-50 dark:bg-blue-900/20 text-blue-600 dark:text-blue-400 border border-blue-200 dark:border-blue-900/30'
                            : 'hover:bg-zinc-50 dark:hover:bg-zinc-900/40 text-zinc-600 dark:text-zinc-400 border border-transparent'
                        }`}
                      >
                        <div className="flex items-center justify-between w-full">
                          <span>{model.name}</span>
                          <span className="font-mono font-bold text-[10px]">F1: {model.metrics.f1?.toFixed(4)}</span>
                        </div>
                        {selectedModel?.name === model.name && (
                          <div className="text-[10px] text-zinc-500 font-normal leading-relaxed mt-1">
                            <strong>Winning Parameters:</strong>
                            <div className="font-mono text-[9px] mt-1 bg-white/80 dark:bg-zinc-950/80 p-1 rounded max-w-full overflow-hidden text-ellipsis whitespace-nowrap">
                              {JSON.stringify(model.parameters)}
                            </div>
                          </div>
                        )}
                      </button>
                    ))}
                  </div>
                </div>
              </div>

              {/* Model Visualizations (Confusion Matrix & Feature Importance) */}
              <div className="lg:col-span-2 space-y-4">
                <div className="bg-white dark:bg-[#0c0c0f] border border-zinc-200 dark:border-zinc-800 rounded-xl p-5 shadow-sm min-h-[420px] flex flex-col">
                  {selectedModel ? (
                    <div className="space-y-4 flex-grow flex flex-col">
                      <h3 className="text-base font-bold tracking-tight">Visual Evaluation: {selectedModel.name}</h3>
                      <div className="grid grid-cols-1 sm:grid-cols-2 gap-6 flex-grow items-center">
                        {/* Confusion Matrix */}
                        <div className="text-center space-y-2">
                          <div className="text-xs text-zinc-500 font-mono">Confusion Matrix</div>
                          <img 
                            src={`${API_BASE}${selectedModel.confusion_matrix_url}`} 
                            alt={`Confusion Matrix for ${selectedModel.name}`}
                            className="max-h-[280px] w-auto mx-auto rounded-lg border border-zinc-200 dark:border-zinc-800 shadow-sm"
                            onError={(e) => {
                              e.target.style.display = 'none';
                            }}
                          />
                        </div>

                        {/* Feature Importance (if tree model) or ROC curve */}
                        <div className="text-center space-y-2">
                          {selectedModel.feature_importance_url ? (
                            <>
                              <div className="text-xs text-zinc-500 font-mono">Top Feature Importance</div>
                              <img 
                                src={`${API_BASE}${selectedModel.feature_importance_url}`} 
                                alt={`Feature Importance for ${selectedModel.name}`}
                                className="max-h-[280px] w-auto mx-auto rounded-lg border border-zinc-200 dark:border-zinc-800 shadow-sm"
                              />
                            </>
                          ) : (
                            <>
                              <div className="text-xs text-zinc-500 font-mono">Composite ROC Curves</div>
                              {overallPlots.roc_curves ? (
                                <img 
                                  src={`${API_BASE}${overallPlots.roc_curves}`} 
                                  alt="ROC Curves All Models"
                                  className="max-h-[280px] w-auto mx-auto rounded-lg border border-zinc-200 dark:border-zinc-800 shadow-sm"
                                />
                              ) : (
                                <div className="h-[280px] flex items-center justify-center text-xs text-zinc-400">ROC curves plot not generated.</div>
                              )}
                            </>
                          )}
                        </div>
                      </div>
                    </div>
                  ) : (
                    <div className="flex-grow flex items-center justify-center text-zinc-400">Select a model to view detailed analysis.</div>
                  )}
                </div>
              </div>

            </div>

          </div>
        )}

        {/* ====================================================================== */}
        {/* TAB 3: LIVE DIAGNOSIS (INFERENCE) */}
        {/* ====================================================================== */}
        {activeTab === 'inference' && (
          <div className="space-y-6 animate-fadeIn">
            {/* Page Header */}
            <div>
              <h2 className="text-2xl font-bold tracking-tight">Live Diagnosis Form</h2>
              <p className="text-sm text-zinc-500 dark:text-zinc-400">
                Type the clinical attributes manually or pre-fill with a real random sample to test the models in real-time.
              </p>
            </div>

            {/* Predictor Panel Grid */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              
              {/* Form Input fields */}
              <div className="lg:col-span-2 bg-white dark:bg-[#0c0c0f] border border-zinc-200 dark:border-zinc-800 rounded-xl p-5 shadow-sm">
                
                {/* Form Toolbar presets */}
                <div className="flex flex-wrap items-center justify-between gap-3 mb-6 pb-4 border-b border-zinc-100 dark:border-zinc-800">
                  <div className="flex items-center gap-2">
                    <label className="text-xs font-semibold text-zinc-500 dark:text-zinc-400">Active Classifier:</label>
                    <select
                      value={predictorModel}
                      onChange={(e) => setPredictorModel(e.target.value)}
                      className="px-3 py-1.5 text-xs border border-zinc-200 dark:border-zinc-800 rounded-lg bg-white dark:bg-zinc-950 font-bold text-blue-600 dark:text-blue-400 focus:outline-none focus:ring-1 focus:ring-blue-500 shadow-sm"
                    >
                      <option value="KNN">K-Nearest Neighbors</option>
                      <option value="Logistic Regression">Logistic Regression</option>
                      <option value="SVM">Support Vector Machine</option>
                      <option value="Random Forest">Random Forest</option>
                      <option value="MLP">Multi-Layer Perceptron (NN)</option>
                    </select>
                  </div>

                  {/* Sample presets helpers */}
                  <div className="flex items-center gap-1.5">
                    <button
                      type="button"
                      onClick={() => loadRandomSample(0)}
                      className="px-2.5 py-1.5 text-[10px] font-bold border border-rose-200 dark:border-rose-950/40 rounded-lg bg-rose-50/50 dark:bg-rose-950/10 text-rose-600 dark:text-rose-400 hover:bg-rose-100 dark:hover:bg-rose-950/30 transition-all shadow-sm"
                    >
                      Prefill Malignant Sample
                    </button>
                    <button
                      type="button"
                      onClick={() => loadRandomSample(1)}
                      className="px-2.5 py-1.5 text-[10px] font-bold border border-emerald-200 dark:border-emerald-950/40 rounded-lg bg-emerald-50/50 dark:bg-emerald-950/10 text-emerald-600 dark:text-emerald-400 hover:bg-emerald-100 dark:hover:bg-emerald-950/30 transition-all shadow-sm"
                    >
                      Prefill Benign Sample
                    </button>
                  </div>
                </div>

                {/* Core Predictor Form */}
                <form onSubmit={handlePredict} className="space-y-6">
                  
                  {/* Top 10 Primary inputs grid */}
                  <div>
                    <h3 className="text-xs font-bold uppercase tracking-wider text-zinc-400 mb-3 font-mono">Primary Features (Highly Discriminative)</h3>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      {TOP_10_FEATURES.map((featureName) => {
                        const stats = edaStats.find(s => s.feature === featureName) || {};
                        return (
                          <div key={featureName} className="flex flex-col gap-1.5">
                            <label className="text-xs font-semibold capitalize text-zinc-700 dark:text-zinc-300 flex items-center justify-between">
                              <span>{featureName}</span>
                              <span className="text-[10px] font-mono text-zinc-400">Range: {stats.min?.toFixed(1)} - {stats.max?.toFixed(1)}</span>
                            </label>
                            <input
                              type="number"
                              step="any"
                              required
                              value={features[featureName] || ''}
                              onChange={(e) => setFeatures({...features, [featureName]: parseFloat(e.target.value) || 0})}
                              className="w-full bg-white dark:bg-[#09090b] border border-zinc-200 dark:border-zinc-800 rounded-lg px-3 py-2 text-xs font-mono text-zinc-900 dark:text-zinc-50 focus:outline-none focus:ring-2 focus:ring-blue-500/20 focus:border-blue-500 shadow-sm"
                            />
                          </div>
                        );
                      })}
                    </div>
                  </div>

                  {/* Collapsible toggle expander for remaining 20 features */}
                  <div className="border-t border-zinc-100 dark:border-zinc-800 pt-4">
                    <button
                      type="button"
                      onClick={() => setIsExpanderOpen(!isExpanderOpen)}
                      className="flex items-center justify-between w-full px-4 py-2 bg-zinc-50 dark:bg-zinc-900/30 rounded-lg text-xs font-semibold hover:bg-zinc-100 dark:hover:bg-zinc-900 transition-all text-zinc-600 dark:text-zinc-300"
                    >
                      <span className="flex items-center gap-2">
                        <HelpCircle className="w-4 h-4 text-zinc-400" />
                        {isExpanderOpen ? 'Hide' : 'View'} remaining 20 secondary clinical features (Pre-filled)
                      </span>
                      <span>{isExpanderOpen ? '▲' : '▼'}</span>
                    </button>

                    {isExpanderOpen && (
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4 p-4 border border-zinc-100 dark:border-zinc-800/80 rounded-xl bg-zinc-50/20 dark:bg-zinc-950/10">
                        {edaStats.filter(s => !TOP_10_FEATURES.includes(s.feature)).map((stats) => {
                          const featureName = stats.feature;
                          return (
                            <div key={featureName} className="flex flex-col gap-1.5">
                              <label className="text-xs font-semibold capitalize text-zinc-700 dark:text-zinc-300 flex items-center justify-between">
                                <span>{featureName}</span>
                                <span className="text-[10px] font-mono text-zinc-400">Mean: {stats.mean?.toFixed(2)}</span>
                              </label>
                              <input
                                type="number"
                                step="any"
                                required
                                value={features[featureName] || ''}
                                onChange={(e) => setFeatures({...features, [featureName]: parseFloat(e.target.value) || 0})}
                                className="w-full bg-white dark:bg-[#09090b] border border-zinc-200 dark:border-zinc-800 rounded-lg px-3 py-2 text-xs font-mono text-zinc-900 dark:text-zinc-50 focus:outline-none focus:ring-2 focus:ring-blue-500/20 focus:border-blue-500 shadow-sm"
                              />
                            </div>
                          );
                        })}
                      </div>
                    )}
                  </div>

                  <button
                    type="submit"
                    disabled={predicting}
                    className="w-full flex justify-center items-center gap-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg py-3 font-semibold text-sm transition-colors disabled:opacity-50 shadow-md"
                  >
                    {predicting ? (
                      <>
                        <RefreshCw className="w-4 h-4 animate-spin" />
                        Running classification model inference...
                      </>
                    ) : (
                      <>
                        <Play className="w-4 h-4 fill-current" />
                        Diagnose Features
                      </>
                    )}
                  </button>
                </form>
              </div>

              {/* Inference diagnosis visual cards output */}
              <div className="space-y-4">
                <div className="bg-white dark:bg-[#0c0c0f] border border-zinc-200 dark:border-zinc-800 rounded-xl p-5 shadow-sm h-[400px] flex flex-col justify-center items-center text-center">
                  
                  {predictionError && (
                    <div className="space-y-3 p-4 bg-rose-50 dark:bg-rose-950/20 border border-rose-200 dark:border-rose-900/30 rounded-xl text-rose-600 dark:text-rose-400">
                      <AlertCircle className="w-8 h-8 mx-auto" />
                      <h4 className="text-sm font-bold">Prediction Failed</h4>
                      <p className="text-xs">{predictionError}</p>
                    </div>
                  )}

                  {!predictionResult && !predictionError && (
                    <div className="space-y-3 text-zinc-400 max-w-xs">
                      <Sliders className="w-12 h-12 mx-auto stroke-1" />
                      <h4 className="text-sm font-bold text-zinc-600 dark:text-zinc-300">Awaiting Attributes</h4>
                      <p className="text-xs">
                        Enter values on the left and submit the diagnosis query to evaluate benign vs malignant classification.
                      </p>
                    </div>
                  )}

                  {predictionResult && (
                    <div className="space-y-6 w-full animate-fadeIn">
                      <div className="space-y-2">
                        <div className="text-[10px] uppercase font-bold text-zinc-400 tracking-widest font-mono">Classification Result</div>
                        <div className={`text-3xl font-black ${
                          predictionResult.prediction === 0 ? 'text-rose-600' : 'text-emerald-600'
                        }`}>
                          {predictionResult.class_label}
                        </div>
                        <p className="text-[10px] text-zinc-500 font-mono">using {predictionResult.model_used}</p>
                      </div>

                      {/* Probability Gauge Mock */}
                      <div className="space-y-2 max-w-xs mx-auto">
                        <div className="flex justify-between text-xs font-semibold text-zinc-500">
                          <span>Malignant</span>
                          <span>Benign</span>
                        </div>
                        
                        {/* Progress Bar */}
                        <div className="w-full h-3.5 bg-zinc-100 dark:bg-zinc-800 rounded-full overflow-hidden flex shadow-inner">
                          <div 
                            style={{ width: `${predictionResult.probabilities.Malignant}%` }}
                            className="bg-rose-500 transition-all duration-500" 
                          />
                          <div 
                            style={{ width: `${predictionResult.probabilities.Benign}%` }}
                            className="bg-emerald-500 transition-all duration-500" 
                          />
                        </div>

                        <div className="flex justify-between text-[10px] font-mono text-zinc-400">
                          <span>{predictionResult.probabilities.Malignant}%</span>
                          <span>{predictionResult.probabilities.Benign}%</span>
                        </div>
                      </div>

                      {/* Feedback banner */}
                      <div className={`p-4 rounded-xl text-xs flex gap-3 text-left ${
                        predictionResult.prediction === 0
                          ? 'bg-rose-50 dark:bg-rose-950/20 border border-rose-100 dark:border-rose-900/30 text-rose-800 dark:text-rose-300'
                          : 'bg-emerald-50 dark:bg-emerald-950/20 border border-emerald-100 dark:border-emerald-900/30 text-emerald-800 dark:text-emerald-300'
                      }`}>
                        <AlertCircle className="w-5 h-5 shrink-0" />
                        <div>
                          <strong className="font-bold block mb-1">
                            {predictionResult.prediction === 0 ? 'Urgent Review Advised' : 'No Immediate Malignancy Detected'}
                          </strong>
                          {predictionResult.prediction === 0 
                            ? 'High probability of malignancy. Features demonstrate standard visual properties of aggressive cell masses.' 
                            : 'Features fall within typical baseline boundaries of benign non-cancerous masses. Continue standard checkups.'}
                        </div>
                      </div>

                    </div>
                  )}

                </div>
              </div>

            </div>

          </div>
        )}

      </main>

      {/* ─── Footer ──────────────────────────────────────────────────────────── */}
      <footer className="border-t border-zinc-200 dark:border-zinc-800 py-6 bg-white dark:bg-[#0c0c0f] text-center text-xs text-zinc-500">
        <div className="max-w-[1600px] mx-auto px-6 flex flex-col sm:flex-row items-center justify-between gap-4">
          <span className="font-mono">Wisconsin Diagnostic Breast Cancer Classification Systems</span>
          <span className="text-[10px] px-2 py-1 bg-zinc-100 dark:bg-zinc-900 rounded">Phase 1B — Active Run</span>
        </div>
      </footer>
    </div>
  );
}

export default App;
