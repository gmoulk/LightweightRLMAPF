import React, { useState, useEffect } from 'react';

export default function App() {
  const [theme, setTheme] = useState('dark'); // 'dark' or 'light'
  const [trainingActive, setTrainingActive] = useState(true);
  const [loading, setLoading] = useState(false);
  const [mode, setMode] = useState('random'); // 'random' or 'custom'
  
  // Random Map Settings
  const [params, setParams] = useState({
    num_agents: 4,
    size: 16,
    density: 0.1,
    max_steps: 128,
  });

  // Custom Map Drawing State
  const [gridSize, setGridSize] = useState(8);
  const [selectedTool, setSelectedTool] = useState('obstacle'); // 'obstacle', 'agent', 'goal'
  const [customGrid, setCustomGrid] = useState([]); // 0: empty, 1: obstacle
  const [agents, setAgents] = useState([]); // [{ x, y }]
  const [goals, setGoals] = useState([]);   // [{ x, y }]

  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  // Dynamic Theme Styling Tokens
  const themeStyles = {
    dark: {
      bg: '#0F172A',
      cardBg: '#1E293B',
      border: '#334155',
      text: '#F8FAFC',
      subtext: '#94A3B8',
      inputBg: '#0F172A',
      tabBg: '#0F172A',
      cellEmpty: '#0F172A',
      cellWall: '#334155',
      cellBorder: '#334155',
      primaryBtn: '#3B82F6',
      primaryBtnText: '#FFFFFF',
      disabledBtn: '#334155',
      disabledText: '#64748B',
      errorBg: '#450A0A',
      errorBorder: '#991B1B',
      errorText: '#FCA5A5',
      statusTrainBg: '#312E81',
      statusTrainText: '#A5B4FC',
      statusReadyBg: '#064E3B',
      statusReadyText: '#6EE7B7',
    },
    light: {
      bg: '#F8FAFC',
      cardBg: '#FFFFFF',
      border: '#E2E8F0',
      text: '#0F172A',
      subtext: '#64748B',
      inputBg: '#F1F5F9',
      tabBg: '#F1F5F9',
      cellEmpty: '#FFFFFF',
      cellWall: '#94A3B8',
      cellBorder: '#CBD5E1',
      primaryBtn: '#2563EB',
      primaryBtnText: '#FFFFFF',
      disabledBtn: '#E2E8F0',
      disabledText: '#94A3B8',
      errorBg: '#FEF2F2',
      errorBorder: '#FCA5A5',
      errorText: '#991B1B',
      statusTrainBg: '#E0E7FF',
      statusTrainText: '#3730A3',
      statusReadyBg: '#D1FAE5',
      statusReadyText: '#065F46',
    }
  };

  const t = themeStyles[theme];

  // Initialize/Resize Custom Grid Matrix
  useEffect(() => {
    const newGrid = Array(gridSize).fill(0).map(() => Array(gridSize).fill(0));
    setCustomGrid(newGrid);
    setAgents([]);
    setGoals([]);
  }, [gridSize]);

  // Poll Backend Startup State
  useEffect(() => {
    const checkStatus = async () => {
      try {
        const res = await fetch('http://localhost:8000/');
        const data = await res.json();
        setTrainingActive(data.training_active);
      } catch (err) {
        setError('Backend server is offline. Run main.py first.');
      }
    };

    checkStatus();
    const interval = setInterval(checkStatus, 3000);
    return () => clearInterval(interval);
  }, []);

  const handleChange = (e) => {
    setParams({ ...params, [e.target.name]: parseFloat(e.target.value) });
  };

  // Grid Cell Click Logic
  const handleCellClick = (r, c) => {
    if (selectedTool === 'obstacle') {
      const newGrid = customGrid.map((row, ri) =>
        row.map((val, ci) => (ri === r && ci === c ? (val === 1 ? 0 : 1) : val))
      );
      setCustomGrid(newGrid);
      setAgents(agents.filter((a) => !(a.x === r && a.y === c)));
      setGoals(goals.filter((g) => !(g.x === r && g.y === c)));
    } else if (selectedTool === 'agent') {
      if (customGrid[r][c] === 1) return;
      const exists = agents.findIndex((a) => a.x === r && a.y === c);
      if (exists !== -1) {
        setAgents(agents.filter((_, idx) => idx !== exists));
      } else {
        setAgents([...agents, { x: r, y: c }]);
      }
    } else if (selectedTool === 'goal') {
      if (customGrid[r][c] === 1) return;
      const exists = goals.findIndex((g) => g.x === r && g.y === c);
      if (exists !== -1) {
        setGoals(goals.filter((_, idx) => idx !== exists));
      } else {
        setGoals([...goals, { x: r, y: c }]);
      }
    }
  };

  const handleSolve = async () => {
    setLoading(true);
    setError(null);

    if (mode === 'custom') {
      if (agents.length === 0) {
        setError('Please place at least one agent on the grid.');
        setLoading(false);
        return;
      }
      if (agents.length !== goals.length) {
        setError(`Agent count (${agents.length}) must match Goal count (${goals.length}).`);
        setLoading(false);
        return;
      }
    }

    const payload = mode === 'random' 
      ? params 
      : {
          custom_map: customGrid,
          agents: agents,
          goals: goals,
          max_steps: params.max_steps
        };

    try {
      const res = await fetch('http://localhost:8000/api/v1/solve', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.detail || 'Failed to generate solution');
      }

      const data = await res.json();
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ ...styles.container, backgroundColor: t.bg, color: t.text }}>
      <div style={{ ...styles.card, backgroundColor: t.cardBg, borderColor: t.border }}>
        
        {/* Header & Theme Toggle Switch */}
        <header style={styles.header}>
          <div style={styles.headerTop}>
            <h1 style={{ ...styles.title, color: t.text }}>MAPF RL Visualizer</h1>
            <button 
              onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
              style={{ ...styles.themeToggle, backgroundColor: t.inputBg, borderColor: t.border, color: t.text }}
            >
              {theme === 'dark' ? '☀️ Light Mode' : '🌙 Dark Mode'}
            </button>
          </div>
          <p style={{ ...styles.subtitle, color: t.subtext }}>Multi-Agent Pathfinding Powered by Communication Transformers</p>
        </header>

        {/* Status Indicator Banner */}
        <div style={{ 
          ...styles.statusBanner, 
          backgroundColor: trainingActive ? t.statusTrainBg : t.statusReadyBg,
          color: trainingActive ? t.statusTrainText : t.statusReadyText
        }}>
          <span style={styles.statusDot(trainingActive)}></span>
          <strong>Status: </strong> 
          {trainingActive ? ' Model is training in background...' : ' Pretrained model ready'}
        </div>

        {/* Mode Selector Tabs */}
        <div style={styles.tabs}>
          <button 
            style={mode === 'random' ? styles.activeTab(t) : styles.tab(t)} 
            onClick={() => setMode('random')}
          >
            Random Generator
          </button>
          <button 
            style={mode === 'custom' ? styles.activeTab(t) : styles.tab(t)} 
            onClick={() => setMode('custom')}
          >
            Custom Map Designer
          </button>
        </div>

        {/* Random Map Controls Form */}
        {mode === 'random' && (
          <div style={styles.gridParams}>
            <div style={styles.inputGroup}>
              <label style={{ ...styles.label, color: t.subtext }}>Agents</label>
              <input type="number" name="num_agents" value={params.num_agents} onChange={handleChange} style={{ ...styles.input(t) }} />
            </div>
            <div style={styles.inputGroup}>
              <label style={{ ...styles.label, color: t.subtext }}>Grid Size ($N \times N$)</label>
              <input type="number" name="size" value={params.size} onChange={handleChange} style={{ ...styles.input(t) }} />
            </div>
            <div style={styles.inputGroup}>
              <label style={{ ...styles.label, color: t.subtext }}>Obstacle Density</label>
              <input type="number" step="0.05" name="density" value={params.density} onChange={handleChange} style={{ ...styles.input(t) }} />
            </div>
            <div style={styles.inputGroup}>
              <label style={{ ...styles.label, color: t.subtext }}>Max Steps</label>
              <input type="number" name="max_steps" value={params.max_steps} onChange={handleChange} style={{ ...styles.input(t) }} />
            </div>
          </div>
        )}

        {/* Custom Map Designer Canvas */}
        {mode === 'custom' && (
          <div style={styles.designerContainer}>
            <div style={styles.toolbar}>
              <div style={styles.inputGroup}>
                <label style={{ ...styles.label, color: t.subtext }}>Map Dimension</label>
                <input 
                  type="number" 
                  min="4" 
                  max="32" 
                  value={gridSize} 
                  onChange={(e) => setGridSize(parseInt(e.target.value) || 8)} 
                  style={{ ...styles.input(t) }} 
                />
              </div>

              {/* Tool Selector Controls */}
              <div style={styles.tools}>
                <button 
                  style={selectedTool === 'obstacle' ? styles.activeTool : styles.tool(t)} 
                  onClick={() => setSelectedTool('obstacle')}
                >
                  Wall 🧱
                </button>
                <button 
                  style={selectedTool === 'agent' ? styles.activeTool : styles.tool(t)} 
                  onClick={() => setSelectedTool('agent')}
                >
                  Agent 🤖 ({agents.length})
                </button>
                <button 
                  style={selectedTool === 'goal' ? styles.activeTool : styles.tool(t)} 
                  onClick={() => setSelectedTool('goal')}
                >
                  Goal 🎯 ({goals.length})
                </button>
              </div>
            </div>

            {/* Interactive Grid Canvas */}
            <div style={{ ...styles.canvasWrapper, backgroundColor: t.inputBg, borderColor: t.border }}>
              <div 
                style={{
                  ...styles.canvas,
                  backgroundColor: t.cellBorder,
                  gridTemplateColumns: `repeat(${gridSize}, 32px)`,
                  gridTemplateRows: `repeat(${gridSize}, 32px)`
                }}
              >
                {customGrid.map((row, r) =>
                  row.map((cell, c) => {
                    const agentIdx = agents.findIndex((a) => a.x === r && a.y === c);
                    const goalIdx = goals.findIndex((g) => g.x === r && g.y === c);

                    return (
                      <div
                        key={`${r}-${c}`}
                        onClick={() => handleCellClick(r, c)}
                        style={{
                          ...styles.cell,
                          backgroundColor: cell === 1 ? t.cellWall : t.cellEmpty,
                        }}
                      >
                        {agentIdx !== -1 && <span style={styles.agentTag}>A{agentIdx}</span>}
                        {goalIdx !== -1 && <span style={styles.goalTag}>G{goalIdx}</span>}
                      </div>
                    );
                  })
                )}
              </div>
            </div>
          </div>
        )}

        <button 
          onClick={handleSolve} 
          disabled={trainingActive || loading} 
          style={trainingActive || loading ? styles.buttonDisabled(t) : styles.button(t)}
        >
          {loading ? 'Solving Custom Environment...' : 'Solve Map & Render SVG'}
        </button>

        {error && (
          <div style={{ ...styles.errorBox, backgroundColor: t.errorBg, borderColor: t.errorBorder, color: t.errorText }}>
            {error}
          </div>
        )}

        {/* Results Metrics & SVG Rendering Section */}
        {result && (
          <div style={{ ...styles.resultsContainer, borderColor: t.border }}>
            <h2 style={{ ...styles.sectionTitle, color: t.text }}>Evaluation Output</h2>
            <div style={styles.metricsGrid}>
              <div style={{ ...styles.metricCard, backgroundColor: t.inputBg, borderColor: t.border }}>
                <span style={{ ...styles.metricLabel, color: t.subtext }}>Solved Status</span>
                <span style={{ ...styles.metricValue, color: result.success ? '#10B981' : '#EF4444' }}>
                  {result.success ? 'Success' : 'Failed'}
                </span>
              </div>
              <div style={{ ...styles.metricCard, backgroundColor: t.inputBg, borderColor: t.border }}>
                <span style={{ ...styles.metricLabel, color: t.subtext }}>Steps Taken</span>
                <span style={{ ...styles.metricValue, color: t.text }}>{result.num_steps}</span>
              </div>
            </div>

            {result.svg_animation && (
              <div style={{ ...styles.svgWrapper, backgroundColor: t.inputBg, borderColor: t.border }}>
                <h3 style={{ ...styles.svgTitle, color: t.subtext }}>Pogema Real-time Visualizer</h3>
                <div style={styles.svgContent} dangerouslySetInnerHTML={{ __html: result.svg_animation }} />
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

// React CSS Styles Definition Object
const styles = {
  container: { minHeight: '100vh', fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif', display: 'flex', justifyContent: 'center', padding: '40px 20px', transition: 'all 0.2s ease' },
  card: { width: '100%', maxWidth: '850px', borderRadius: '16px', padding: '32px', border: '1px solid', boxShadow: '0 10px 25px -5px rgba(0, 0, 0, 0.05)', transition: 'all 0.2s ease' },
  header: { marginBottom: '20px' },
  headerTop: { display: 'flex', justifyContent: 'space-between', alignItems: 'center' },
  title: { fontSize: '26px', margin: 0, fontWeight: '700' },
  themeToggle: { padding: '6px 12px', border: '1px solid', borderRadius: '8px', cursor: 'pointer', fontSize: '13px', fontWeight: '600' },
  subtitle: { fontSize: '14px', margin: '8px 0 0 0' },
  statusBanner: { display: 'flex', alignItems: 'center', gap: '10px', padding: '12px 16px', borderRadius: '8px', marginBottom: '20px', fontSize: '14px' },
  statusDot: (isTraining) => ({ width: '8px', height: '8px', borderRadius: '50%', backgroundColor: isTraining ? '#818CF8' : '#34D399' }),
  tabs: { display: 'flex', gap: '10px', marginBottom: '20px' },
  tab: (t) => ({ flex: 1, padding: '10px', backgroundColor: t.tabBg, color: t.subtext, border: `1px solid ${t.border}`, borderRadius: '8px', cursor: 'pointer' }),
  activeTab: (t) => ({ flex: 1, padding: '10px', backgroundColor: t.primaryBtn, color: t.primaryBtnText, border: 'none', borderRadius: '8px', cursor: 'pointer', fontWeight: 'bold' }),
  gridParams: { display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '16px', marginBottom: '20px' },
  designerContainer: { marginBottom: '20px' },
  toolbar: { display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' },
  tools: { display: 'flex', gap: '8px' },
  tool: (t) => ({ padding: '8px 12px', backgroundColor: t.inputBg, border: `1px solid ${t.border}`, color: t.text, borderRadius: '6px', cursor: 'pointer' }),
  activeTool: { padding: '8px 12px', backgroundColor: '#10B981', border: 'none', color: '#FFF', borderRadius: '6px', cursor: 'pointer', fontWeight: 'bold' },
  canvasWrapper: { display: 'flex', justifyContent: 'center', padding: '16px', borderRadius: '8px', border: '1px solid', overflowX: 'auto' },
  canvas: { display: 'grid', gap: '1px', padding: '1px', borderRadius: '4px' },
  cell: { width: '32px', height: '32px', display: 'flex', alignItems: 'center', justifyContent: 'center', cursor: 'pointer', fontSize: '11px', fontWeight: 'bold', userSelect: 'none' },
  agentTag: { color: '#FFFFFF', backgroundColor: '#2563EB', padding: '2px 4px', borderRadius: '4px' },
  goalTag: { color: '#FFFFFF', backgroundColor: '#E11D48', padding: '2px 4px', borderRadius: '4px' },
  inputGroup: { display: 'flex', flexDirection: 'column', gap: '6px' },
  label: { fontSize: '12px', textTransform: 'uppercase', fontWeight: '600' },
  input: (t) => ({ backgroundColor: t.inputBg, border: `1px solid ${t.border}`, borderRadius: '8px', color: t.text, padding: '8px', fontSize: '14px', outline: 'none' }),
  button: (t) => ({ width: '100%', padding: '14px', backgroundColor: t.primaryBtn, color: t.primaryBtnText, fontWeight: '600', border: 'none', borderRadius: '8px', cursor: 'pointer' }),
  buttonDisabled: (t) => ({ width: '100%', padding: '14px', backgroundColor: t.disabledBtn, color: t.disabledText, border: 'none', borderRadius: '8px', cursor: 'not-allowed' }),
  errorBox: { marginTop: '16px', padding: '12px', border: '1px solid', borderRadius: '8px', fontSize: '14px' },
  resultsContainer: { marginTop: '24px', borderTop: '1px solid', paddingTop: '20px' },
  sectionTitle: { fontSize: '18px', margin: '0 0 16px 0' },
  metricsGrid: { display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px', marginBottom: '20px' },
  metricCard: { padding: '16px', borderRadius: '8px', border: '1px solid', textAlign: 'center' },
  metricLabel: { fontSize: '12px' },
  metricValue: { fontSize: '20px', fontWeight: '700', marginTop: '4px', display: 'block' },
  svgWrapper: { borderRadius: '12px', padding: '20px', border: '1px solid', textAlign: 'center' },
  svgTitle: { fontSize: '14px', marginBottom: '16px', marginTop: 0 },
  svgContent: { display: 'flex', justifyContent: 'center' }
};