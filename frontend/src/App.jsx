import React, { useState, useEffect } from 'react';

export default function App() {
  const [trainingActive, setTrainingActive] = useState(true);
  const [loading, setLoading] = useState(false);
  const [params, setParams] = useState({
    num_agents: 4,
    size: 16,
    density: 0.1,
    max_steps: 128,
  });
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

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

  const handleSolve = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch('http://localhost:8000/api/v1/solve', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(params),
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
    <div style={styles.container}>
      <div style={styles.card}>
        {/* Header */}
        <header style={styles.header}>
          <h1 style={styles.title}>MAPF RL Visualizer</h1>
          <p style={styles.subtitle}>Multi-Agent Pathfinding Powered by Communication Transformers</p>
        </header>

        {/* Status Indicator Banner */}
        <div style={trainingActive ? styles.statusTraining : styles.statusReady}>
          <span style={styles.statusDot(trainingActive)}></span>
          <strong>Status: </strong> 
          {trainingActive ? ' Model is training in background...' : ' Pretrained model ready'}
        </div>

        {/* Input Controls Form */}
        <div style={styles.grid}>
          <div style={styles.inputGroup}>
            <label style={styles.label}>Number of Agents</label>
            <input 
              type="number" 
              name="num_agents" 
              value={params.num_agents} 
              onChange={handleChange} 
              style={styles.input} 
            />
          </div>

          <div style={styles.inputGroup}>
            <label style={styles.label}>Grid Size ($N \times N$)</label>
            <input 
              type="number" 
              name="size" 
              value={params.size} 
              onChange={handleChange} 
              style={styles.input} 
            />
          </div>

          <div style={styles.inputGroup}>
            <label style={styles.label}>Obstacle Density (0.0 - 0.5)</label>
            <input 
              type="number" 
              step="0.05" 
              name="density" 
              value={params.density} 
              onChange={handleChange} 
              style={styles.input} 
            />
          </div>

          <div style={styles.inputGroup}>
            <label style={styles.label}>Max Steps Horizon</label>
            <input 
              type="number" 
              name="max_steps" 
              value={params.max_steps} 
              onChange={handleChange} 
              style={styles.input} 
            />
          </div>
        </div>

        {/* Action Button */}
        <button 
          onClick={handleSolve} 
          disabled={trainingActive || loading} 
          style={trainingActive || loading ? styles.buttonDisabled : styles.button}
        >
          {loading ? 'Running Inference & Rendering...' : 'Solve Map & Render SVG'}
        </button>

        {/* Error Messaging */}
        {error && <div style={styles.errorBox}>{error}</div>}

        {/* Evaluation Output Section */}
        {result && (
          <div style={styles.resultsContainer}>
            <h2 style={styles.sectionTitle}>Evaluation Output</h2>

            <div style={styles.metricsGrid}>
              <div style={styles.metricCard}>
                <span style={styles.metricLabel}>Solved Status</span>
                <span style={{
                  ...styles.metricValue, 
                  color: result.success ? '#10B981' : '#EF4444'
                }}>
                  {result.success ? 'Success' : 'Failed'}
                </span>
              </div>

              <div style={styles.metricCard}>
                <span style={styles.metricLabel}>Time Steps</span>
                <span style={styles.metricValue}>{result.num_steps}</span>
              </div>

              <div style={styles.metricCard}>
                <span style={styles.metricLabel}>Total Agents</span>
                <span style={styles.metricValue}>{result.paths.length}</span>
              </div>
            </div>

            {/* Pogema Native SVG Frame */}
            {result.svg_animation && (
              <div style={styles.svgWrapper}>
                <h3 style={styles.svgTitle}>Pogema Real-time Visualizer</h3>
                <div 
                  style={styles.svgContent}
                  dangerouslySetInnerHTML={{ __html: result.svg_animation }} 
                />
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

// Styling definitions
const styles = {
  container: {
    minHeight: '100vh',
    backgroundColor: '#0F172A',
    color: '#F8FAFC',
    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
    display: 'flex',
    justifyContent: 'center',
    padding: '40px 20px',
  },
  card: {
    width: '100%',
    maxWidth: '850px',
    backgroundColor: '#1E293B',
    borderRadius: '16px',
    padding: '32px',
    boxShadow: '0 20px 25px -5px rgba(0, 0, 0, 0.5), 0 10px 10px -5px rgba(0, 0, 0, 0.04)',
    border: '1px solid #334155',
  },
  header: {
    marginBottom: '24px',
    textAlign: 'center',
  },
  title: {
    fontSize: '28px',
    fontWeight: '700',
    letterSpacing: '-0.5px',
    color: '#F8FAFC',
    margin: '0 0 8px 0',
  },
  subtitle: {
    fontSize: '14px',
    color: '#94A3B8',
    margin: 0,
  },
  statusTraining: {
    display: 'flex',
    alignItems: 'center',
    gap: '10px',
    padding: '12px 16px',
    backgroundColor: '#312E81',
    color: '#A5B4FC',
    borderRadius: '8px',
    fontSize: '14px',
    marginBottom: '24px',
    border: '1px solid #4338CA',
  },
  statusReady: {
    display: 'flex',
    alignItems: 'center',
    gap: '10px',
    padding: '12px 16px',
    backgroundColor: '#064E3B',
    color: '#6EE7B7',
    borderRadius: '8px',
    fontSize: '14px',
    marginBottom: '24px',
    border: '1px solid #047857',
  },
  statusDot: (isTraining) => ({
    width: '8px',
    height: '8px',
    borderRadius: '50%',
    backgroundColor: isTraining ? '#818CF8' : '#34D399',
    boxShadow: isTraining ? '0 0 8px #818CF8' : '0 0 8px #34D399',
  }),
  grid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))',
    gap: '16px',
    marginBottom: '24px',
  },
  inputGroup: {
    display: 'flex',
    flexDirection: 'column',
    gap: '6px',
  },
  label: {
    fontSize: '12px',
    fontWeight: '600',
    color: '#94A3B8',
    textTransform: 'uppercase',
    letterSpacing: '0.5px',
  },
  input: {
    backgroundColor: '#0F172A',
    border: '1px solid #334155',
    borderRadius: '8px',
    color: '#F8FAFC',
    padding: '10px 12px',
    fontSize: '14px',
    outline: 'none',
  },
  button: {
    width: '100%',
    padding: '14px',
    backgroundColor: '#3B82F6',
    color: '#FFFFFF',
    fontWeight: '600',
    border: 'none',
    borderRadius: '8px',
    cursor: 'pointer',
    fontSize: '15px',
    transition: 'background-color 0.2s ease',
  },
  buttonDisabled: {
    width: '100%',
    padding: '14px',
    backgroundColor: '#334155',
    color: '#64748B',
    fontWeight: '600',
    border: 'none',
    borderRadius: '8px',
    cursor: 'not-allowed',
    fontSize: '15px',
  },
  errorBox: {
    marginTop: '16px',
    padding: '12px',
    backgroundColor: '#450A0A',
    border: '1px solid #991B1B',
    color: '#FCA5A5',
    borderRadius: '8px',
    fontSize: '14px',
  },
  resultsContainer: {
    marginTop: '32px',
    borderTop: '1px solid #334155',
    paddingTop: '24px',
  },
  sectionTitle: {
    fontSize: '18px',
    fontWeight: '600',
    marginBottom: '16px',
    color: '#F8FAFC',
  },
  metricsGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(3, 1fr)',
    gap: '12px',
    marginBottom: '24px',
  },
  metricCard: {
    backgroundColor: '#0F172A',
    padding: '16px',
    borderRadius: '8px',
    border: '1px solid #334155',
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
  },
  metricLabel: {
    fontSize: '12px',
    color: '#94A3B8',
    marginBottom: '4px',
  },
  metricValue: {
    fontSize: '20px',
    fontWeight: '700',
  },
  svgWrapper: {
    backgroundColor: '#0F172A',
    borderRadius: '12px',
    padding: '20px',
    border: '1px solid #334155',
    textAlign: 'center',
  },
  svgTitle: {
    fontSize: '14px',
    color: '#94A3B8',
    marginTop: 0,
    marginBottom: '16px',
  },
  svgContent: {
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'center',
    overflow: 'hidden',
  }
};