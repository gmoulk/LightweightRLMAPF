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

  // Poll backend startup state
  useEffect(() => {
    const checkStatus = async () => {
      try {
        const res = await fetch('http://localhost:8000/');
        const data = await res.json();
        setTrainingActive(data.training_active);
      } catch (err) {
        setError('Backend is offline. Start main.py first.');
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
        throw new Error(errData.detail || 'Failed to solve');
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
    <div style={{ fontFamily: 'sans-serif', maxWidth: '900px', margin: '0 auto', padding: '20px' }}>
      <h2>MAPF Solver Visualizer</h2>
      
      <div style={{ padding: '10px', background: trainingActive ? '#fff3cd' : '#d4edda', borderRadius: '5px', marginBottom: '20px' }}>
        <strong>Status: </strong> 
        {trainingActive ? 'Model is Training... (Please wait)' : 'Model Ready'}
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '15px', marginBottom: '20px' }}>
        <label>
          Agents:
          <input type="number" name="num_agents" value={params.num_agents} onChange={handleChange} style={{ width: '100%', padding: '8px' }} />
        </label>
        <label>
          Grid Size:
          <input type="number" name="size" value={params.size} onChange={handleChange} style={{ width: '100%', padding: '8px' }} />
        </label>
        <label>
          Obstacle Density (0 - 0.5):
          <input type="number" step="0.05" name="density" value={params.density} onChange={handleChange} style={{ width: '100%', padding: '8px' }} />
        </label>
        <label>
          Max Steps:
          <input type="number" name="max_steps" value={params.max_steps} onChange={handleChange} style={{ width: '100%', padding: '8px' }} />
        </label>
      </div>

      <button 
        onClick={handleSolve} 
        disabled={trainingActive || loading} 
        style={{ width: '100%', padding: '12px', background: '#007bff', color: '#fff', border: 'none', borderRadius: '5px', cursor: 'pointer' }}
      >
        {loading ? 'Solving...' : 'Solve Map & Generate SVG'}
      </button>

      {error && <p style={{ color: 'red', marginTop: '15px' }}>{error}</p>}

      {result && (
        <div style={{ marginTop: '30px' }}>
          <h3>Solution Summary</h3>
          <p><strong>Success:</strong> {result.success ? 'Yes' : 'No'}</p>
          <p><strong>Steps Taken:</strong> {result.num_steps}</p>
          
          {result.svg_animation && (
            <div style={{ marginTop: '20px', border: '1px solid #ccc', padding: '10px', borderRadius: '5px' }}>
              <h3>Pogema SVG Animation</h3>
              <div dangerouslySetInnerHTML={{ __html: result.svg_animation }} />
            </div>
          )}
        </div>
      )}
    </div>
  );
}