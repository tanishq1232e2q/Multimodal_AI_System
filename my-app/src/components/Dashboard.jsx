
import { useState, useEffect } from 'react';
import { Search, Mic, Brain, MapPin, X, Globe } from 'lucide-react';
import "../dashbord.css"

// Simple sentiment detection (for positive English text)
// const isPositiveSentence = (text: string): boolean => {
//   const positives = [
//     "good", "great", "happy", "fine", "okay", "awesome", "excellent",
//     "nice", "positive", "better", "grateful", "joy", "love", "relaxed",
//   ];
//   const lowered = text.toLowerCase();
//   return positives.some(word => lowered.includes(word));
// };

export default function Dashboard() {
  const [text, setText] = useState('');
  const [audio, setAudio] = useState(null);
  const [eeg, setEeg] = useState(null);
  const [fips, setFips] = useState('');
  const [predictions, setPredictions] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [languageMode, setLanguageMode] = useState('english');

const [graphData, setGraphData] = useState({
    text: 0,
  
    eeg: 0,
    spatial: 0,
  });
  // Lightweight language check: English vs Non-English
  useEffect(() => {
    if (!text.trim()) {
      setLanguageMode('english');
      return;
    }

    const spanishAlphabetPattern = /[áéíóúüñÁÉÍÓÚÜÑ¡¿]/;
    const commonSpanishWords = /\b(el|la|de|que|y|en|a|los|las|me|te|porque|vida|feliz|jugar|gustar|uno|una|pero|por|para|todo|muy)\b/i;

    // If all characters are English letters, numbers, and punctuation → English
    const isEnglish = /^[A-Za-z0-9\s.,!?'"-]+$/.test(text) && !spanishAlphabetPattern.test(text) && !commonSpanishWords.test(text);
    setLanguageMode(isEnglish ? 'english' : 'other');

   
  }, [text]);


  useEffect(() => {
  const saved = localStorage.getItem("mental_graph");
  if (saved) setGraphData(JSON.parse(saved));
}, []);



  useEffect(() => {
  if (!predictions) return;

  setGraphData((prev) => {
    const updated = {
      text: prev.text + (predictions.text_score || 0),
      eeg: prev.eeg + (predictions.eeg_score || 0),
      spatial: prev.spatial + (predictions.spatial_score || 0),
    };

    localStorage.setItem("mental_graph", JSON.stringify(updated));
    return updated;
  });
}, [predictions]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!text && !audio && !eeg && !fips) return;

    setLoading(true);
    setError('');
    setPredictions(null);

    const formData = new FormData();

    // ----- NEW -----
    if (text) {
      const key = languageMode === 'english' ? 'text' : 'multilingual_text';
      formData.append(key, text);
    }
    // --------------

    if (audio) formData.append('audio', audio);
    if (eeg) formData.append('eeg', eeg);
    if (fips) formData.append('fips_code', fips);

    try {
      const res = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        body: formData,
      });
      const data = await res.json();
      if (data.error) throw new Error(data.error);
      setPredictions(data.predictions);
    } catch (err) {
      setError(err.message || 'Prediction failed');
    } finally {
      setLoading(false);
    }
  };

  const removeAudio = () => setAudio(null);
  const removeEeg = () => setEeg(null);
  const hasInput = text || audio || eeg || fips;
  


  return (

    <div >


      <div className="dashboard-container">
        <div className="dashboard-box">
          <h1 className="dashboard-title">
            Multimodal Mental Disorder Prediction System
          </h1>

          <form onSubmit={handleSubmit}>
            <div className="input-group">
              <input
                type="text"
                value={text}
                onChange={(e) => setText(e.target.value)}
                placeholder="Type your thoughts (Any Language)..."
                className="text-input"
              />
              <button type="submit" className="submit-btn" disabled={loading}>
                🔍
              </button>
            </div>

            <div style={{ display: "flex", justifyContent: "center", alignItems: "center" }} className="file-section">
              {/* Audio */}
              <label className="file-card">
                🎤 Audio
                <input
                  type="file"
                  accept=".wav,.mp3"
                  onChange={(e) => setAudio(e.target.files?.[0] || null)}
                />
                <span>{audio ? audio.name : "Upload Audio File"}</span>
                {audio && (
                  <button
                    type="button"
                    className="remove-btn"
                    onClick={() => setAudio(null)}
                  >
                    ✖
                  </button>
                )}
              </label>

              {/* EEG */}
              <label className="file-card">
                🧠 EEG
                <input
                  type="file"
                  accept=".mat"
                  onChange={(e) => setEeg(e.target.files?.[0] || null)}
                />
                <span>{eeg ? eeg.name : "Upload EEG File"}</span>
                {eeg && (
                  <button
                    type="button"
                    className="remove-btn"
                    onClick={() => setEeg(null)}
                  >
                    ✖
                  </button>
                )}
              </label>

              {/* FIPS */}
              <div className="file-card">
                📍 FIPS Code
                <input
                  type="text"
                  value={fips}
                  onChange={(e) => setFips(e.target.value)}
                  placeholder="Enter FIPS Code"
                />
              </div>
            </div>

            {loading && (
              <div className="loading">
                <div className="spinner"></div>
                <p>Analyzing...</p>
              </div>
            )}
          </form>

          {predictions && (
            <div className="result-box">
              <h2>AI Analysis</h2>

              {languageMode === 'english' && predictions.text && (
                <div className="result-card">
                  <strong>Text Analysis:</strong> {predictions.text}
                  <div>Score: {predictions.text_score}</div>
                </div>
              )}

              {languageMode !== 'english' && predictions.multilingual && (
                <div className="result-card">
                  <strong>Multilingual Text:</strong> {predictions.multilingual}
                  {/* <div>Score: {predictions.test_score}</div> */}
                </div>
              )}

              {predictions.audio && (
                <div className="result-card">
                  <strong>Audio:</strong> {predictions.audio}
                  {/* <div>Score: {predictions.test_score}</div> */}
                </div>
              )}
              {predictions.eeg && (
                <div className="result-card">
                  <strong>EEG:</strong> {predictions.eeg}
                  <div>Score: {predictions.eeg_score}</div>
                </div>
              )}
              {predictions.spatial_risk && (
                <div className="result-card">
                  <strong>Spatial Risk:</strong> {predictions.spatial_risk}
                  <div>Score: {predictions.spatial_score}</div>
                </div>
              )}
            </div>
          )}
          {error && <div className="error-box">{error}</div>}
        </div>


      </div>
    </div>

  );
}
