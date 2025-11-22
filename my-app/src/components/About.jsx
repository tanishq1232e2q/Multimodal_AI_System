import React from "react";
import "../App.css"; // 👈 Import manual CSS

export default function AboutPage() {
  return (
    <main className="about-page">
      {/* Header Section */}
      <section className="about-header">
        <h1>About Us</h1>
        <p>Your safe space to reflect, heal, and grow — with the help of AI.</p>
      </section>

      {/* Why We Built This */}
      <section className="about-section">
        <div className="about-image">
          {/* Optional image */}
          {/* <img src="/images/mental-health-support.jpg" alt="Mental Health Support" /> */}
        </div>
        <div className="about-text">
          <h2>Why We Built This</h2>
          <p>
            Mental health matters — and we believe everyone deserves a tool that helps them
            process thoughts, track emotions, and feel supported, even when they’re alone.
          </p>
          <p>
            This AI-powered journal was designed to make mental health support more
            accessible, reflective, and personal using modern technology.
          </p>
        </div>
      </section>

      {/* What Makes Us Different */}
      <section className="about-section reverse">
        <div className="about-text">
          <h2>What Makes Us Different</h2>
          <ul>
            <li>Get detailed analysis of your problem and suggested solutions</li>
            <li>AI-generated reflections using Mistral LLM via OpenRouter</li>
            <li>Mood & depression tracking with charts</li>
            <li>Privacy-first design with user-specific entries</li>
          </ul>
        </div>
        <div className="about-image">
          {/* <img src="/images/ai-insight.jpg" alt="AI Insight" /> */}
        </div>
      </section>

      {/* Vision Section */}
      <section className="about-vision">
        <h2>Our Vision</h2>
        <p>
          We want to empower individuals to take control of their emotional well-being
          through reflective journaling.
        </p>
      </section>

      {/* Footer */}
      <footer className="about-footer">
        &copy; {new Date().getFullYear()} Multimodal Mental Disorder Prediction System
      </footer>
    </main>
  );
}
