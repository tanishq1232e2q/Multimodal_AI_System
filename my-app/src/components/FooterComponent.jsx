import React from "react";
import { Link } from "react-router-dom";
import "../App.css"; // 👈 Import manual CSS

export default function FooterComponent() {
  return (
    <footer className="footer">
      <div className="footer-container">
        <div className="footer-section">
          <h6 className="footer-title">Menya Mental Journal</h6>
          <p className="footer-text">
            Mental health matters — and we believe everyone deserves a tool
            that helps them process thoughts, track emotions, and feel
            supported, even when they’re alone.
          </p>
        </div>

        <div className="footer-section">
          <h6 className="footer-title">Products</h6>
          <p className="footer-link">Menya AI</p>
        </div>

        <div className="footer-section">
          <h6 className="footer-title">Useful Links</h6>
          <p><Link to="/" className="footer-link">Home</Link></p>
          <p><Link to="/dashboard" className="footer-link">Dashboard</Link></p>
          <p><Link to="/about" className="footer-link">About Us</Link></p>
        </div>

        <div className="footer-section">
          <h6 className="footer-title">Contact</h6>
          <p className="footer-text">New York, NY 10012, US</p>
          <p className="footer-text">tpalkhe648@gmail.com</p>
          <p className="footer-text">+ 01 234 567 88</p>
        </div>
      </div>

      <div className="footer-bottom">
        <span>© 2023 Copyright: </span>
        <a href="https://tw-elements.com/" className="footer-brand">Menya</a>
      </div>
    </footer>
  );
}
