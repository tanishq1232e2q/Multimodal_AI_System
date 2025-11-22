import React from 'react'
import { Link } from 'react-router-dom';

export default function Navbar() {
  return (
    <nav className="navbar navbar-expand-lg" style={{ backgroundColor: "#0070f3",fontSize:"1.1rem", padding: "0.6rem 2rem" }}> 
      <div className="container-fluid">

        
        <a className="navbar-brand text-white fw-bold" href="#">Menya Ai</a>

     
        <button 
          className="navbar-toggler" 
          type="button" 
          data-bs-toggle="collapse" 
          data-bs-target="#navbarSupportedContent"
        >
          <span className="navbar-toggler-icon"></span>
        </button>

        {/* Right side menu */}
        <div className="collapse navbar-collapse justify-content-end justify-center" id="navbarSupportedContent">
          <ul className="navbar-nav mb-2 mb-lg-0">

            <li className="nav-item">
              <Link className="nav-link text-white" to="/">Home</Link>
            </li>

            <li className="nav-item">
              <Link className="nav-link text-white" to="/dashboard">Dashboard</Link>
            </li>

            <li className="nav-item">
              <Link className="nav-link text-white" to="/about">About Us</Link>
            </li>

            <li className="nav-item">
              <Link className="nav-link text-white" to="/metrics">Metrics</Link>
            </li>

          </ul>
        </div>

      </div>
    </nav>
  )
}
