import { Routes, Route } from "react-router-dom";
import Home from "./components/Home";
import Dashboard from "./components/Dashboard";
import About from "./components/About";
import Navbar from "./components/Navbar";
import FooterComponent from "./components/FooterComponent";
import Maingraph from "./components/Maingraph";
export default function App() {
  return (
    <>
    
    <Navbar/>
    <Routes>
      <Route path="/" element={<Home />} />
      <Route path="/about" element={<About />} />
      <Route path="/dashboard" element={<Dashboard />} />
      <Route path="/metrics" element={<Maingraph/>} />
    </Routes>
    <FooterComponent/>
    </>
  );
}
