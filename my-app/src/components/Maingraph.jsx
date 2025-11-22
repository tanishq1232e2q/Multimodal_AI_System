import React, { useEffect, useState } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";

export default function Maingraph() {
  const [graphData, setGraphData] = useState({
    text: 0,
    
    eeg: 0,
    spatial: 0,
  });

  // Load persistent graph data
  useEffect(() => {
    const saved = localStorage.getItem("mental_graph");
    if (saved) setGraphData(JSON.parse(saved));
  }, []);

  const chartData = [
    { name: "Text", value: graphData.text },
    
    { name: "EEG", value: graphData.eeg },
    { name: "Spatial", value: graphData.spatial },
  ];

  

  return (
    <>
    
      <h2 style={{margin:"2rem 0rem",fontSize:"2rem", textAlign: "center" }}>User Input Activity Graph</h2>
    <div style={{ padding: 20,display:"flex",justifyContent:"center" }}>

      <div style={{ width: "40%", height: 350 }}>
        <ResponsiveContainer>
          <BarChart data={chartData} barCategoryGap="30%">
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" />
            <YAxis />
            
            <Tooltip />
            <Legend />
            <Bar dataKey="value" fill="#6366f1" />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
    </>
  );
}
