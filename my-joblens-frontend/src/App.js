import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Homepage from "./pages/Homepage";  // Ensure this import exists
import Dashboard from "./pages/Dashboard";

import Prediction from "./pages/Prediction";

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Homepage />} />   {/* Fixes the warning */}
        <Route path="/dashboard" element={<Dashboard />} />
        <Route path="/prediction" element={<Prediction />} />
      </Routes>
    </Router>
  );
}

export default App;
