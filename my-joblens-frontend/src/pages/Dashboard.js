import { useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import "./Dashboard.css";

const Dashboard = () => {
  const navigate = useNavigate();

  const dashboardItems = [
    
   
    {
      title: "Smart Resume Analysis",
      description: "Get instant AI-powered resume analysis and recommendations",
      icon: "🎯",
      path: "/prediction",
      color: "#EC4899"
    }
  ];

  return (
    <div className="dashboard-container">
      <motion.nav 
        className="dashboard-nav"
        initial={{ y: -100 }}
        animate={{ y: 0 }}
        transition={{ type: "spring", stiffness: 100 }}
      >
        <div className="nav-content">
          <motion.h1 
            className="nav-title"
            whileHover={{ scale: 1.05 }}
          >
            JobLens <span className="accent">AI</span>
          </motion.h1>
          
        </div>
      </motion.nav>

      <main className="dashboard-main">
        <motion.h2 
          className="welcome-text"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
        >
          Welcome to Your AI Dashboard
        </motion.h2>

        <div className="dashboard-grid">
          {dashboardItems.map((item, index) => (
            <motion.div
              key={index}
              className="dashboard-card"
              style={{ '--hover-color': item.color }}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.1 + 0.5 }}
              whileHover={{ 
                scale: 1.02,
                boxShadow: `0 8px 32px ${item.color}33`
              }}
              whileTap={{ scale: 0.98 }}
              onClick={() => navigate(item.path)}
            >
              <div className="card-icon">{item.icon}</div>
              <h3>{item.title}</h3>
              <p>{item.description}</p>
              <motion.div 
                className="card-overlay"
                whileHover={{ opacity: 0.1 }}
                style={{ backgroundColor: item.color }}
              />
            </motion.div>
          ))}
        </div>
      </main>

      <motion.footer 
        className="dashboard-footer"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 1 }}
      >
        <p>© 2025 JobLens AI | Transforming Recruitment with Artificial Intelligence</p>
        <div className="footer-links">
          <a href="/privacy">Privacy Policy</a>
          <a href="/terms">Terms of Service</a>
          <a href="/contact">Contact Us</a>
        </div>
      </motion.footer>
    </div>
  );
};

export default Dashboard;