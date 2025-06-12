import { useNavigate } from "react-router-dom";
import "./Homepage.css";
import { motion } from "framer-motion";

const Home = () => {
  const navigate = useNavigate();

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: { opacity: 1, transition: { duration: 1, staggerChildren: 0.3 } }
  };

  const itemVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: { opacity: 1, y: 0 }
  };

  const features = [
    {
      title: "Smart Job Predictions",
      description: "Get accurate predictions for job roles and categories based on your experience",
      color: "#60a5fa"
    },
    {
      title: "Information Extraction",
      description: "Automatically extract key details like contact information, education, and skills",
      color: "#a78bfa"
    },
    {
      title: "ATS Score Analysis",
      description: "Receive ATS compatibility score with detailed suggestions for improvement",
      color: "#f472b6"
    },
    {
      title: "Multiple AI Models",
      description: "Choose from various AI models for the most accurate predictions",
      color: "#34d399"
    }
  ];

  return (
    <motion.div 
      className="home-container"
      initial="hidden"
      animate="visible"
      variants={containerVariants}
    >
      <motion.header className="hero-section">
        <motion.h1
          variants={itemVariants}
          className="gradient-text"
        >
          JobLens AI
        </motion.h1>
        <motion.p
          variants={itemVariants}
        >
          Your Resume Through The Lens of AI Intelligence
        </motion.p>
        
        <motion.div 
          className="features-grid"
          variants={itemVariants}
        >
          {features.map((feature, index) => (
            <motion.div
              key={index}
              className="feature-card"
              whileHover={{ 
                scale: 1.05, 
                boxShadow: `0 8px 25px ${feature.color}33`
              }}
              whileTap={{ scale: 0.95 }}
              variants={itemVariants}
              style={{ borderTop: `4px solid ${feature.color}` }}
            >
              <h3>{feature.title}</h3>
              <p>{feature.description}</p>
            </motion.div>
          ))}
        </motion.div>

        <motion.div 
          className="cta-section"
          variants={itemVariants}
        >
          <h2>Start Your Career Analysis Today</h2>
          <div className="buttons">
            <motion.button
              whileHover={{ 
                scale: 1.05,
                boxShadow: '0 6px 20px rgba(96, 165, 250, 0.4)'
              }}
              whileTap={{ scale: 0.95 }}
              onClick={() => navigate("/Dashboard")}
              className="get-started-btn"
            >
              Get Started
            </motion.button>
          </div>
        </motion.div>
      </motion.header>
    </motion.div>
  );
};

export default Home;