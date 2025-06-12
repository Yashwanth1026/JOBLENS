import { useState } from "react";
import { motion } from "framer-motion";
import { useNavigate } from "react-router-dom";
import "./Login.css";

const Login = () => {
  const navigate = useNavigate();
  const [formData, setFormData] = useState({
    email: "",
    password: ""
  });
  const [error, setError] = useState("");

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
    setError("");
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!formData.email || !formData.password) {
      setError("All fields are required");
      return;
    }

    try {
      // Here you would typically make an API call to verify credentials
      
      // Clear form data
      setFormData({
        email: "",
        password: ""
      });

      // Show success message
      alert("Login successful!");
      
      // Redirect to dashboard
      navigate("/dashboard");
      
    } catch (err) {
      setError(err.message || "Login failed");
    }
  };

  return (
    <div className="login-container">
      <div className="login-background">
        <motion.div 
          className="circle circle-1"
          animate={{
            scale: [1, 1.2, 1],
            opacity: [0.5, 0.8, 0.5]
          }}
          transition={{
            duration: 8,
            repeat: Infinity,
            ease: "easeInOut"
          }}
        />
        <motion.div 
          className="circle circle-2"
          animate={{
            scale: [1, 1.3, 1],
            opacity: [0.3, 0.6, 0.3]
          }}
          transition={{
            duration: 10,
            repeat: Infinity,
            ease: "easeInOut"
          }}
        />
      </div>
      
      <motion.div 
        className="login-card"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8 }}
      >
        <motion.h1
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.3 }}
        >
          Welcome Back
        </motion.h1>
        <motion.p
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.4 }}
        >
          Login to JobLens AI
        </motion.p>

        {error && <div className="error-message">{error}</div>}

        <motion.form 
          onSubmit={handleSubmit}
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.5 }}
        >
          <div className="input-group">
            <input
              type="email"
              name="email"
              required
              value={formData.email}
              onChange={handleChange}
            />
            <label>Email</label>
          </div>

          <div className="input-group">
            <input
              type="password"
              name="password"
              required
              value={formData.password}
              onChange={handleChange}
            />
            <label>Password</label>
          </div>

          <div className="options">
            <label className="remember">
              <input type="checkbox" />
              <span>Remember me</span>
            </label>
            <a href="/forgot-password">Forgot Password?</a>
          </div>

          <motion.button
            type="submit"
            className="login-btn"
            whileHover={{ 
              scale: 1.02,
              boxShadow: "0 8px 25px rgba(96, 165, 250, 0.5)"
            }}
            whileTap={{ scale: 0.98 }}
          >
            Login
          </motion.button>
        </motion.form>

        <div className="signup-prompt">
          <p>Don't have an account?</p>
          <motion.button
            onClick={() => navigate("/signup")}
            className="signup-link"
            whileHover={{ 
              scale: 1.02,
              backgroundColor: "rgba(96, 165, 250, 0.15)"
            }}
            whileTap={{ scale: 0.98 }}
          >
            Sign Up
          </motion.button>
        </div>
      </motion.div>
    </div>
  );
};

export default Login;