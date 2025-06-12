import { useState } from "react";
import { motion } from "framer-motion";
import { useNavigate } from "react-router-dom";
import "./Signup.css";

const Signup = () => {
  const navigate = useNavigate();
  const [formData, setFormData] = useState({
    name: "",
    email: "",
    password: "",
    confirmPassword: ""
  });
  const [error, setError] = useState("");

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
    setError(""); // Clear error when user types
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    // Basic validation
    if (!formData.name || !formData.email || !formData.password || !formData.confirmPassword) {
      setError("All fields are required");
      return;
    }

    if (formData.password !== formData.confirmPassword) {
      setError("Passwords do not match");
      return;
    }

    if (formData.password.length < 6) {
      setError("Password must be at least 6 characters long");
      return;
    }

    try {
      // Here you would typically make an API call to your backend
      // For now, we'll simulate a successful signup
      
      // Clear form data
      setFormData({
        name: "",
        email: "",
        password: "",
        confirmPassword: ""
      });

      // Show success message
      alert("Account created successfully!");
      
      // Redirect to login
      navigate("/login");
      
    } catch (err) {
      setError(err.message || "Something went wrong");
    }
  };

  return (
    <div className="signup-container">
      <div className="signup-background">
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
        className="signup-card"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8 }}
      >
        <motion.h1
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.3 }}
        >
          Create Account
        </motion.h1>
        <motion.p
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.4 }}
        >
          Join JobLens AI Today
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
              type="text"
              name="name"
              required
              value={formData.name}
              onChange={handleChange}
            />
            <label>Full Name</label>
          </div>

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

          <div className="input-group">
            <input
              type="password"
              name="confirmPassword"
              required
              value={formData.confirmPassword}
              onChange={handleChange}
            />
            <label>Confirm Password</label>
          </div>

          <motion.button
            type="submit"
            className="signup-btn"
            whileHover={{ 
              scale: 1.02,
              boxShadow: "0 8px 25px rgba(96, 165, 250, 0.5)"
            }}
            whileTap={{ scale: 0.98 }}
          >
            Create Account
          </motion.button>
        </motion.form>

        <div className="login-prompt">
          <p>Already have an account?</p>
          <motion.button
            onClick={() => navigate("/login")}
            className="login-link"
            whileHover={{ 
              scale: 1.02,
              backgroundColor: "rgba(96, 165, 250, 0.15)"
            }}
            whileTap={{ scale: 0.98 }}
          >
            Log In
          </motion.button>
        </div>
      </motion.div>
    </div>
  );
};

export default Signup;