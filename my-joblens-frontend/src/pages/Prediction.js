import React, { useState } from "react";
import axios from "axios";
import "./Prediction.css";

const Prediction = () => {
    const [file, setFile] = useState(null);
    const [categorizationModel, setCategorizationModel] = useState("rf");
    const [recommendationModel, setRecommendationModel] = useState("rf");
    const [jobDescription, setJobDescription] = useState("");
    const [industry, setIndustry] = useState("");
    const [result, setResult] = useState(null);
    const [atsResult, setAtsResult] = useState(null);
    const [error, setError] = useState(null);
    const [loading, setLoading] = useState(false);

    // Handle file selection
    const handleFileChange = (e) => {
        const selectedFile = e.target.files[0];

        if (selectedFile) {
            const allowedTypes = ["application/pdf", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"];
            if (!allowedTypes.includes(selectedFile.type)) {
                setError("❌ Invalid file type. Please upload a PDF or DOCX file.");
                setFile(null);
                return;
            }

            // Update file name to avoid spaces and special characters
            const safeFileName = selectedFile.name.replace(/[^a-zA-Z0-9.]/g, "_");
            
            // Create a new file object with the safe name
            const renamedFile = new File([selectedFile], safeFileName, {
                type: selectedFile.type,
                lastModified: selectedFile.lastModified,
            });
            
            setFile(renamedFile);
            setError(null);
        }
    };

    // Handle prediction request
    const handlePredict = async () => {
        if (!file) {
            setError("⚠️ Please upload a resume file.");
            return;
        }

        setLoading(true);
        setError(null);

        // Create formData for prediction request
        const formData = new FormData();
        formData.append("resume", file);
        formData.append("categorization_model", categorizationModel);
        formData.append("recommendation_model", recommendationModel);

        // Create formData for ATS check request
        const atsFormData = new FormData();
        atsFormData.append("resume", file);
        if (jobDescription.trim()) {
            atsFormData.append("job_description", jobDescription);
        }
        if (industry.trim()) {
            atsFormData.append("industry", industry);
        }

        try {
            // Make prediction request first
            const predictionResponse = await axios.post("http://127.0.0.1:5000/predict", formData, {
                headers: { "Content-Type": "multipart/form-data" },
                timeout: 30000 // 30 second timeout
            });
            
            console.log("Prediction Response:", predictionResponse.data);
            setResult(predictionResponse.data);
            
            // Then make ATS check request only if prediction succeeded
            const atsResponse = await axios.post("http://127.0.0.1:5000/ats_check", atsFormData, {
                headers: { "Content-Type": "multipart/form-data" },
                timeout: 30000 // 30 second timeout
            });
            
            console.log("ATS Response:", atsResponse.data);
            setAtsResult(atsResponse.data);
        } catch (err) {
            console.error("Error:", err.response?.data || err.message);
            setError(err.response?.data?.error || "Request failed. Please try again with a different file.");
            // Keep any successful results that we might have gotten before the error
        } finally {
            setLoading(false);
        }
    };

    // Reset the form
    const handleReset = () => {
        setFile(null);
        setJobDescription("");
        setIndustry("");
        setResult(null);
        setAtsResult(null);
        setError(null);
        
        // Reset file input by clearing its value
        const fileInput = document.getElementById("resume-upload");
        if (fileInput) {
            fileInput.value = "";
        }
    };

    return (
        <div className="container">
            <h2 className="app-title">JobLens AI</h2>

            {/* File Upload Section */}
            <div className="upload-section">
                <input type="file" id="resume-upload" onChange={handleFileChange} accept=".pdf,.docx" />
                <label htmlFor="resume-upload" className="upload-btn">📤 Upload Resume</label>
                {file && <p>✅ Selected File: {file.name}</p>}
            </div>

            {/* Job Description and Industry Input */}
            <div className="input-section">
                <label htmlFor="job-description">📝 Job Description (Optional):</label>
                <textarea 
                    id="job-description" 
                    value={jobDescription}
                    onChange={(e) => setJobDescription(e.target.value)}
                    placeholder="Paste the job description for better ATS analysis..."
                    rows="3"
                ></textarea>
                
                <label htmlFor="industry">🏢 Industry (Optional):</label>
                <input 
                    type="text" 
                    id="industry" 
                    value={industry}
                    onChange={(e) => setIndustry(e.target.value)}
                    placeholder="e.g., Technology, Healthcare, Finance..."
                />
            </div>

            {/* Model Selection Section */}
            <div className="select-models">
                <label>🔍 Choose Categorization Model:</label>
                <select value={categorizationModel} onChange={(e) => setCategorizationModel(e.target.value)}>
                    <option value="nb">Model 1 (best)</option>
                    <option value="logistic">Model 2</option>
                    <option value="knn">Model 3</option>
                </select>

                <label>💼 Choose Recommendation Model:</label>
                <select value={recommendationModel} onChange={(e) => setRecommendationModel(e.target.value)}>
                    <option value="nb">Model 1</option>
                    <option value="logistic">Model 2 (best)</option>
                    <option value="knn">Model 3</option>
                </select>
            </div>

            {/* Action Buttons */}
            <div className="action-buttons">
                <button className="predict-btn" onClick={handlePredict} disabled={loading}>
                    {loading ? "⏳ Processing..." : "🔍 Predict"}
                </button>
                <button className="reset-btn" onClick={handleReset} disabled={loading}>
                    🔄 Reset
                </button>
            </div>

            {/* Display Error Message */}
            {error && <div className="result-section error">❌ {error}</div>}

            {/* Display Prediction Results */}
            {result && (
                <div className="result-section">
                    <h3>📊 Prediction Results</h3>
                    <p><strong>📌 Job Category:</strong> {result.Predictions?.Categorization || "Not Found"}</p>
                    <p><strong>💼 Recommended Job:</strong> {result.Predictions?.["Job Recommendation"] || "Not Found"}</p>

                    {/* ATS Score */}
                    {atsResult && (
                        <div className="ats-score">
                            <h4>📈 ATS Score: {atsResult.score}%</h4>
                            <div className="progress-bar">
                                <div className="progress" style={{ width: `${atsResult.score}%`, 
                                    backgroundColor: atsResult.score > 80 ? '#4CAF50' : 
                                                    atsResult.score > 60 ? '#FFC107' : '#F44336' }}>
                                </div>
                            </div>
                            
                            {/* ATS Warnings */}
                            {atsResult.warnings && atsResult.warnings.length > 0 && (
                                <div className="ats-warnings">
                                    <h5>⚠️ Warnings:</h5>
                                    <ul>
                                        {atsResult.warnings.map((warning, index) => (
                                            <li key={index}>{warning}</li>
                                        ))}
                                    </ul>
                                </div>
                            )}
                            
                            {/* ATS Recommendations */}
                            {atsResult.recommendations && atsResult.recommendations.length > 0 && (
                                <div className="ats-recommendations">
                                    <h5>💡 Recommendations:</h5>
                                    <ul>
                                        {atsResult.recommendations.map((recommendation, index) => (
                                            <li key={index}>{recommendation}</li>
                                        ))}
                                    </ul>
                                </div>
                            )}
                            
                            {/* File Format Information */}
                            {atsResult.file_format && (
                                <div className="file-format">
                                    <h5>📄 File Information:</h5>
                                    <p><strong>Format:</strong> {atsResult.file_format.format}</p>
                                    <p><strong>Size:</strong> {atsResult.file_format.size_mb} MB</p>
                                    {atsResult.file_format.issues && atsResult.file_format.issues.length > 0 && (
                                        <div>
                                            <p><strong>Issues:</strong></p>
                                            <ul>
                                                {atsResult.file_format.issues.map((issue, index) => (
                                                    <li key={index}>{issue}</li>
                                                ))}
                                            </ul>
                                        </div>
                                    )}
                                </div>
                            )}
                        </div>
                    )}

                    <h4>📜 Extracted Resume Details:</h4>
                    <p><strong>👤 Name:</strong> {result["Resume Data"]?.Name || "Not Found"}</p>
                    <p><strong>📧 Email:</strong> {result["Resume Data"]?.Email || "Not Found"}</p>
                    <p><strong>📞 Phone:</strong> {result["Resume Data"]?.Phone || "Not Found"}</p>
                    <p><strong>🛠 Skills:</strong> {Array.isArray(result["Resume Data"]?.Skills) ? result["Resume Data"]?.Skills.join(", ") : (result["Resume Data"]?.Skills || "Not Available")}</p>
                    <p><strong>🎓 Education:</strong> {result["Resume Data"]?.Education || "Not Available"}</p>
                    
                    {/* Display Experience if available */}
                    {result["Resume Data"]?.Experience && (
                        <div className="experience-section">
                            <h5><strong>💼 Experience:</strong></h5>
                            {Array.isArray(result["Resume Data"].Experience) && result["Resume Data"].Experience.length > 0 && result["Resume Data"].Experience[0] !== "Not Found" ? (
                                result["Resume Data"].Experience.map((exp, index) => (
                                    <div
                                        key={index}
                                        style={{
                                            marginBottom: '20px',
                                            borderBottom: '1px solid #ccc',
                                            paddingBottom: '10px',
                                        }}
                                    >
                                        <h6>
                                            {exp["Job Title"] || "Unknown Title"} at {exp["Company"] || "Unknown Company"}
                                        </h6>
                                        <p><strong>Duration:</strong> {exp["Duration"] || "Not Available"}</p>
                                        <p><strong>Location:</strong> {exp["Location"] || "Not Available"}</p>
                                        <p><strong>Key Responsibilities:</strong></p>
                                        {Array.isArray(exp["Key Responsibilities"]) && exp["Key Responsibilities"].length > 0 ? (
                                            <ul>
                                                {exp["Key Responsibilities"].map((responsibility, idx) => (
                                                    <li key={idx}>{responsibility}</li>
                                                ))}
                                            </ul>
                                        ) : (
                                            <p>No responsibilities listed.</p>
                                        )}
                                    </div>
                                ))
                            ) : (
                                <p>Not Found</p>
                            )}
                        </div>
                    )}
                    
                    {/* Display category probabilities if available */}
                    {result.Predictions?.["Category Probabilities"] && (
                        <div className="probabilities">
                            <h5>📊 Category Match Percentages:</h5>
                            <ul>
                                {Object.entries(result.Predictions["Category Probabilities"])
                                    .sort((a, b) => b[1] - a[1])
                                    .map(([category, probability], index) => (
                                        <li key={index}>{category}: {probability}%</li>
                                    ))}
                            </ul>
                        </div>
                    )}
                    
                    {/* Display job recommendation probabilities if available */}
                    {result.Predictions?.["Job Recommendation Probabilities"] && (
                        <div className="probabilities">
                            <h5>📊 Job Match Percentages:</h5>
                            <ul>
                                {Object.entries(result.Predictions["Job Recommendation Probabilities"])
                                    .sort((a, b) => b[1] - a[1])
                                    .map(([job, probability], index) => (
                                        <li key={index}>{job}: {probability}%</li>
                                    ))}
                            </ul>
                        </div>
                    )}
                </div>
            )}
        </div>
    );
};

export default Prediction;