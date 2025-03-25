import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { FaPaperPlane, FaFileUpload } from 'react-icons/fa'; // Import icons
import './Ui.css';

function Ui() {
  const [file, setFile] = useState(null);
  const [fileName, setFileName] = useState('');
  const [query, setQuery] = useState('');
  const [messages, setMessages] = useState([]);
  const [uploadStatus, setUploadStatus] = useState('');
  const chatEndRef = useRef(null);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Handle file selection
  const handleFileChange = (event) => {
    const selectedFile = event.target.files[0];
    if (selectedFile) {
      setFile(selectedFile);
      setFileName(selectedFile.name);
      setUploadStatus('');
    }
  };

  // Handle query input change
  const handleQueryChange = (event) => {
    setQuery(event.target.value);
  };

  // Upload file to backend
  const handleFileUpload = async () => {
    if (!file) {
      alert("Please select a file before uploading.");
      return;
    }

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post('http://127.0.0.1:5000/upload', formData);
      setUploadStatus(response.data.message);
    } catch (error) {
      console.error("File upload error:", error);
      setUploadStatus("❌ File upload failed. Try again.");
    }
  };

  // Handle query submission
  const handleSubmit = async (event) => {
    event.preventDefault();
    if (!query.trim()) return;

    const newUserMessage = { text: query, type: 'user' };
    setMessages((prevMessages) => [...prevMessages, newUserMessage]);

    try {
      const response = await axios.post('http://127.0.0.1:5000/answer', { query });
      const botMessage = { text: response.data.answer, type: 'bot' };
      setMessages((prevMessages) => [...prevMessages, botMessage]);
    } catch (error) {
      console.error("Query error:", error);
      setMessages((prevMessages) => [...prevMessages, { text: "Error processing your query.", type: 'bot' }]);
    }

    setQuery('');
  };

  return (
    <div className="chat-container">
      <h1 className="chat-title">🤖 AI Chatbot</h1>

      {/* Chat Box */}
      <div className="chat-box">
        {messages.map((msg, index) => (
          <div key={index} className={`message ${msg.type}`}>
            {msg.type === 'user' ? '🧑‍💻' : '🤖'} {msg.text}
          </div>
        ))}
        <div ref={chatEndRef} />
      </div>

      {/* File Upload Section */}
      <div className="file-upload-container">
        <label className="file-upload">
          <FaFileUpload size={20} /> Choose File
          <input type="file" onChange={handleFileChange} />
        </label>
        <button className="upload-btn" onClick={handleFileUpload}>Upload</button>
      </div>

      {/* Upload Status */}
      {uploadStatus && <p className="upload-status">{uploadStatus}</p>}
      {fileName && <p className="file-name">📁 File: {fileName}</p>}

      {/* Query Input & Send Button */}
      <form onSubmit={handleSubmit} className="chat-form">
        <input 
          type="text" 
          value={query} 
          onChange={handleQueryChange} 
          placeholder="Ask something..." 
          className="chat-input"
        />
        <button type="submit" className="send-btn">
          <FaPaperPlane size={16} />
        </button>
      </form>
    </div>
  );
}

export default Ui;
