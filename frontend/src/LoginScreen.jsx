import React, { useState } from "react";

const LoginScreen = ({ onLogin }) => {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");

  const handleSubmit = (e) => {
    e.preventDefault();

    // simple UI login – consider this as "success"
    if (!username || !password) {
      alert("Please enter username and password");
      return;
    }

    onLogin(username, password);
  };

  return (
    <div className="login-wrapper">
      <div className="login-card">
        {/* Logo */}
        <img src="/grow logo.png" alt="Grow" className="login-logo" />

        {/* Tagline */}
        <p className="login-tagline">Your Cameras See, Our AI Understands</p>

        {/* Login form */}
        <form className="login-form" onSubmit={handleSubmit}>
          <div className="form-group">
            <label>Username</label>
            <input
              type="text"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              placeholder="Enter username"
            />
          </div>

          <div className="form-group">
            <label>Password</label>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="Enter password"
            />
          </div>

          <button type="submit" className="login-button">
            Sign In
          </button>
        </form>
      </div>
    </div>
  );
};

export default LoginScreen;