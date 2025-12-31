import React from 'react';

const Header = ({ onAboutClick }) => {
  return (
    <header className="header">
      <div className="logo-container">
        <img src="/grow logo.png" alt="Grow Logo" className="logo" />
      </div>
      <button className="about-btn" onClick={onAboutClick}>
        About
      </button>
    </header>
  );
};

export default Header;