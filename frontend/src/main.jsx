import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import App from '../App.jsx';

// This finds the <div id="root"> in your index.html and renders your App inside it
createRoot(document.getElementById('root')).render(
  <StrictMode>
    <App />
  </StrictMode>
);