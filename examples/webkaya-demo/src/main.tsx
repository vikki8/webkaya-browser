import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { installEgressMonitor } from './privacy';
import { App } from './App';
import './styles.css';

// Install before anything else can make a request.
installEgressMonitor();

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
