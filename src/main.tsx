import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import App from './App';
import './styles.css';

const container = document.getElementById('root');
if (container === null) {
  // index.html ships the mount node; its absence is a build error, not a runtime condition.
  throw new Error('Missing #root element in index.html');
}

createRoot(container).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
