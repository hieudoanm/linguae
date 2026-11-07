'use client';

import { useEffect } from 'react';

interface GlobalErrorProps {
  error: Error;
  reset: () => void;
}

// global-error replaces the root layout, so the Tailwind/daisyUI stylesheet
// is not loaded here — inline styles are required.
const GlobalError = ({ error, reset }: GlobalErrorProps) => {
  useEffect(() => {
    console.error('Global error:', error);
  }, [error]);

  return (
    <html lang="en">
      <body
        style={{
          margin: 0,
          minHeight: '100vh',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          gap: '1rem',
          fontFamily: 'system-ui, sans-serif',
          backgroundColor: '#f2f2f0',
          color: '#2f2e2b',
          textAlign: 'center',
          padding: '1rem',
        }}>
        <h2 style={{ margin: 0, fontSize: '1.5rem' }}>Application error</h2>
        <p style={{ margin: 0, opacity: 0.7 }}>{error.message}</p>
        <button
          type="button"
          onClick={reset}
          style={{
            border: 'none',
            borderRadius: '0.375rem',
            backgroundColor: '#3f3e3a',
            color: '#f2f2f0',
            padding: '0.5rem 1rem',
            cursor: 'pointer',
          }}>
          Try again
        </button>
      </body>
    </html>
  );
};

export default GlobalError;
