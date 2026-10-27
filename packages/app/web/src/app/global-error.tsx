'use client';

import { useEffect } from 'react';

interface GlobalErrorProps {
  error: Error;
  reset: () => void;
}

const GlobalError = ({ error, reset }: GlobalErrorProps) => {
  useEffect(() => {
    console.error('Global error:', error);
  }, [error]);

  return (
    <html>
      <body className="flex min-h-screen flex-col items-center justify-center p-4 text-center">
        <h2 className="mb-2 text-2xl font-bold text-gray-900 dark:text-white">
          Application error
        </h2>
        <p className="mb-6 text-gray-600 dark:text-gray-300">{error.message}</p>
        <button
          onClick={() => reset()}
          className="rounded bg-blue-600 px-4 py-2 text-white hover:bg-blue-700">
          Try again
        </button>
      </body>
    </html>
  );
};

export default GlobalError;
