'use client';

import { useEffect } from 'react';

interface ErrorProps {
  error: Error;
  reset: () => void;
}

const Error = ({ error, reset }: ErrorProps) => {
  useEffect(() => {
    console.error('Page error:', error);
  }, [error]);

  return (
    <div className="flex min-h-screen flex-col items-center justify-center text-center">
      <h2 className="mb-2 text-2xl font-bold text-gray-900 dark:text-white">
        Something went wrong
      </h2>
      <p className="mb-6 text-gray-600 dark:text-gray-300">{error.message}</p>
      <button
        onClick={() => reset()}
        className="rounded bg-blue-600 px-4 py-2 text-white hover:bg-blue-700">
        Try again
      </button>
    </div>
  );
};

export default Error;
