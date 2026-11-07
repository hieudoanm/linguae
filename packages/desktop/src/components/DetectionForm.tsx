import { useState, type FC } from 'react';

import type { Prediction } from './types';

const DetectionForm: FC = () => {
  const [text, setText] = useState('');
  const [result, setResult] = useState<Prediction | null>(null);
  const [isPending, setIsPending] = useState(false);
  const [isError, setIsError] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const input = text.trim();
    if (!input || isPending) {
      return;
    }
    setIsPending(true);
    setIsError(false);
    try {
      const { invoke } = await import('@tauri-apps/api/core');
      const data = await invoke<Prediction>('detect', {
        text: input,
        topK: 3,
      });
      setResult(data);
    } catch {
      setIsError(true);
    } finally {
      setIsPending(false);
    }
  };

  return (
    <section
      id="detect"
      className="flex h-full items-center justify-center bg-white py-12 dark:bg-gray-900">
      <div className="mx-auto w-full max-w-4xl px-4">
        <div className="mb-8 text-center">
          <h2 className="mb-2 text-2xl font-bold text-gray-900 dark:text-white">
            Try It
          </h2>
          <p className="text-gray-600 dark:text-gray-300">
            Enter text to detect its language.
          </p>
        </div>

        <form onSubmit={handleSubmit} className="mb-6">
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Enter text to detect language..."
            className="h-32 w-full rounded-lg border border-gray-300 p-3 text-gray-900 focus:ring-2 focus:ring-blue-500 focus:outline-none dark:border-gray-600 dark:text-white"
            required
          />
          <button
            type="submit"
            disabled={isPending || !text.trim()}
            className="mt-3 w-full rounded-lg bg-blue-600 px-4 py-2 text-white hover:bg-blue-700 disabled:cursor-not-allowed disabled:bg-gray-400">
            {isPending ? 'Detecting...' : 'Submit'}
          </button>
        </form>

        {isError && (
          <div className="mb-4 rounded-lg border border-red-200 bg-red-50 p-4 text-red-700">
            Error detecting language. Make sure the Tauri runtime is available.
          </div>
        )}

        {result && (
          <div className="rounded-lg border border-gray-200 p-4 shadow-sm dark:border-gray-700">
            <h2 className="mb-3 text-xl font-semibold text-gray-900 dark:text-white">
              Result
            </h2>
            <p className="mb-2 text-gray-700">
              <span className="font-medium">Language:</span> {result.language}
            </p>
            <p className="mb-3 text-gray-700">
              <span className="font-medium">Confidence:</span>{' '}
              {(result.confidence * 100).toFixed(2)}%
            </p>
            {result.top_k && result.top_k.length > 0 && (
              <div>
                <p className="mb-2 font-medium text-gray-700">
                  Top alternatives:
                </p>
                <ul className="list-inside list-disc space-y-1">
                  {result.top_k.map((item, idx) => (
                    <li key={idx} className="text-gray-600 dark:text-gray-300">
                      {item.language}: {(item.score * 100).toFixed(2)}%
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        )}
      </div>
    </section>
  );
};

export default DetectionForm;
