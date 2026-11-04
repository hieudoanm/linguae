import Link from 'next/link';

const NotFound = () => {
  return (
    <div className="flex min-h-screen flex-col items-center justify-center p-4 text-center">
      <h1 className="mb-2 text-6xl font-bold text-gray-900 dark:text-white">
        404
      </h1>
      <h2 className="mb-6 text-xl font-semibold text-gray-600 dark:text-gray-300">
        Page not found
      </h2>
      <Link
        href="/"
        className="rounded bg-blue-600 px-4 py-2 text-white hover:bg-blue-700">
        Go home
      </Link>
    </div>
  );
};

export default NotFound;
