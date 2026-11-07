import Link from 'next/link';

const NotFound = () => {
  return (
    <div className="flex min-h-screen flex-col items-center justify-center gap-4 p-4 text-center">
      <h1 className="text-6xl font-bold">404</h1>
      <h2 className="text-base-content/70 text-xl font-semibold">
        Page not found
      </h2>
      <Link href="/" className="btn btn-primary">
        Go home
      </Link>
    </div>
  );
};

export default NotFound;
