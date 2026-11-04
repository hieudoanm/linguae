'use client';

import { type FC } from 'react';
import { NAV_LINKS } from './types';
import { useTheme } from './ThemeProvider';
import Image from 'next/image';
import Link from 'next/link';

interface NavbarProps {
  scrollTo: (id: string) => void;
}

const Navbar: FC<NavbarProps> = ({ scrollTo }) => {
  const { theme, toggleTheme } = useTheme();

  return (
    <nav className="fixed top-0 right-0 left-0 z-50 border-b border-gray-200 bg-white backdrop-blur dark:border-gray-700 dark:bg-gray-900/80">
      <div className="mx-auto w-full max-w-4xl px-4">
        <div className="flex h-16 items-center justify-between">
          <Link
            href="/"
            onClick={(e) => {
              e.preventDefault();
              scrollTo('#hero');
            }}
            className="flex cursor-pointer items-center gap-2 text-xl font-bold text-gray-900 dark:text-white"
            role="link"
            tabIndex={0}>
            <Image
              src="/svg/logo.svg"
              alt=""
              width={32}
              height={32}
              className="h-8 w-8"
            />
            <span>Linguae</span>
          </Link>
          <div className="hidden items-center gap-6 md:flex">
            {NAV_LINKS.map((link) => (
              <Link
                key={link.href}
                href={link.href}
                className="text-sm font-medium text-gray-600 transition-colors hover:text-blue-600 dark:text-gray-300 dark:hover:text-blue-400">
                {link.label}
              </Link>
            ))}
            <button
              onClick={toggleTheme}
              className="rounded-lg p-2 text-gray-600 transition-colors hover:bg-gray-100 dark:text-gray-300 dark:hover:bg-gray-800"
              aria-label={
                theme === 'light'
                  ? 'Switch to dark mode'
                  : 'Switch to light mode'
              }>
              {theme === 'light' ? (
                <svg
                  className="h-5 w-5"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                  aria-hidden="true">
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z"
                  />
                </svg>
              ) : (
                <svg
                  className="h-5 w-5"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                  aria-hidden="true">
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z"
                  />
                </svg>
              )}
            </button>
          </div>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;
