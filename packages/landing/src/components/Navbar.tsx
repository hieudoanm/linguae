import type { FC } from 'react';
import Image from 'next/image';
import Link from 'next/link';
import { assetPath } from '@/lib/site';
import ThemeToggle from './ThemeToggle';

const NAV_LINKS = [
  { href: '#features', label: 'Features' },
  { href: '#demo', label: 'Demo' },
  { href: '#how-it-works', label: 'How it works' },
  { href: '#install', label: 'Install' },
];

const GITHUB_URL = 'https://github.com/hieudoanm/linguae';

const Navbar: FC = () => {
  return (
    <div className="navbar border-base-300 bg-base-100/90 sticky top-0 z-50 border-b backdrop-blur">
      <div className="navbar-start">
        <Link href="/" className="btn btn-ghost gap-2 text-lg font-bold">
          <Image
            src={assetPath('/svg/logo-light.svg')}
            alt=""
            width={32}
            height={32}
            className="size-8 rounded-lg dark:hidden"
          />
          <Image
            src={assetPath('/svg/logo.svg')}
            alt=""
            width={32}
            height={32}
            className="hidden size-8 rounded-lg dark:block"
          />
          Linguae
        </Link>
      </div>

      <div className="navbar-center hidden lg:flex">
        <ul className="menu menu-horizontal gap-1 px-1">
          {NAV_LINKS.map((link) => (
            <li key={link.href}>
              <a href={link.href}>{link.label}</a>
            </li>
          ))}
        </ul>
      </div>

      <div className="navbar-end gap-1">
        <ThemeToggle />
        <a
          href={GITHUB_URL}
          target="_blank"
          rel="noreferrer"
          className="btn btn-primary btn-sm hidden sm:inline-flex">
          GitHub
        </a>
        <div className="dropdown dropdown-end lg:hidden">
          <div
            tabIndex={0}
            role="button"
            className="btn btn-ghost btn-circle"
            aria-label="Open menu">
            <svg
              xmlns="http://www.w3.org/2000/svg"
              className="size-5"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
              strokeWidth={2}>
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                d="M4 6h16M4 12h16M4 18h16"
              />
            </svg>
          </div>
          <ul
            tabIndex={0}
            className="menu dropdown-content rounded-box border-base-300 bg-base-100 mt-3 w-52 border p-2 shadow">
            {NAV_LINKS.map((link) => (
              <li key={link.href}>
                <a href={link.href}>{link.label}</a>
              </li>
            ))}
            <li>
              <a href={GITHUB_URL} target="_blank" rel="noreferrer">
                GitHub
              </a>
            </li>
          </ul>
        </div>
      </div>
    </div>
  );
};

export default Navbar;
