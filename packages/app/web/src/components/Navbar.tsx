'use client';

import { NAV_LINKS } from './types';

interface NavbarProps {
  scrollTo: (id: string) => void;
}

export default function Navbar({ scrollTo }: NavbarProps) {
  return (
    <nav className="fixed top-0 left-0 right-0 z-50 bg-white/80 backdrop-blur border-b border-gray-200">
      <div className="max-w-2xl mx-auto px-4">
        <div className="flex items-center justify-between h-16">
          <div
            onClick={() => scrollTo('#hero')}
            className="flex items-center gap-2 cursor-pointer font-bold text-xl text-gray-900"
            role="link"
            tabIndex={0}
          >
            <span className="bg-blue-600 text-white rounded px-2 py-1 text-sm">L</span>
            <span>Linguae</span>
          </div>
          <div className="hidden md:flex items-center gap-6">
            {NAV_LINKS.map((link) => (
              <a
                key={link.href}
                href={link.href}
                onClick={(e) => {
                  e.preventDefault();
                  scrollTo(link.href.replace('#', ''));
                }}
                className="text-sm font-medium text-gray-600 hover:text-blue-600 transition-colors"
              >
                {link.label}
              </a>
            ))}
          </div>
        </div>
      </div>
    </nav>
  );
}
