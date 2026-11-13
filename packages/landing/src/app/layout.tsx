import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import type { FC, ReactNode } from 'react';
import Footer from '@/components/molecules/Footer';
import Navbar from '@/components/molecules/Navbar';
import './globals.css';

const inter = Inter({ subsets: ['latin'] });

const GITHUB = 'https://github.com/hieudoanm/linguae';
const LOGO =
  process.env.NODE_ENV === 'production' ? '/linguae/logo.svg' : '/logo.svg';

const NAV_LINKS = [
  { label: 'Features', href: '#features' },
  { label: 'Demo', href: '#demo' },
  { label: 'How it works', href: '#how-it-works' },
  { label: 'Install', href: '#install' },
  { label: 'Docs', href: '/docs/' },
];

const DESCRIPTION =
  'Linguae detects the language of any text with a character-level CNN trained from scratch in Rust. No cloud calls, no pretrained weights — your text never leaves your machine.';

export const metadata: Metadata = {
  metadataBase: new URL('https://hieudoanm.github.io/linguae/'),
  title: {
    default: 'Linguae — Local-first language identification',
    template: '%s — Linguae',
  },
  description: DESCRIPTION,
  openGraph: {
    siteName: 'Linguae',
    type: 'website',
    title: 'Linguae — Local-first language identification',
    description: DESCRIPTION,
    url: 'https://hieudoanm.github.io/linguae/',
    images: ['/og.png'],
  },
  twitter: { card: 'summary_large_image', images: ['/og.png'] },
  icons: {
    icon: [
      { url: '/favicon.ico', sizes: 'any' },
      { url: '/favicon.svg', type: 'image/svg+xml' },
    ],
  },
};

interface RootLayoutProps {
  children: ReactNode;
}

const RootLayout: FC<RootLayoutProps> = ({ children }) => (
  <html
    lang="en"
    data-theme="linguae"
    className={`scroll-smooth antialiased motion-reduce:scroll-auto ${inter.className}`}>
    <body className="bg-base-100 text-base-content m-0 leading-relaxed">
      <Navbar
        name="Linguae"
        logoSrc={LOGO}
        links={NAV_LINKS}
        githubUrl={GITHUB}
      />
      {children}
      <Footer
        name="linguae"
        tagline="Local-first language identification · MIT licensed"
        githubUrl={GITHUB}
      />
    </body>
  </html>
);

export default RootLayout;
