import type { Metadata, Viewport } from 'next';
import { Geist, Geist_Mono } from 'next/font/google';
import './globals.css';
import Providers from './providers';

const geistSans = Geist({
  variable: '--font-geist-sans',
  subsets: ['latin'],
});

const geistMono = Geist_Mono({
  variable: '--font-geist-mono',
  subsets: ['latin'],
});

export const metadata: Metadata = {
  title: 'Linguae - Language Detection',
  description: 'Detect the language of text with a local-first, character-level CNN',
  openGraph: {
    title: 'Linguae - Language Detection',
    description: 'Detect the language of text with a local-first, character-level CNN',
    url: 'https://linguae.dev',
    siteName: 'Linguae',
    type: 'website',
    locale: 'en_US',
    images: [
      {
        url: '/svg/og-image.svg',
        width: 1200,
        height: 630,
        alt: 'Linguae - Language Detection',
      },
    ],
  },
  twitter: {
    card: 'summary_large_image',
    title: 'Linguae - Language Detection',
    description: 'Detect the language of text with a local-first, character-level CNN',
    images: ['/og-image.svg'],
  },
  icons: {
    icon: '/favicon.svg',
    shortcut: '/favicon.svg',
    apple: '/logo.svg',
  },
  manifest: '/manifest.json',
};

export const viewport: Viewport = {
  themeColor: [
    { media: '(prefers-color-scheme: light)', color: '#ffffff' },
    { media: '(prefers-color-scheme: dark)', color: '#0a0a0a' },
  ],
};

export default function RootLayout({ children }: LayoutProps<'/'>) {
  return (
    <html
      lang="en"
      className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}
      suppressHydrationWarning
    >
      <body className="min-h-full flex flex-col">
        <Providers>{children}</Providers>
      </body>
    </html>
  );
}
