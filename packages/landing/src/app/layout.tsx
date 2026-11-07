import type { Metadata, Viewport } from 'next';
import { Geist, Geist_Mono } from 'next/font/google';
import { assetPath } from '@/lib/site';
import './globals.css';

const geistSans = Geist({
  variable: '--font-geist-sans',
  subsets: ['latin'],
});

const geistMono = Geist_Mono({
  variable: '--font-geist-mono',
  subsets: ['latin'],
});

export const metadata: Metadata = {
  metadataBase: new URL('https://linguae.dev'),
  title: 'Linguae — Local-first language detection',
  description:
    'Detect the language of text with a local character-level CNN. No cloud, no pretrained weights, no network calls.',
  openGraph: {
    title: 'Linguae — Local-first language detection',
    description:
      'Detect the language of text with a local character-level CNN. No cloud, no pretrained weights, no network calls.',
    url: 'https://linguae.dev',
    siteName: 'Linguae',
    type: 'website',
    locale: 'en_US',
    images: [
      {
        url: assetPath('/svg/og-image.svg'),
        width: 1200,
        height: 630,
        alt: 'Linguae — Local-first language detection',
      },
    ],
  },
  twitter: {
    card: 'summary_large_image',
    title: 'Linguae — Local-first language detection',
    description:
      'Detect the language of text with a local character-level CNN. No cloud, no pretrained weights, no network calls.',
    images: [assetPath('/svg/og-image.svg')],
  },
  icons: {
    icon: assetPath('/svg/favicon.svg'),
    shortcut: assetPath('/svg/favicon.svg'),
    apple: assetPath('/svg/logo.svg'),
  },
};

export const viewport: Viewport = {
  themeColor: [
    { media: '(prefers-color-scheme: light)', color: '#f2f2f0' },
    { media: '(prefers-color-scheme: dark)', color: '#17171b' },
  ],
};

const themeInitScript = `(function(){try{var t=localStorage.getItem('linguae-theme');if(t!=='linguae'&&t!=='linguae-dark'){t=window.matchMedia('(prefers-color-scheme: dark)').matches?'linguae-dark':'linguae'}document.documentElement.setAttribute('data-theme',t)}catch(e){document.documentElement.setAttribute('data-theme','linguae')}})();`;

const RootLayout = ({ children }: LayoutProps<'/'>) => {
  return (
    <html
      lang="en"
      className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}
      suppressHydrationWarning>
      <body className="bg-base-100 text-base-content flex min-h-full flex-col">
        <script dangerouslySetInnerHTML={{ __html: themeInitScript }} />
        {children}
      </body>
    </html>
  );
};

export default RootLayout;
