import type { Metadata } from 'next';
import type { FC } from 'react';
import type { IconType } from 'react-icons';
import {
  FaApple,
  FaBoxOpen,
  FaDownload,
  FaLinux,
  FaTerminal,
  FaWindows,
} from 'react-icons/fa';

export const metadata: Metadata = {
  title: 'Downloads',
  description:
    'Download Linguae — local-first language identification — as a desktop app or CLI for macOS, Windows, and Linux.',
  alternates: { canonical: '/downloads/' },
};

const GITHUB = 'https://github.com/hieudoanm/linguae';
const RELEASES = `${GITHUB}/releases/latest/download`;
const RELEASE_PAGE = `${GITHUB}/releases/latest`;
const INSTALL_SH = `curl -fsSL ${GITHUB}/raw/master/scripts/install.sh | bash`;

interface Asset {
  label: string;
  href: string;
}

interface Platform {
  name: string;
  icon: IconType;
  assets: Asset[];
}

const CLI: Platform[] = [
  {
    name: 'macOS',
    icon: FaApple,
    assets: [
      {
        label: 'Apple Silicon',
        href: `${RELEASES}/cli-linguae-aarch64-apple-darwin`,
      },
    ],
  },
  {
    name: 'Windows',
    icon: FaWindows,
    assets: [
      {
        label: '.exe',
        href: `${RELEASES}/cli-linguae-x86_64-pc-windows-msvc.exe`,
      },
    ],
  },
  {
    name: 'Linux',
    icon: FaLinux,
    assets: [
      {
        label: 'x86_64',
        href: `${RELEASES}/cli-linguae-x86_64-unknown-linux-gnu`,
      },
    ],
  },
];

const DESKTOP: Platform[] = [
  {
    name: 'macOS',
    icon: FaApple,
    assets: [
      {
        label: '.dmg (Apple Silicon)',
        href: `${RELEASES}/Linguae_0.0.1_aarch64.dmg`,
      },
      { label: '.app.zip', href: `${RELEASES}/Linguae.app.zip` },
    ],
  },
  {
    name: 'Windows',
    icon: FaWindows,
    assets: [
      { label: 'MSI', href: `${RELEASES}/Linguae_0.0.1_x64_en-US.msi` },
      {
        label: 'Setup (.exe)',
        href: `${RELEASES}/Linguae_0.0.1_x64-setup.exe`,
      },
    ],
  },
  {
    name: 'Linux',
    icon: FaLinux,
    assets: [
      { label: '.deb (amd64)', href: `${RELEASES}/Linguae_0.0.1_amd64.deb` },
      {
        label: '.rpm (x86_64)',
        href: `${RELEASES}/Linguae-0.0.1-1.x86_64.rpm`,
      },
      {
        label: '.AppImage (amd64)',
        href: `${RELEASES}/Linguae_0.0.1_amd64.AppImage`,
      },
    ],
  },
];

function PlatformGrid({ platforms }: { platforms: Platform[] }) {
  return (
    <div className="grid grid-cols-1 gap-6 sm:grid-cols-3">
      {platforms.map((p) => (
        <div
          key={p.name}
          className="card border-base-300 bg-base-100 border p-6">
          <div className="bg-primary/10 text-primary mb-3.5 flex h-11 w-11 items-center justify-center rounded-lg">
            <p.icon className="h-6 w-6" />
          </div>
          <h3 className="text-base-content m-0 mb-1 text-lg leading-tight">
            {p.name}
          </h3>
          <div className="mt-4 flex flex-col gap-2">
            {p.assets.map((a) => (
              <a
                key={a.href}
                href={a.href}
                className="btn btn-primary btn-sm gap-2">
                <FaDownload className="h-3.5 w-3.5" />
                {a.label}
              </a>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}

const Downloads: FC = () => (
  <main>
    <section className="py-20">
      <div className="mx-auto w-full max-w-6xl px-5">
        <div className="mx-auto mb-12 max-w-2xl text-center">
          <h1 className="m-0 text-4xl leading-tight tracking-normal text-balance">
            Download Linguae
          </h1>
          <p className="mt-3 text-lg text-gray-500">
            Desktop app or command-line tool — language identification that runs
            entirely on your machine. Assets are in the{' '}
            <a href={RELEASE_PAGE} className="text-primary">
              latest release
            </a>
            .
          </p>
        </div>

        <h2 className="text-base-content mb-6 flex items-center gap-2 text-2xl leading-tight">
          <FaTerminal className="text-primary h-6 w-6" />
          CLI
        </h2>
        <div className="mb-12">
          <PlatformGrid platforms={CLI} />
        </div>

        <h2 className="text-base-content mb-6 flex items-center gap-2 text-2xl leading-tight">
          <FaBoxOpen className="text-primary h-6 w-6" />
          Desktop app
        </h2>
        <div className="mb-12">
          <PlatformGrid platforms={DESKTOP} />
        </div>

        <div className="border-base-300 bg-base-200/50 rounded-2xl border p-6">
          <h2 className="text-base-content m-0 mb-2 text-lg leading-tight">
            Install script
          </h2>
          <p className="text-base-content/60 mb-4 text-sm">
            Downloads the right prebuilt CLI binary, or builds from source if no
            release matches your platform. Installs to <code>~/.local/bin</code>
            .
          </p>
          <pre className="bg-base-300 overflow-x-auto rounded-lg p-4 font-mono text-sm">
            <code>{INSTALL_SH}</code>
          </pre>
        </div>
      </div>
    </section>
  </main>
);

export default Downloads;
