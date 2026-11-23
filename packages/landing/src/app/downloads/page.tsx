import type { Metadata } from 'next';
import type { FC, ReactNode } from 'react';
import type { IconType } from 'react-icons';
import {
  FaApple,
  FaBoxOpen,
  FaBrain,
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
const INSTALL_SH = `curl -fsSL ${GITHUB}/raw/master/scripts/install.sh`;

interface Asset {
  label: string;
  href: string;
}

interface Cell {
  name: string;
  icon: IconType;
  desc: string;
  assets: Asset[];
}

const CLI: Cell[] = [
  {
    name: 'macOS (Silicon)',
    icon: FaApple,
    desc: 'Apple Silicon Macs — M1 and later.',
    assets: [
      {
        label: 'Apple Silicon',
        href: `${RELEASES}/cli-linguae-aarch64-apple-darwin`,
      },
    ],
  },
  {
    name: 'macOS (Intel)',
    icon: FaApple,
    desc: 'Intel-based Macs.',
    assets: [
      { label: 'Intel', href: `${RELEASES}/cli-linguae-x86_64-apple-darwin` },
    ],
  },
  {
    name: 'Linux',
    icon: FaLinux,
    desc: 'x86_64 GNU/Linux distributions.',
    assets: [
      {
        label: 'x86_64',
        href: `${RELEASES}/cli-linguae-x86_64-unknown-linux-gnu`,
      },
    ],
  },
  {
    name: 'Windows',
    icon: FaWindows,
    desc: 'Windows 10/11, x86_64.',
    assets: [
      {
        label: '.exe',
        href: `${RELEASES}/cli-linguae-x86_64-pc-windows-msvc.exe`,
      },
    ],
  },
];

const DESKTOP: Cell[] = [
  {
    name: 'macOS (Silicon)',
    icon: FaApple,
    desc: 'Apple Silicon Macs — M1 and later.',
    assets: [
      {
        label: '.dmg',
        href: `${RELEASES}/desktop-linguae-aarch64-apple-darwin.dmg`,
      },
      {
        label: '.app.zip',
        href: `${RELEASES}/desktop-linguae-aarch64-apple-darwin.app.zip`,
      },
    ],
  },
  {
    name: 'macOS (Intel)',
    icon: FaApple,
    desc: 'Intel-based Macs.',
    assets: [
      {
        label: '.dmg',
        href: `${RELEASES}/desktop-linguae-x86_64-apple-darwin.dmg`,
      },
      {
        label: '.app.zip',
        href: `${RELEASES}/desktop-linguae-x86_64-apple-darwin.app.zip`,
      },
    ],
  },
  {
    name: 'Linux',
    icon: FaLinux,
    desc: 'x86_64 GNU/Linux distributions.',
    assets: [
      {
        label: '.deb',
        href: `${RELEASES}/desktop-linguae-x86_64-unknown-linux-gnu.deb`,
      },
      {
        label: '.rpm',
        href: `${RELEASES}/desktop-linguae-x86_64-unknown-linux-gnu.rpm`,
      },
      {
        label: '.AppImage',
        href: `${RELEASES}/desktop-linguae-x86_64-unknown-linux-gnu.AppImage`,
      },
    ],
  },
  {
    name: 'Windows',
    icon: FaWindows,
    desc: 'Windows 10/11, x86_64.',
    assets: [
      {
        label: 'MSI',
        href: `${RELEASES}/desktop-linguae-x86_64-pc-windows-msvc.msi`,
      },
      {
        label: 'Setup (.exe)',
        href: `${RELEASES}/desktop-linguae-x86_64-pc-windows-msvc-setup.exe`,
      },
    ],
  },
];

const MODELS: Cell[] = [
  {
    name: 'joblib (scikit-learn)',
    icon: FaBrain,
    desc: 'scikit-learn joblib export of the CharCNN weights.',
    assets: [{ label: 'joblib', href: `${RELEASES}/model-linguae.joblib` }],
  },
  {
    name: 'pkl (pickle)',
    icon: FaBrain,
    desc: 'Python pickle — config, numpy weights, vocab, labels.',
    assets: [{ label: 'pickle', href: `${RELEASES}/model-linguae.pkl` }],
  },
  {
    name: 'pt (PyTorch)',
    icon: FaBrain,
    desc: 'PyTorch checkpoint — config and state_dict.',
    assets: [{ label: 'PyTorch', href: `${RELEASES}/model-linguae.pt` }],
  },
  {
    name: 'CLI artifact',
    icon: FaBoxOpen,
    desc: 'Full artifact bundle the linguae CLI loads at runtime.',
    assets: [
      { label: 'Artifact (.zip)', href: `${RELEASES}/cli-linguae-model.zip` },
    ],
  },
];

function Section({
  icon: Icon,
  title,
  children,
}: {
  icon: IconType;
  title: string;
  children: ReactNode;
}) {
  return (
    <div className="mb-12 last:mb-0">
      <h2 className="text-base-content mb-6 flex items-center gap-2 text-2xl leading-tight">
        <Icon className="text-primary h-6 w-6" />
        {title}
      </h2>
      {children}
    </div>
  );
}

function CardGrid({ cells, cols }: { cells: Cell[]; cols: string }) {
  const visible = cells.filter((c) => c.assets.length > 0);
  if (visible.length === 0) return null;
  return (
    <div className={`grid grid-cols-1 gap-6 sm:grid-cols-2 ${cols}`}>
      {visible.map((c) => (
        <div
          key={c.name}
          className="card border-base-300 bg-base-100 border p-6">
          <div className="bg-primary/10 text-primary mb-3.5 flex h-11 w-11 items-center justify-center rounded-lg">
            <c.icon className="h-6 w-6" />
          </div>
          <h3 className="text-base-content m-0 text-lg leading-tight">
            {c.name}
          </h3>
          <p className="text-base-content/60 mt-1 truncate text-sm">{c.desc}</p>
          <div className="mt-4 flex flex-col gap-2">
            {c.assets.map((a) => (
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
            entirely on your machine.
          </p>
          <p className="mt-3 text-lg text-gray-500">
            Assets are in the{' '}
            <a href={RELEASE_PAGE} className="text-primary">
              latest release
            </a>
            .
          </p>
        </div>

        <Section icon={FaTerminal} title="CLI">
          <CardGrid cells={CLI} cols="lg:grid-cols-4" />
        </Section>

        <Section icon={FaBoxOpen} title="Desktop">
          <CardGrid cells={DESKTOP} cols="lg:grid-cols-4" />
        </Section>

        {MODELS.length > 0 && (
          <Section icon={FaBrain} title="Models">
            <CardGrid cells={MODELS} cols="lg:grid-cols-3" />
          </Section>
        )}

        <div className="border-base-300 bg-base-200/50 mt-12 rounded-2xl border p-6">
          <h2 className="text-base-content m-0 mb-2 flex items-center gap-2 text-lg leading-tight">
            <FaApple className="text-primary h-5 w-5" />
            macOS: unsigned app workaround
          </h2>
          <p className="text-base-content/60 mb-4 text-sm">
            The macOS builds are not signed or notarized, so Gatekeeper will
            block the first launch. Either right-click the app and choose{' '}
            <strong>Open</strong>, or remove the quarantine attribute:
          </p>
          <pre className="bg-base-300 overflow-x-auto rounded-lg p-4 font-mono text-sm">
            <code>
              xattr -dr com.apple.quarantine /Applications/Linguae.app
            </code>
          </pre>
        </div>

        <div className="border-base-300 bg-base-200/50 rounded-2xl border p-6">
          <h2 className="text-base-content m-0 mb-2 flex items-center gap-2 text-lg leading-tight">
            <FaTerminal className="text-primary h-5 w-5" />
            Install script
          </h2>
          <p className="text-base-content/60 mb-4 text-sm">
            Downloads the right prebuilt binary, or builds from source if no
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
