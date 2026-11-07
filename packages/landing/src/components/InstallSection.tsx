import type { FC } from 'react';
import TerminalWindow from './TerminalWindow';

const PREREQUISITES = ['Rust 1.85+'];

const INSTALL_LINES = [
  { prefix: '$', code: 'cd crates && cargo build --release -p linguae' },
  {
    prefix: '$',
    code: 'crates/target/release/linguae "Bonjour tout le monde"',
  },
  { code: 'fra', output: true },
];

const InstallSection: FC = () => {
  return (
    <section id="install" className="bg-base-200 py-20">
      <div className="mx-auto max-w-4xl px-4 text-center">
        <h2 className="text-3xl font-bold">Get started</h2>
        <p className="text-base-content/70 mt-4">
          Build the CLI from source, or grab the latest binary from GitHub
          Releases — no Python runtime required.
        </p>
        <div className="mt-6 flex flex-wrap justify-center gap-2">
          {PREREQUISITES.map((item) => (
            <span key={item} className="badge badge-lg">
              {item}
            </span>
          ))}
        </div>
        <div className="mx-auto mt-10 max-w-2xl text-left">
          <TerminalWindow title="zsh" lines={INSTALL_LINES} />
        </div>
        <div className="mt-8 flex flex-wrap justify-center gap-4">
          <a
            href="https://github.com/hieudoanm/linguae"
            target="_blank"
            rel="noreferrer"
            className="btn btn-primary">
            View on GitHub
          </a>
          <a
            href="https://github.com/hieudoanm/linguae/blob/master/AGENTS.md"
            target="_blank"
            rel="noreferrer"
            className="btn btn-ghost">
            Documentation
          </a>
        </div>
      </div>
    </section>
  );
};

export default InstallSection;
