import type { FC } from 'react';
import Image from 'next/image';
import { assetPath } from '@/lib/site';

const GITHUB_URL = 'https://github.com/hieudoanm/linguae';
const REPO_BLOB_URL = `${GITHUB_URL}/blob/master`;

const FOOTER_COLUMNS = [
  {
    title: 'Product',
    links: [
      { label: 'Features', href: '#features' },
      { label: 'Demo', href: '#demo' },
      { label: 'How it works', href: '#how-it-works' },
      { label: 'Install', href: '#install' },
    ],
  },
  {
    title: 'Resources',
    links: [
      { label: 'GitHub', href: GITHUB_URL, external: true },
      {
        label: 'Documentation',
        href: `${REPO_BLOB_URL}/AGENTS.md`,
        external: true,
      },
      {
        label: 'CLI reference',
        href: `${REPO_BLOB_URL}/crates/README.md`,
        external: true,
      },
      {
        label: 'Training pipeline',
        href: `${REPO_BLOB_URL}/crates/linguae-train/README.md`,
        external: true,
      },
    ],
  },
  {
    title: 'Legal',
    links: [
      {
        label: 'MIT License',
        href: `${REPO_BLOB_URL}/LICENSE`,
        external: true,
      },
    ],
  },
];

const Footer: FC = () => {
  return (
    <footer className="border-base-300 bg-base-100 border-t px-4 py-12">
      <div className="mx-auto grid w-full max-w-5xl gap-10 sm:grid-cols-2 lg:grid-cols-4">
        <div>
          <div className="text-base-content flex items-center gap-2 font-bold">
            <Image
              src={assetPath('/svg/logo-light.svg')}
              alt=""
              width={24}
              height={24}
              className="size-6 rounded dark:hidden"
            />
            <Image
              src={assetPath('/svg/logo.svg')}
              alt=""
              width={24}
              height={24}
              className="hidden size-6 rounded dark:block"
            />
            Linguae
          </div>
          <p className="text-base-content/70 mt-3 text-sm">
            Local-first language identification.
            <br />
            No cloud, no pretrained weights.
          </p>
        </div>
        {FOOTER_COLUMNS.map((column) => (
          <div key={column.title}>
            <span className="text-base-content mb-3 block text-sm font-semibold tracking-wide uppercase">
              {column.title}
            </span>
            <ul className="space-y-2">
              {column.links.map((link) => (
                <li key={link.label}>
                  <a
                    href={link.href}
                    className="link link-hover text-base-content/70"
                    {...('external' in link && link.external
                      ? { target: '_blank', rel: 'noreferrer' }
                      : {})}>
                    {link.label}
                  </a>
                </li>
              ))}
            </ul>
          </div>
        ))}
      </div>
      <div className="border-base-300 text-base-content/60 mx-auto mt-10 w-full max-w-5xl border-t pt-6 text-center text-sm">
        © {new Date().getFullYear()} Linguae — Built with Rust, Candle, Next.js
        and daisyUI.
      </div>
    </footer>
  );
};

export default Footer;
