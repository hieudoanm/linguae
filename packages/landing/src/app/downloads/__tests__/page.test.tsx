import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';
import Downloads, { metadata } from '../page';

const html = renderToStaticMarkup(<Downloads />);
const RELEASES =
  'https://github.com/hieudoanm/linguae/releases/latest/download';

describe('downloads page', () => {
  it('exports downloads metadata', () => {
    expect(metadata.title).toBe('Downloads');
    expect(metadata.description).toBeTruthy();
  });

  it('renders the intro with a link to the latest release', () => {
    expect(html).toContain(
      'https://github.com/hieudoanm/linguae/releases/latest'
    );
    expect(html).toContain('latest release');
  });

  it('renders the expected sections', () => {
    expect(html).toContain('CLI</h2>');
    expect(html).toContain('Desktop</h2>');
  });

  it('hides sections that have no assets', () => {
    expect(html).not.toContain('Editor Extensions</h2>');
    expect(html).not.toContain('Browser Extensions</h2>');
  });

  it('links every release asset', () => {
    expect(html).toContain(`${RELEASES}/cli-linguae-aarch64-apple-darwin`);
    expect(html).toContain(`${RELEASES}/cli-linguae-x86_64-apple-darwin`);
    expect(html).toContain(
      `${RELEASES}/cli-linguae-x86_64-pc-windows-msvc.exe`
    );
    expect(html).toContain(`${RELEASES}/cli-linguae-x86_64-unknown-linux-gnu`);
    expect(html).toContain(
      `${RELEASES}/desktop-linguae-aarch64-apple-darwin.app.zip`
    );
    expect(html).toContain(
      `${RELEASES}/desktop-linguae-aarch64-apple-darwin.dmg`
    );
    expect(html).toContain(
      `${RELEASES}/desktop-linguae-x86_64-apple-darwin.app.zip`
    );
    expect(html).toContain(
      `${RELEASES}/desktop-linguae-x86_64-apple-darwin.dmg`
    );
    expect(html).toContain(
      `${RELEASES}/desktop-linguae-x86_64-pc-windows-msvc-setup.exe`
    );
    expect(html).toContain(
      `${RELEASES}/desktop-linguae-x86_64-pc-windows-msvc.msi`
    );
    expect(html).toContain(
      `${RELEASES}/desktop-linguae-x86_64-unknown-linux-gnu.AppImage`
    );
    expect(html).toContain(
      `${RELEASES}/desktop-linguae-x86_64-unknown-linux-gnu.deb`
    );
    expect(html).toContain(
      `${RELEASES}/desktop-linguae-x86_64-unknown-linux-gnu.rpm`
    );
  });
});
