import { assetPath } from '../site';

describe('assetPath', () => {
  it('prefixes asset paths with the base path', () => {
    // NODE_ENV is 'test' under vitest, so BASE_PATH is empty.
    expect(assetPath('/svg/logo.svg')).toBe('/svg/logo.svg');
  });
});
