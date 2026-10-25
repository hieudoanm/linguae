import { type FC } from 'react';

const Footer: FC = () => {
  return (
    <footer className="border-t border-gray-200 py-4 text-center dark:border-gray-700">
      <p className="text-sm text-gray-500">
        © {new Date().getFullYear()} Linguae
      </p>
    </footer>
  );
};

export default Footer;
