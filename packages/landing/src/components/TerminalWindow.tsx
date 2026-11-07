import type { FC } from 'react';

export type TerminalLine = {
  prefix?: string | null;
  code: string;
  output?: boolean;
};

type TerminalWindowProps = {
  title: string;
  lines: TerminalLine[];
};

const TerminalWindow: FC<TerminalWindowProps> = ({ title, lines }) => {
  return (
    <figure className="w-full">
      <figcaption className="text-base-content/60 mb-2 font-mono text-xs">
        {title}
      </figcaption>
      <div className="mockup-code w-full">
        {lines.map((line, index) => (
          <pre key={index} data-prefix={line.prefix ?? undefined}>
            <code className={line.output ? 'text-emerald-300' : ''}>
              {line.code}
            </code>
          </pre>
        ))}
      </div>
    </figure>
  );
};

export default TerminalWindow;
