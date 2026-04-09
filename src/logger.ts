// NCAAM Oracle v4.1 — Logger (pino)

import pino from 'pino';
import { mkdirSync } from 'fs';
import { resolve } from 'path';

mkdirSync(resolve('logs'), { recursive: true });

const isCI = process.env.CI === 'true';
const logLevel = process.env.LOG_LEVEL ?? 'info';

export const logger = pino(
  { level: logLevel },
  isCI
    ? pino.destination(1)
    : pino.transport({
        targets: [
          {
            target: 'pino-pretty',
            options: { colorize: true, translateTime: 'SYS:HH:MM:ss', ignore: 'pid,hostname' },
            level: logLevel,
          },
          {
            target: 'pino/file',
            options: { destination: resolve('logs/ncaam-oracle.log'), mkdir: true },
            level: 'debug',
          },
        ],
      })
);
