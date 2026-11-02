/**
 * Minimal leveled logger.
 *
 * WHY: the project forbids `console.log` in production. Routing everything
 * through one module gives a single place to enforce that (ESLint's
 * `no-console` is switched off for this file only) and to silence noisy
 * levels in production builds.
 *
 * ASSUMPTION: production keeps `warn` and `error` so real failures (camera
 * denied, model load failure) remain visible in DevTools; `debug` and `info`
 * are dev-only.
 */

export type LogLevel = 'debug' | 'info' | 'warn' | 'error';

const LEVEL_RANK: Record<LogLevel, number> = {
  debug: 10,
  info: 20,
  warn: 30,
  error: 40,
};

const MIN_LEVEL: LogLevel = import.meta.env.DEV ? 'debug' : 'warn';

export interface Logger {
  debug: (message: string, ...details: unknown[]) => void;
  info: (message: string, ...details: unknown[]) => void;
  warn: (message: string, ...details: unknown[]) => void;
  error: (message: string, ...details: unknown[]) => void;
}

function emit(level: LogLevel, scope: string, message: string, details: unknown[]): void {
  if (LEVEL_RANK[level] < LEVEL_RANK[MIN_LEVEL]) return;
  const prefix = `[${scope}] ${message}`;
  // console.debug/info/warn/error are intentionally used instead of console.log.
  console[level](prefix, ...details);
}

/**
 * Creates a logger whose lines are prefixed with `scope`, e.g. "vision/hands".
 */
export function createLogger(scope: string): Logger {
  return {
    debug: (message, ...details) => {
      emit('debug', scope, message, details);
    },
    info: (message, ...details) => {
      emit('info', scope, message, details);
    },
    warn: (message, ...details) => {
      emit('warn', scope, message, details);
    },
    error: (message, ...details) => {
      emit('error', scope, message, details);
    },
  };
}
