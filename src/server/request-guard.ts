import { NextApiRequest, NextApiResponse } from 'next';

interface RateLimitOptions {
  bucket: string;
  max: number;
  windowMs: number;
}

interface RateLimitEntry {
  count: number;
  resetAt: number;
}

const rateLimitBuckets = new Map<string, Map<string, RateLimitEntry>>();

function normalizeHeaderValue(value: string | string[] | undefined): string | undefined {
  if (!value) return undefined;
  if (Array.isArray(value)) return value[0];
  return value;
}

export function getClientIdentifier(req: NextApiRequest): string {
  const forwardedFor = normalizeHeaderValue(req.headers['x-forwarded-for']);
  if (forwardedFor) {
    const first = forwardedFor.split(',')[0]?.trim();
    if (first) return first;
  }
  const realIp = normalizeHeaderValue(req.headers['x-real-ip']);
  if (realIp) return realIp;
  return req.socket.remoteAddress || 'unknown';
}

function pruneBucket(bucketState: Map<string, RateLimitEntry>, now: number) {
  for (const [key, entry] of bucketState) {
    if (entry.resetAt <= now) {
      bucketState.delete(key);
    }
  }
}

export function applyRateLimit(req: NextApiRequest, res: NextApiResponse, options: RateLimitOptions): boolean {
  const now = Date.now();
  let bucketState = rateLimitBuckets.get(options.bucket);
  if (!bucketState) {
    bucketState = new Map<string, RateLimitEntry>();
    rateLimitBuckets.set(options.bucket, bucketState);
  }

  pruneBucket(bucketState, now);
  const key = getClientIdentifier(req);
  const current = bucketState.get(key);
  if (!current || current.resetAt <= now) {
    const next: RateLimitEntry = {
      count: 1,
      resetAt: now + options.windowMs,
    };
    bucketState.set(key, next);
    res.setHeader('X-RateLimit-Limit', String(options.max));
    res.setHeader('X-RateLimit-Remaining', String(Math.max(0, options.max - 1)));
    res.setHeader('X-RateLimit-Reset', String(Math.ceil(next.resetAt / 1000)));
    return true;
  }

  current.count += 1;
  const remaining = Math.max(0, options.max - current.count);
  res.setHeader('X-RateLimit-Limit', String(options.max));
  res.setHeader('X-RateLimit-Remaining', String(remaining));
  res.setHeader('X-RateLimit-Reset', String(Math.ceil(current.resetAt / 1000)));
  if (current.count > options.max) {
    const retryAfterSeconds = Math.max(1, Math.ceil((current.resetAt - now) / 1000));
    res.setHeader('Retry-After', String(retryAfterSeconds));
    res.status(429).json({
      error: 'Rate limit exceeded. Please retry shortly.',
      retryAfterSeconds,
    });
    return false;
  }
  return true;
}

function extractRequestHost(req: NextApiRequest): string | null {
  const forwardedHost = normalizeHeaderValue(req.headers['x-forwarded-host']);
  const host = forwardedHost || normalizeHeaderValue(req.headers.host);
  return host ? host.toLowerCase() : null;
}

function extractOriginHost(req: NextApiRequest): string | null {
  const origin = normalizeHeaderValue(req.headers.origin);
  if (!origin) return null;
  try {
    return new URL(origin).host.toLowerCase();
  } catch {
    return null;
  }
}

function extractRefererHost(req: NextApiRequest): string | null {
  const referer = normalizeHeaderValue(req.headers.referer);
  if (!referer) return null;
  try {
    return new URL(referer).host.toLowerCase();
  } catch {
    return null;
  }
}

export function enforceSameOrigin(req: NextApiRequest, res: NextApiResponse): boolean {
  const requestHost = extractRequestHost(req);
  if (!requestHost) return true;

  const originHost = extractOriginHost(req);
  if (originHost && originHost !== requestHost) {
    res.status(403).json({ error: 'Cross-origin request blocked.' });
    return false;
  }

  const refererHost = extractRefererHost(req);
  if (!originHost && refererHost && refererHost !== requestHost) {
    res.status(403).json({ error: 'Cross-origin request blocked.' });
    return false;
  }

  return true;
}
