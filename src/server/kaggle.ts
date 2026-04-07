import { NextApiRequest } from 'next';

export interface KaggleAuth {
  mode: 'oauth_token' | 'legacy_key';
  username?: string;
  apiKey?: string;
  apiToken?: string;
}

export interface KaggleAuthValidationResult {
  ok: boolean;
  status: number;
  detail?: string;
}

type CachedValidation = KaggleAuthValidationResult & {
  expiresAt: number;
};

const AUTH_VALIDATION_TTL_MS = 60_000;
const DEFAULT_KAGGLE_TIMEOUT_MS = 20_000;
const authValidationCache = new Map<string, CachedValidation>();

export function readAuthFromRequest(req: NextApiRequest): KaggleAuth | null {
  const apiTokenRaw =
    (req.body?.apiToken as string | undefined) ??
    (req.headers['x-kaggle-api-token'] as string | undefined) ??
    (req.headers.authorization?.toLowerCase().startsWith('bearer ')
      ? req.headers.authorization.slice(7)
      : undefined);
  const apiToken = apiTokenRaw?.trim();
  if (apiToken) {
    return { mode: 'oauth_token', apiToken };
  }

  const usernameRaw =
    (req.body?.username as string | undefined) ??
    (req.headers['x-kaggle-username'] as string | undefined);
  const apiKeyRaw =
    (req.body?.apiKey as string | undefined) ??
    (req.headers['x-kaggle-key'] as string | undefined);
  const username = usernameRaw?.trim();
  const apiKey = apiKeyRaw?.trim();

  if (!username || !apiKey) return null;
  return { mode: 'legacy_key', username, apiKey };
}

export function kaggleAuthorization(auth: KaggleAuth): string {
  if (auth.mode === 'oauth_token') {
    return `Bearer ${auth.apiToken}`;
  }
  const token = Buffer.from(`${auth.username}:${auth.apiKey}`).toString('base64');
  return `Basic ${token}`;
}

export async function fetchKaggle(
  path: string,
  auth: KaggleAuth,
  options?: { timeoutMs?: number }
): Promise<Response> {
  const timeoutMs = Math.max(1_000, options?.timeoutMs ?? DEFAULT_KAGGLE_TIMEOUT_MS);
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(`https://www.kaggle.com/api/v1/${path}`, {
      headers: {
        Authorization: kaggleAuthorization(auth),
        Accept: '*/*',
        'User-Agent': 'BrowserFirstAI-Platform/0.1.0',
      },
      signal: controller.signal,
    });
  } catch (error: any) {
    if (error?.name === 'AbortError') {
      throw new Error(`Kaggle API request timed out after ${timeoutMs}ms.`);
    }
    throw error;
  } finally {
    clearTimeout(timer);
  }
}

function validationCacheKey(auth: KaggleAuth): string {
  return auth.mode === 'oauth_token'
    ? `oauth:${auth.apiToken}`
    : `legacy:${auth.username}:${auth.apiKey}`;
}

export function looksLikeKaggleOAuthToken(value: string): boolean {
  const token = value.trim();
  if (!token) return false;
  return /^[A-Za-z0-9._-]{16,}$/.test(token);
}

async function readResponseDetail(response: Response): Promise<string | undefined> {
  try {
    const text = await response.text();
    return text.slice(0, 500);
  } catch {
    return undefined;
  }
}

async function strictAuthProbe(auth: KaggleAuth): Promise<KaggleAuthValidationResult> {
  const candidates = [
    // Auth-gated endpoint. Invalid credentials return 401/403.
    'kernels/list?page=1&pageSize=1',
    // Secondary auth-gated probe fallback.
    'datasets/status/uciml/iris',
  ];

  let lastFailure: KaggleAuthValidationResult | null = null;
  for (const path of candidates) {
    const response = await fetchKaggle(path, auth);
    if (response.ok) {
      try {
        await response.body?.cancel();
      } catch {
        // ignore; probe already succeeded
      }
      return { ok: true, status: 200 };
    }

    const detail = await readResponseDetail(response);
    if (response.status === 401 || response.status === 403) {
      return {
        ok: false,
        status: response.status,
        detail,
      };
    }

    lastFailure = {
      ok: false,
      status: response.status,
      detail,
    };
  }

  return (
    lastFailure ?? {
      ok: false,
      status: 500,
      detail: 'Auth probe failed unexpectedly.',
    }
  );
}

/**
 * Uses auth-gated Kaggle API probes so invalid credentials cannot pass validation.
 */
export async function validateKaggleAuth(
  auth: KaggleAuth,
  options?: { force?: boolean }
): Promise<KaggleAuthValidationResult> {
  const key = validationCacheKey(auth);
  const now = Date.now();
  const cached = authValidationCache.get(key);
  if (!options?.force && cached && cached.expiresAt > now) {
    return { ok: cached.ok, status: cached.status, detail: cached.detail };
  }

  const strictProbe = await strictAuthProbe(auth);
  if (!strictProbe.ok) {
    authValidationCache.set(key, { ...strictProbe, expiresAt: now + AUTH_VALIDATION_TTL_MS });
    return strictProbe;
  }

  const result = { ok: true, status: 200 } satisfies KaggleAuthValidationResult;
  authValidationCache.set(key, { ...result, expiresAt: now + AUTH_VALIDATION_TTL_MS });
  return result;
}

export function toDatasetRef(input: string): string {
  const trimmed = input.trim();
  if (trimmed.includes('/datasets/')) {
    const match = trimmed.match(/\/datasets\/([^/]+\/[^/?#]+)/i);
    if (!match) throw new Error('Invalid Kaggle dataset URL.');
    return match[1];
  }
  if (!/^[a-z0-9_.-]+\/[a-z0-9_.-]+$/i.test(trimmed)) {
    throw new Error('Dataset reference must look like "owner/dataset".');
  }
  return trimmed;
}
