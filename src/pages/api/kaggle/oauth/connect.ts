import type { NextApiRequest, NextApiResponse } from 'next';
import {
  looksLikeKaggleOAuthToken,
  readAuthFromRequest,
  validateKaggleAuth,
} from '../../../../server/kaggle';
import { applyRateLimit, enforceSameOrigin } from '../../../../server/request-guard';

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method !== 'POST') {
    res.setHeader('Allow', 'POST');
    return res.status(405).json({ error: 'Method not allowed' });
  }
  if (!enforceSameOrigin(req, res)) return;
  if (
    !applyRateLimit(req, res, {
      bucket: 'kaggle_oauth_connect',
      max: 10,
      windowMs: 60_000,
    })
  ) {
    return;
  }

  const auth = readAuthFromRequest(req);
  if (!auth || auth.mode !== 'oauth_token') {
    return res.status(400).json({
      error: 'OAuth token is required. Provide apiToken for Kaggle OAuth connect.',
    });
  }
  if (!auth.apiToken || !looksLikeKaggleOAuthToken(auth.apiToken)) {
    return res.status(400).json({
      error: 'Invalid Kaggle OAuth token format. Use a valid token/key from your Kaggle account settings.',
    });
  }

  try {
    const validation = await validateKaggleAuth(auth, { force: true });
    if (!validation.ok) {
      return res.status(validation.status).json({
        error: 'Kaggle OAuth token validation failed.',
        detail: validation.detail,
      });
    }

    return res.status(200).json({
      connected: true,
      tokenType: auth.apiToken.startsWith('KGAT') ? 'KAGGLE_API_TOKEN' : 'KAGGLE_TOKEN_OR_KEY',
      message: 'Kaggle OAuth token connected successfully.',
    });
  } catch (error: any) {
    return res.status(500).json({
      error: 'Kaggle OAuth connect failed unexpectedly.',
      detail: error?.message ?? 'Unknown server error',
    });
  }
}
