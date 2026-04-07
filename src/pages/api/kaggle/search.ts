import type { NextApiRequest, NextApiResponse } from 'next';
import { fetchKaggle, readAuthFromRequest, validateKaggleAuth } from '../../../server/kaggle';
import { applyRateLimit, enforceSameOrigin } from '../../../server/request-guard';

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method !== 'POST') {
    res.setHeader('Allow', 'POST');
    return res.status(405).json({ error: 'Method not allowed' });
  }
  if (!enforceSameOrigin(req, res)) return;
  if (
    !applyRateLimit(req, res, {
      bucket: 'kaggle_search',
      max: 30,
      windowMs: 60_000,
    })
  ) {
    return;
  }

  const auth = readAuthFromRequest(req);
  if (!auth) {
    return res.status(400).json({
      error: 'Kaggle authentication is required (OAuth token or username+API key).',
    });
  }
  const authValidation = await validateKaggleAuth(auth);
  if (!authValidation.ok) {
    return res.status(authValidation.status).json({
      error: 'Kaggle authentication is invalid. Please reconnect OAuth/API key.',
      detail: authValidation.detail,
    });
  }

  const query = String(req.body?.query ?? '').trim();
  if (!query) return res.status(400).json({ error: 'Search query is required.' });

  try {
    const response = await fetchKaggle(
      `datasets/list?search=${encodeURIComponent(query)}&page=1&pageSize=20`,
      auth
    );
    if (!response.ok) {
      const errorText = await response.text();
      return res.status(response.status).json({
        error: 'Kaggle search failed.',
        detail: errorText.slice(0, 500),
      });
    }

    const datasets = (await response.json()) as any[];
    return res.status(200).json({
      results: datasets.map((dataset) => ({
        ref: dataset.ref,
        title: dataset.title,
        subtitle: dataset.subtitle,
        owner: dataset.ownerName ?? dataset.ownerSlug,
        totalBytes: dataset.totalBytes,
        downloadCount: dataset.downloadCount,
        voteCount: dataset.voteCount,
        lastUpdated: dataset.lastUpdated,
      })),
    });
  } catch (error: any) {
    return res.status(500).json({
      error: 'Kaggle search failed unexpectedly.',
      detail: error?.message ?? 'Unknown server error',
    });
  }
}
