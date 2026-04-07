import type { NextApiRequest, NextApiResponse } from 'next';

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method !== 'GET') {
    res.setHeader('Allow', 'GET');
    return res.status(405).json({ error: 'Method not allowed' });
  }

  return res.status(200).json({
    authorizeUrl: 'https://www.kaggle.com/settings',
    message:
      'Kaggle OAuth for public API uses account token generation. Open settings and generate API token.',
    steps: [
      'Open Kaggle settings.',
      'Under API section click Generate New Token.',
      'Copy token value from downloaded file (or use KAGGLE_API_TOKEN).',
      'Paste token into WebKaya OAuth connect.',
    ],
  });
}
