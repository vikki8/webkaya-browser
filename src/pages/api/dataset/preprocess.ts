import type { NextApiRequest, NextApiResponse } from 'next';
import { preprocessDataset } from '../../../data/preprocess';
import { DataRow, PreprocessingConfig } from '../../../types/data';
import { applyRateLimit, enforceSameOrigin } from '../../../server/request-guard';

interface PreprocessRequestBody {
  rows?: DataRow[];
  columns?: string[];
  config?: PreprocessingConfig;
  previewOnly?: boolean;
}

export const config = {
  api: {
    bodyParser: {
      sizeLimit: '250mb',
    },
    responseLimit: false,
  },
};

const MAX_PREPROCESS_ROWS = 200_000;
const MAX_PREPROCESS_COLUMNS = 2_048;
const MAX_PREPROCESS_CELLS = 25_000_000;

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method !== 'POST') {
    res.setHeader('Allow', 'POST');
    return res.status(405).json({ error: 'Method not allowed' });
  }
  if (!enforceSameOrigin(req, res)) return;
  if (
    !applyRateLimit(req, res, {
      bucket: 'dataset_preprocess',
      max: 10,
      windowMs: 60_000,
    })
  ) {
    return;
  }

  try {
    const body = (req.body ?? {}) as PreprocessRequestBody;
    const rows = Array.isArray(body.rows) ? body.rows : [];
    const columns = Array.isArray(body.columns) ? body.columns : [];
    const preprocessConfig = body.config;
    const previewOnly = Boolean(body.previewOnly);

    if (!rows.length) {
      return res.status(400).json({ error: 'Dataset rows are required for preprocessing.' });
    }
    if (!columns.length) {
      return res.status(400).json({ error: 'Dataset columns are required for preprocessing.' });
    }
    if (!preprocessConfig) {
      return res.status(400).json({ error: 'Preprocessing config is required.' });
    }
    if (rows.length > MAX_PREPROCESS_ROWS) {
      return res.status(413).json({
        error: `Dataset exceeds preprocessing row limit (${MAX_PREPROCESS_ROWS.toLocaleString()}).`,
      });
    }
    if (columns.length > MAX_PREPROCESS_COLUMNS) {
      return res.status(413).json({
        error: `Dataset exceeds preprocessing column limit (${MAX_PREPROCESS_COLUMNS.toLocaleString()}).`,
      });
    }
    if (rows.length * columns.length > MAX_PREPROCESS_CELLS) {
      return res.status(413).json({
        error: `Dataset exceeds preprocessing cell limit (${MAX_PREPROCESS_CELLS.toLocaleString()}).`,
      });
    }

    const result = preprocessDataset(rows, columns, preprocessConfig, {
      previewOnly,
      allowInPlace: true,
    });
    return res.status(200).json(result);
  } catch (error: any) {
    return res.status(500).json({
      error: 'Server preprocessing failed.',
      detail: error?.message ?? 'Unknown server error',
    });
  }
}
