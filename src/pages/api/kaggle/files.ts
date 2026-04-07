import type { NextApiRequest, NextApiResponse } from 'next';
import {
  fetchKaggle,
  KaggleAuth,
  readAuthFromRequest,
  toDatasetRef,
  validateKaggleAuth,
} from '../../../server/kaggle';
import { applyRateLimit, enforceSameOrigin } from '../../../server/request-guard';

async function fetchFiles(datasetRef: string, auth: KaggleAuth) {
  const [owner, dataset] = datasetRef.split('/');
  const candidates = [
    `datasets/list/${owner}/${dataset}?pageSize=200`,
    `datasets/files/${owner}/${dataset}?pageSize=200`,
  ];

  for (const path of candidates) {
    const response = await fetchKaggle(path, auth);
    if (!response.ok) continue;
    const payload = await response.json();
    const files = Array.isArray(payload?.files)
      ? payload.files
      : Array.isArray(payload)
        ? payload
        : [];
    if (files.length) {
      return files.map((file: any) => ({
        name: file.name,
        totalBytes: file.totalBytes ?? file.size ?? 0,
        creationDate: file.creationDate,
      }));
    }
  }
  return [];
}

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method !== 'POST') {
    res.setHeader('Allow', 'POST');
    return res.status(405).json({ error: 'Method not allowed' });
  }
  if (!enforceSameOrigin(req, res)) return;
  if (
    !applyRateLimit(req, res, {
      bucket: 'kaggle_files',
      max: 20,
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

  const rawRef = String(req.body?.datasetRef ?? '').trim();
  if (!rawRef) return res.status(400).json({ error: 'datasetRef is required.' });

  try {
    const datasetRef = toDatasetRef(rawRef);
    const files = await fetchFiles(datasetRef, auth);
    return res.status(200).json({ datasetRef, files });
  } catch (error: any) {
    return res.status(500).json({
      error: 'Failed to fetch dataset file list.',
      detail: error?.message ?? 'Unknown server error',
    });
  }
}
