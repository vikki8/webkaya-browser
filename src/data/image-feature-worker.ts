import { DataRow } from '../types/data';

interface ImageWorkerRequest {
  type: 'process_image';
  taskId: number;
  entryName: string;
  label: string;
  imageBuffer: ArrayBuffer;
  includePreview?: boolean;
}

interface ImageWorkerSuccess {
  type: 'process_image_success';
  taskId: number;
  row: DataRow;
}

interface ImageWorkerFailure {
  type: 'process_image_error';
  taskId: number;
  error: string;
}

type ImageWorkerResponse = ImageWorkerSuccess | ImageWorkerFailure;

function computeImageStats(pixelData: Uint8ClampedArray): Record<string, number> {
  let sumR = 0;
  let sumG = 0;
  let sumB = 0;
  let sqR = 0;
  let sqG = 0;
  let sqB = 0;
  const px = pixelData.length / 4;
  for (let i = 0; i < pixelData.length; i += 4) {
    const r = pixelData[i] / 255;
    const g = pixelData[i + 1] / 255;
    const b = pixelData[i + 2] / 255;
    sumR += r;
    sumG += g;
    sumB += b;
    sqR += r * r;
    sqG += g * g;
    sqB += b * b;
  }
  const meanR = sumR / px;
  const meanG = sumG / px;
  const meanB = sumB / px;
  return {
    mean_r: meanR,
    mean_g: meanG,
    mean_b: meanB,
    std_r: Math.sqrt(Math.max(0, sqR / px - meanR * meanR)),
    std_g: Math.sqrt(Math.max(0, sqG / px - meanG * meanG)),
    std_b: Math.sqrt(Math.max(0, sqB / px - meanB * meanB)),
  };
}

function bytesToBase64(bytes: Uint8Array): string {
  let binary = '';
  const chunkSize = 0x8000;
  for (let i = 0; i < bytes.length; i += chunkSize) {
    const chunk = bytes.subarray(i, i + chunkSize);
    binary += String.fromCharCode(...chunk);
  }
  return btoa(binary);
}

async function decodeImageToFeatures(
  buffer: ArrayBuffer,
  includePreview: boolean
): Promise<{ stats: Record<string, number>; previewDataUrl: string | null }> {
  if (typeof OffscreenCanvas === 'undefined') {
    throw new Error('OffscreenCanvas is unavailable in this browser worker.');
  }
  const blob = new Blob([buffer]);
  const bitmap = await createImageBitmap(blob);
  try {
    const width = 32;
    const height = 32;
    const canvas = new OffscreenCanvas(width, height);
    const ctx = canvas.getContext('2d');
    if (!ctx) {
      throw new Error('Unable to create 2D context for image preprocessing.');
    }
    ctx.drawImage(bitmap, 0, 0, width, height);
    const { data } = ctx.getImageData(0, 0, width, height);
    const stats = computeImageStats(data);
    let previewDataUrl: string | null = null;
    if (includePreview) {
      const previewBlob = await canvas.convertToBlob({ type: 'image/png' });
      const previewBytes = new Uint8Array(await previewBlob.arrayBuffer());
      previewDataUrl = `data:image/png;base64,${bytesToBase64(previewBytes)}`;
    }
    return { stats, previewDataUrl };
  } finally {
    bitmap.close();
  }
}

const workerScope = self as unknown as DedicatedWorkerGlobalScope;

workerScope.onmessage = async (event: MessageEvent<ImageWorkerRequest>) => {
  const message = event.data;
  if (!message || message.type !== 'process_image') return;
  try {
    const { stats, previewDataUrl } = await decodeImageToFeatures(message.imageBuffer, Boolean(message.includePreview));
    const response: ImageWorkerResponse = {
      type: 'process_image_success',
      taskId: message.taskId,
      row: {
        label: message.label,
        image_name: message.entryName,
        image_preview: previewDataUrl,
        ...stats,
      },
    };
    workerScope.postMessage(response);
  } catch (error: any) {
    const response: ImageWorkerResponse = {
      type: 'process_image_error',
      taskId: message.taskId,
      error: error?.message ?? 'Image processing failed in worker.',
    };
    workerScope.postMessage(response);
  }
};

export {};
