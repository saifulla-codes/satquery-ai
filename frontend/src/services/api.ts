import type { AnalysisResponse, ChangeDetectionResponse, OpticalSarResponse, SampleDataset } from '../types';

export const API_HOST = (import.meta.env.VITE_API_URL || '').replace(/\/$/, '');
export const API_BASE = `${API_HOST}/api`;

export function resolveAssetUrl(url?: string): string {
  if (!url) return '';
  if (url.startsWith('http://') || url.startsWith('https://')) return url;
  if (API_HOST && url.startsWith('/')) {
    return `${API_HOST}${url}`;
  }
  return url;
}

export async function checkHealth(): Promise<any> {
  const res = await fetch(`${API_BASE}/health`);
  if (!res.ok) throw new Error('Backend offline');
  return res.json();
}

export async function fetchSamples(): Promise<SampleDataset[]> {
  const res = await fetch(`${API_BASE}/samples`);
  if (!res.ok) throw new Error('Failed to load sample datasets');
  return res.json();
}

export async function uploadSatelliteImage(file: File): Promise<any> {
  const formData = new FormData();
  formData.append('file', file);

  const res = await fetch(`${API_BASE}/upload`, {
    method: 'POST',
    body: formData,
  });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.error || 'Image upload failed');
  }
  return res.json();
}

export async function analyzeQuery(
  imageId: string,
  query: string,
  taskHint?: string
): Promise<AnalysisResponse> {
  const res = await fetch(`${API_BASE}/analyze`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ image_id: imageId, query, task_hint: taskHint }),
  });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.error || 'Analysis failed');
  }
  return res.json();
}

export async function analyzeChangeDetection(
  imageBeforeId: string,
  imageAfterId: string,
  query?: string
): Promise<ChangeDetectionResponse> {
  const res = await fetch(`${API_BASE}/change-detection`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      image_before_id: imageBeforeId,
      image_after_id: imageAfterId,
      query: query || 'What changed between these two images?',
    }),
  });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.error || 'Change detection failed');
  }
  return res.json();
}

export async function analyzeOpticalSarPair(
  opticalId: string,
  sarId: string,
  query?: string
): Promise<OpticalSarResponse> {
  const res = await fetch(`${API_BASE}/optical-sar`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      optical_image_id: opticalId,
      sar_image_id: sarId,
      query: query || 'Compare optical and SAR evidence to identify development.',
    }),
  });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.error || 'Optical-SAR fusion failed');
  }
  return res.json();
}
