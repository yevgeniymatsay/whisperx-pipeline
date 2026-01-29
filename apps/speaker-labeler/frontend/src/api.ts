import axios from 'axios';

const api = axios.create({
  baseURL: 'http://localhost:8000',
});

export interface Turn {
  spk: string;
  t0_abs?: number;
  t1_abs?: number;
  t0?: number;
  t1?: number;
  text: string;
}

export interface CallData {
  call_id: string;
  video_id: string;
  audio_url: string | null;
  start_time_s: number;
  duration_s: number;
  speakers: string[];
  turns: Turn[];
}

export interface QueueItem {
  call_id: string;
  video_id: string;
  run_id: string;
  s3_key: string;
  status: 'pending' | 'labeled' | 'skipped';
}

export interface Progress {
  total: number;
  labeled: number;
  skipped: number;
  remaining: number;
  current_index: number;
}

export interface CurrentCall {
  done: boolean;
  message?: string;
  index?: number;
  total?: number;
  call_id?: string;
  video_id?: string;
}

export const getProgress = () => api.get<Progress>('/api/progress').then(r => r.data);
export const getCurrentCall = () => api.get<CurrentCall>('/api/queue/current').then(r => r.data);
export const getCallData = (callId: string) => api.get<CallData>(`/api/calls/${callId}`).then(r => r.data);
export const saveLabels = (callId: string, speakerRoles: Record<string, string>) =>
  api.post(`/api/calls/${callId}/labels`, { speaker_roles: speakerRoles });
export const skipCall = (callId: string) => api.post(`/api/calls/${callId}/skip`);
export const navigate = (direction: 'prev' | 'next') => api.post(`/api/navigate/${direction}`);

// Boundary Editor API
export interface VideoInfo {
  video_id: string;
  call_count: number;
  has_corrections: boolean;
}

export interface AutoBoundary {
  call_id: string;
  start_s: number;
  end_s: number;
  type: 'auto';
}

export interface CorrectedBoundary {
  call_index: number;
  start_s: number;
  end_s: number;
  corrected_at: string;
}

export interface VideoBoundaries {
  video_id: string;
  auto_boundaries: AutoBoundary[];
  corrected_boundaries: CorrectedBoundary[];
  has_corrections: boolean;
}

export const getVideos = () => api.get<VideoInfo[]>('/api/videos').then(r => r.data);
export const getVideoBoundaries = (videoId: string) =>
  api.get<VideoBoundaries>(`/api/videos/${videoId}/boundaries`).then(r => r.data);
export const saveVideoBoundaries = (videoId: string, boundaries: { start_s: number; end_s: number }[]) =>
  api.post(`/api/videos/${videoId}/boundaries`, { boundaries });
export const getVideoAudioUrl = (videoId: string) =>
  // Use streaming endpoint to bypass CORS issues with S3 presigned URLs
  `http://localhost:8000/api/videos/${videoId}/audio`;

// Progress and export API
export interface VideosProgress {
  total_videos: number;
  labeled_videos: number;
  remaining_videos: number;
  completion_percentage: number;
}

export interface NextUnlabeled {
  video_id?: string;
  found: boolean;
  message?: string;
}

export interface LabelExport {
  video_id: string;
  calls: { start: number; end: number }[];
}

export const getVideosProgress = () =>
  api.get<VideosProgress>('/api/videos/progress').then(r => r.data);
export const getNextUnlabeled = () =>
  api.get<NextUnlabeled>('/api/videos/next-unlabeled').then(r => r.data);
export const exportLabels = () =>
  api.get<LabelExport[]>('/api/export/labels').then(r => r.data);

export default api;
