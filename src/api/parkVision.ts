import { Platform } from 'react-native';

// Point this at your FastAPI backend. On a physical phone/simulator, "localhost"
// refers to the device itself, not your computer — use your computer's LAN IP
// (e.g. "http://192.168.1.20:8000") when testing on a real device or the
// Android emulator's special alias "http://10.0.2.2:8000" for Android Studio's
// emulator specifically.
export const API_BASE_URL = 'http://localhost:8000';

export interface PixelVehicle {
  id: number;
  bbox: [number, number, number, number];
  confidence: number;
}

export interface PixelSpot {
  id: number;
  status: 'occupied' | 'empty';
  bbox: [number, number, number, number];
}

export interface PixelDetectResponse {
  camera_id: string | null;
  frame_width: number;
  frame_height: number;
  recheck_enabled: boolean;
  vehicles: PixelVehicle[];
  spots: PixelSpot[];
  processing_time_ms: number;
}

export async function detectPixel(
  imageUri: string,
  options?: { cameraId?: string; enableRecheck?: boolean; conf?: number }
): Promise<PixelDetectResponse> {
  const formData = new FormData();

  if (Platform.OS === 'web') {
    // On web, imageUri is a blob:/data: URL from the browser file picker.
    // fetch's FormData needs a real Blob/File here -- the RN {uri,name,type}
    // shape below is silently ignored by browsers, which is why nothing
    // happened before.
    const fileResponse = await fetch(imageUri);
    const blob = await fileResponse.blob();
    formData.append('image', blob, 'photo.jpg');
  } else {
    // React Native's fetch accepts this object shape for file uploads.
    formData.append('image', {
      uri: imageUri,
      name: 'photo.jpg',
      type: 'image/jpeg',
    } as unknown as Blob);
  }

  if (options?.cameraId) {
    formData.append('camera_id', options.cameraId);
  }
  formData.append('enable_recheck', String(options?.enableRecheck ?? true));
  formData.append('conf', String(options?.conf ?? 0.35));

  const response = await fetch(`${API_BASE_URL}/detect-pixel`, {
    method: 'POST',
    body: formData,
    // No Content-Type header here on purpose: fetch sets
    // "multipart/form-data; boundary=..." automatically when given a
    // FormData body. Setting it manually (as before) omits the boundary
    // and the backend can't parse the request at all.
  });

  if (!response.ok) {
    const detail = await response.text();
    throw new Error(`Detection failed (${response.status}): ${detail}`);
  }

  return response.json();
}
