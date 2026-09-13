# Park-Vision Mobile (Expo)

Minimal skeleton: take or pick a photo, send it to the Park-Vision backend's
`/detect-pixel` endpoint (no camera calibration needed), and see detected
vehicles (red boxes) and candidate empty parking spots (green boxes) drawn
directly on the photo.

## Setup

```bash
npm install
```

## Point it at your backend

Edit `src/api/parkVision.ts`:

```ts
export const API_BASE_URL = 'http://localhost:8000';
```

- **Physical phone (Expo Go)**: use your computer's LAN IP, e.g.
  `http://192.168.1.20:8000` (phone and computer must be on the same Wi-Fi).
- **Android emulator**: `http://10.0.2.2:8000` (special alias to the host machine).
- **iOS simulator**: `http://localhost:8000` works as-is.
- **CORS**: the FastAPI backend needs CORS enabled to accept requests from
  the app — this hasn't been added yet on the backend side (next step).

## Run

```bash
npx expo start
```

Scan the QR code with Expo Go (Android/iOS), or press `a` / `i` for an
emulator/simulator, or `w` to run in a browser.

## Project structure

```
App.tsx
src/
  api/parkVision.ts       API client + TypeScript types matching the backend schema
  components/
    DetectionOverlay.tsx   Draws red/green boxes on the photo using react-native-svg
  screens/
    HomeScreen.tsx          Take/pick photo, call the API, show results + counts
```

## What's next

- Enable CORS on the FastAPI backend.
- Add a settings screen to configure `API_BASE_URL` and `camera_id` from within the app.
- Add a history screen backed by `GET /history/{camera_id}`.
