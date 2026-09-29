# Tastebridge Frontend

The React/Vite interface is intentionally separate from the Python recommender. It currently runs against clearly labeled preview data. Set `VITE_RECOMMENDER_API_URL` to use a compatible service; only `src/api/recommendations.ts` owns the HTTP integration.

## Run

```bash
npm install
npm run dev
```

## API contract

The adapter expects `GET /tracks?query=<text>` to return `{ "tracks": Track[] }`, and `POST /recommendations` to accept `{ "track_id": string, "adventure": number, "limit": number }` and return `{ "recommendations": Recommendation[] }`.

Track fields are `id`, `title`, `artist`, `genre`, `duration`, and `coverUrl`. Recommendation fields add `rank` and `distance`. The UI forwards the adventure value and renders the returned ordering; it does not implement model ranking.