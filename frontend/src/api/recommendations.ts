export type Track = {
    id: string;
    title: string;
    artist: string;
    genre: string;
    duration: string;
    coverUrl: string;
    audioUrl?: string;
    /** The artist's location as FMA lists it, e.g. "Bristol, UK" (about 2/3 of tracks have one). */
    location?: string | null;
    /** When the track was added to FMA, YYYY-MM-DD. */
    dateCreated?: string | null;
    /** Tempo estimated from the audio (can read double or half on some tracks). */
    bpm?: number | null;
    /** Search results only: why this track matched -- its name, its genre, or how it sounds. */
    matchedBy?: "name" | "genre" | "sound";
};

export type Recommendation = Track & {
    rank: number;
    distance: number;
    /** Why it connects to the seed: "genre", "genre_lineage", "related_genre", "era", "geography". */
    connectingFactors?: string[];
};

export type MapPoint = { id: string; x: number; y: number };

const covers = {
    signal: "https://images.unsplash.com/photo-1470225620780-dba8ba36b745?auto=format&fit=crop&w=900&q=85",
    tide: "https://images.unsplash.com/photo-1511379938547-c1f69419868d?auto=format&fit=crop&w=600&q=80",
    geometry: "https://images.unsplash.com/photo-1506157786151-b8491531f063?auto=format&fit=crop&w=600&q=80",
    copper: "https://images.unsplash.com/photo-1493225457124-a3eb161ffa5f?auto=format&fit=crop&w=600&q=80",
    blue: "https://images.unsplash.com/photo-1516280440614-37939bbacd81?auto=format&fit=crop&w=600&q=80",
    night: "https://images.unsplash.com/photo-1524368535928-5b5e00ddc76b?auto=format&fit=crop&w=600&q=80",
    field: "https://images.unsplash.com/photo-1519681393784-d120267933ba?auto=format&fit=crop&w=600&q=80",
    warm: "https://images.unsplash.com/photo-1508700115892-45ecd05ae2ad?auto=format&fit=crop&w=600&q=80",
};

const demoAudioUrl = "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3";

const previewTracks: Track[] = [
    { id: "demo-01", title: "Glass City", artist: "Mira Vale", genre: "Electronic", duration: "3:42", coverUrl: covers.signal, audioUrl: demoAudioUrl },
    { id: "demo-02", title: "Low Tide Radio", artist: "North Arcade", genre: "Indie", duration: "4:06", coverUrl: covers.tide, audioUrl: demoAudioUrl },
    { id: "demo-03", title: "Soft Geometry", artist: "Aster Field", genre: "Ambient", duration: "3:18", coverUrl: covers.geometry, audioUrl: demoAudioUrl },
    { id: "demo-04", title: "Copper Lines", artist: "Juniper Club", genre: "Jazz", duration: "5:11", coverUrl: covers.copper, audioUrl: demoAudioUrl },
    { id: "demo-05", title: "After the Signal", artist: "Common Weather", genre: "Alternative", duration: "3:56", coverUrl: covers.blue, audioUrl: demoAudioUrl },
    { id: "demo-06", title: "Blue Hour Loop", artist: "Sora Bloom", genre: "Electronic", duration: "4:21", coverUrl: covers.night, audioUrl: demoAudioUrl },
    { id: "demo-07", title: "A Map of Stars", artist: "June Static", genre: "Dream pop", duration: "3:37", coverUrl: covers.field, audioUrl: demoAudioUrl },
    { id: "demo-08", title: "Warm Current", artist: "Sunday Syntax", genre: "Soul", duration: "4:28", coverUrl: covers.warm, audioUrl: demoAudioUrl },
];

const configuredApiBase = import.meta.env.VITE_RECOMMENDER_API_URL?.replace(/\/$/, "");
const localFallbackApiBase =
    typeof window !== "undefined" && (window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1")
        ? "http://localhost:8000"
        : undefined;

// "" (built with VITE_RECOMMENDER_API_URL=/) means the API is this same site -- the hosted app
const apiBase = configuredApiBase ?? localFallbackApiBase;

export const isPreview = apiBase === undefined;

/** fetch, but failures come back as messages a listener can read. */
async function apiFetch(path: string, what: string, init?: RequestInit): Promise<Response> {
    let response: Response;
    try {
        response = await fetch(`${apiBase}${path}`, init);
    } catch {
        // fetch only throws when no response arrived at all: server down, wrong port, or CORS
        throw new Error(`Can't reach the Tastey server at ${apiBase || window.location.origin}. Is the API running?`);
    }
    if (!response.ok) {
        const detail = await response
            .json()
            .then((body: { detail?: unknown }) => (typeof body.detail === "string" ? body.detail : ""))
            .catch(() => "");
        throw new Error(detail || `${what} failed (${response.status}).`);
    }
    return response;
}
export const previewCatalog = previewTracks;

export async function searchTracks(query: string): Promise<Track[]> {
    if (isPreview) {
        const normalized = query.trim().toLowerCase();
        return previewTracks.filter((track) =>
            `${track.title} ${track.artist} ${track.genre}`.toLowerCase().includes(normalized),
        );
    }

    const response = await apiFetch(`/tracks?query=${encodeURIComponent(query)}`, "Track search");
    const payload = (await response.json()) as { tracks: Track[] };
    return payload.tracks;
}

/** Tracks by id, in the order given; ids the catalogue doesn't have are left out. */
export async function getTracksById(ids: string[]): Promise<Track[]> {
    if (isPreview) return previewTracks.filter((track) => ids.includes(track.id));
    const response = await apiFetch(`/tracks/by-id?ids=${ids.map(encodeURIComponent).join(",")}`, "Track lookup");
    const payload = (await response.json()) as { tracks: Track[] };
    return payload.tracks;
}

export async function getRecommendations(
    trackId: string,
    adventure: number,
): Promise<Recommendation[]> {
    if (isPreview) {
        // offline preview only -- with a backend, the server decides the count
        const limit = 15;
        const candidates = previewTracks.filter((track) => track.id !== trackId);
        const maxOffset = Math.max(0, candidates.length - limit);
        const offset = Math.round(adventure * maxOffset);
        return candidates.slice(offset, offset + limit).map((track, index) => ({
            ...track,
            rank: offset + index + 1,
            distance: Number((0.12 + (offset + index) * 0.075).toFixed(3)),
        }));
    }

    const response = await apiFetch("/recommendations", "Recommendation request", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ track_id: trackId, adventure }),
    });
    const payload = (await response.json()) as { recommendations: Recommendation[] };
    return payload.recommendations;
}

/** 2D positions for these tracks, laid out by their own sound (a projection
 *  fitted to just this set, so the spread shows how they differ). */
export async function getJourneyMap(ids: string[]): Promise<MapPoint[]> {
    if (isPreview) return ids.map((id, index) => ({ id, x: 0.2 + 0.6 * ((index * 0.37) % 1), y: 0.2 + 0.6 * ((index * 0.61) % 1) }));
    const response = await apiFetch(`/journey-map?ids=${ids.map(encodeURIComponent).join(",")}`, "Taste Map layout");
    const payload = (await response.json()) as { points: MapPoint[] };
    return payload.points;
}
