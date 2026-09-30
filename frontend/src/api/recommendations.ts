export type Track = {
    id: string;
    title: string;
    artist: string;
    genre: string;
    duration: string;
    coverUrl: string;
    previewUrl?: string;
};

export type Recommendation = Track & {
    rank: number;
    distance: number;
};

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

const previewTracks: Track[] = [
    { id: "demo-01", title: "Glass City", artist: "Mira Vale", genre: "Electronic", duration: "3:42", coverUrl: covers.signal },
    { id: "demo-02", title: "Low Tide Radio", artist: "North Arcade", genre: "Indie", duration: "4:06", coverUrl: covers.tide },
    { id: "demo-03", title: "Soft Geometry", artist: "Aster Field", genre: "Ambient", duration: "3:18", coverUrl: covers.geometry },
    { id: "demo-04", title: "Copper Lines", artist: "Juniper Club", genre: "Jazz", duration: "5:11", coverUrl: covers.copper },
    { id: "demo-05", title: "After the Signal", artist: "Common Weather", genre: "Alternative", duration: "3:56", coverUrl: covers.blue },
    { id: "demo-06", title: "Blue Hour Loop", artist: "Sora Bloom", genre: "Electronic", duration: "4:21", coverUrl: covers.night },
    { id: "demo-07", title: "A Map of Stars", artist: "June Static", genre: "Dream pop", duration: "3:37", coverUrl: covers.field },
    { id: "demo-08", title: "Warm Current", artist: "Sunday Syntax", genre: "Soul", duration: "4:28", coverUrl: covers.warm },
];

const apiBase = import.meta.env.VITE_RECOMMENDER_API_URL?.replace(/\/$/, "");

export const isPreview = !apiBase;
export const previewCatalog = previewTracks;

export async function searchTracks(query: string): Promise<Track[]> {
    if (!apiBase) {
        const normalized = query.trim().toLowerCase();
        return previewTracks.filter((track) =>
            `${track.title} ${track.artist} ${track.genre}`.toLowerCase().includes(normalized),
        );
    }

    const response = await fetch(`${apiBase}/tracks?query=${encodeURIComponent(query)}`);
    if (!response.ok) throw new Error(`Track search failed (${response.status})`);
    const payload = (await response.json()) as { tracks: Track[] };
    return payload.tracks;
}

export async function getRecommendations(
    trackId: string,
    adventure: number,
    limit: number,
): Promise<Recommendation[]> {
    if (!apiBase) {
        const candidates = previewTracks.filter((track) => track.id !== trackId);
        const maxOffset = Math.max(0, candidates.length - limit);
        const offset = Math.round(adventure * maxOffset);
        return candidates.slice(offset, offset + limit).map((track, index) => ({
            ...track,
            rank: offset + index + 1,
            distance: Number((0.12 + (offset + index) * 0.075).toFixed(3)),
        }));
    }

    const response = await fetch(`${apiBase}/recommendations`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ track_id: trackId, adventure, limit }),
    });
    if (!response.ok) throw new Error(`Recommendation request failed (${response.status})`);
    const payload = (await response.json()) as { recommendations: Recommendation[] };
    return payload.recommendations;
}