import { configureStore, createSlice, type PayloadAction } from "@reduxjs/toolkit";
import type { Track } from "../api/recommendations";

export type Tier = "Familiar" | "Bridge" | "Adventurous";

/** One seed you explored from. `parentKey` is the seed you were on when you
 *  sprouted this one (null: the start of a tree -- the first track, or a search
 *  jump). `tier` is where the slider was for that step. */
/** Why a track came up for a seed: the recommender's connecting factors
 *  ("genre", "era", ...) and its taste distance (cosine, 0..1) from that seed. */
export type Link = { factors: string[]; distance?: number };

/** How a journey (a tree on the Taste Map) began. */
export type Origin =
    | { via: "default" }                  // the app's starting track
    | { via: "search"; query: string }    // a search
    | { via: "genre"; genre: string }     // a genre in the picker
    | { via: "starter" }                  // one of the picker's starting points
    | { via: "surprise" };                // "Surprise me"

export type JourneyNode = {
    key: string;
    track: Track;
    parentKey: string | null;
    tier: Tier | null;
    link?: Link;  // to the parent seed; missing for roots and journeys saved before links existed
    origin?: Origin;  // roots only
};

/** A track in the Library. `seedKey` is the journey node that was the seed
 *  when it was saved -- the Taste Tree hangs it under that seed. `distance` is
 *  its distance from that seed, when it was saved from the recommendations. */
export type SavedTrack = {
    track: Track & { distance?: number };
    seedKey: string | null;
    link?: Link;  // to that seed, when it was saved from the recommendations
};

export type JourneyState = {
    nodes: JourneyNode[];
    currentKey: string | null;
};

const JOURNEY_STORAGE_KEY = "tasty.journey.v2";
const LIBRARY_STORAGE_KEY = "tasty.library.v1";

function loadStored<T>(key: string, valid: (value: unknown) => boolean, fallback: T): T {
    try {
        const stored = JSON.parse(localStorage.getItem(key) ?? "null");
        if (valid(stored)) return stored as T;
    } catch {
        // private window, blocked storage, or an old format -- start fresh
    }
    return fallback;
}

const loadJourney = () => loadStored<JourneyState>(JOURNEY_STORAGE_KEY,
    (value) => Array.isArray((value as JourneyState | null)?.nodes), { nodes: [], currentKey: null });
const loadLibrary = () => loadStored<SavedTrack[]>(LIBRARY_STORAGE_KEY, Array.isArray, []);

const newKey = (trackId: string) => `${Date.now().toString(36)}${Math.random().toString(36).slice(2, 6)}-${trackId}`;

export type PlaybackState = {
    seedTrackId: string;
    currentTrackId: string;
    playRequestedId: string | null;
    playing: boolean;
    shuffleEnabled: boolean;
    shuffleOrder: string[];
};

const initialState: PlaybackState = {
    seedTrackId: "demo-01",
    currentTrackId: "demo-01",
    playRequestedId: null,
    playing: false,
    shuffleEnabled: false,
    shuffleOrder: [],
};

function seedPlayback(state: PlaybackState, trackId: string) {
    state.seedTrackId = trackId;
    if (state.currentTrackId === trackId) return;
    state.currentTrackId = trackId;
    state.playing = false;
    state.playRequestedId = null;
}

const playbackSlice = createSlice({
    name: "playback",
    initialState,
    reducers: {
        selectSeedTrack(state, action: PayloadAction<string>) {
            seedPlayback(state, action.payload);
        },
        selectPlaybackTrack(state, action: PayloadAction<string>) {
            state.currentTrackId = action.payload;
            state.playing = false;
            state.playRequestedId = null;
        },
        requestPlayback(state, action: PayloadAction<string>) {
            state.currentTrackId = action.payload;
            state.playRequestedId = action.payload;
            state.playing = false;
        },
        clearPlayRequest(state) {
            state.playRequestedId = null;
        },
        setPlaying(state, action: PayloadAction<boolean>) {
            state.playing = action.payload;
        },
        setShuffle(state, action: PayloadAction<{ enabled: boolean; order: string[] }>) {
            state.shuffleEnabled = action.payload.enabled;
            state.shuffleOrder = action.payload.enabled ? action.payload.order : [];
        },
        setShuffleOrder(state, action: PayloadAction<string[]>) {
            state.shuffleOrder = action.payload;
        },
    },
});

const journeySlice = createSlice({
    name: "journey",
    initialState: loadJourney,
    reducers: {
        /** `from: "root"` starts a new tree; `"branch"` grows from the current seed
         *  (or steps onto that branch if it's already there). */
        plantSeed(state, action: PayloadAction<{ track: Track; from: "root" | "branch"; tier: Tier | null; link?: Link; origin?: Origin }>) {
            const { track, from, tier, link, origin } = action.payload;
            const parentKey = from === "branch" ? state.currentKey : null;
            const existing = state.nodes.find((node) => node.parentKey === parentKey && node.track.id === track.id
                && (parentKey !== null || node.key === state.currentKey));
            if (existing) {
                state.currentKey = existing.key;
                return;
            }
            const key = newKey(track.id);
            state.nodes.push(parentKey === null
                ? { key, track, parentKey, tier: null, origin }
                : { key, track, parentKey, tier, link });
            state.currentKey = key;
        },
        revisitSeed(state, action: PayloadAction<string>) {
            if (state.nodes.some((node) => node.key === action.payload)) state.currentKey = action.payload;
        },
        /** A fresh journey starting at `track` (the first-visit onboarding pick). */
        startJourney(state, action: PayloadAction<{ track: Track; origin?: Origin }>) {
            const key = newKey(action.payload.track.id);
            state.nodes = [{ key, track: action.payload.track, parentKey: null, tier: null, origin: action.payload.origin }];
            state.currentKey = key;
        },
        clearJourney(state) {
            const current = state.nodes.find((node) => node.key === state.currentKey);
            const key = current ? newKey(current.track.id) : null;
            state.nodes = current && key ? [{ ...current, key, parentKey: null, tier: null }] : [];
            state.currentKey = key;
        },
    },
});

const librarySlice = createSlice({
    name: "library",
    initialState: loadLibrary,
    reducers: {
        /** Save `track` (newest first), or unsave it if it's already in the Library. */
        toggleSaved(state, action: PayloadAction<SavedTrack>) {
            const index = state.findIndex((item) => item.track.id === action.payload.track.id);
            if (index >= 0) state.splice(index, 1);
            else state.unshift(action.payload);
        },
        /** Manual order: move the track with id `trackId` to position `to`. */
        moveSaved(state, action: PayloadAction<{ trackId: string; to: number }>) {
            const from = state.findIndex((item) => item.track.id === action.payload.trackId);
            if (from < 0) return;
            const [item] = state.splice(from, 1);
            state.splice(Math.max(0, Math.min(action.payload.to, state.length)), 0, item);
        },
    },
});

export const { clearJourney, plantSeed, revisitSeed, startJourney } = journeySlice.actions;
export const { moveSaved, toggleSaved } = librarySlice.actions;

export const {
    clearPlayRequest,
    requestPlayback,
    selectPlaybackTrack,
    selectSeedTrack,
    setPlaying,
    setShuffle,
    setShuffleOrder,
} = playbackSlice.actions;

export const store = configureStore({
    reducer: { playback: playbackSlice.reducer, journey: journeySlice.reducer, library: librarySlice.reducer },
});

store.subscribe(() => {
    try {
        localStorage.setItem(JOURNEY_STORAGE_KEY, JSON.stringify(store.getState().journey));
        localStorage.setItem(LIBRARY_STORAGE_KEY, JSON.stringify(store.getState().library));
    } catch {
        // storage full or blocked: the journey and Library just won't survive a reload
    }
});

export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;
