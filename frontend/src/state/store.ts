import { configureStore, createSlice, type PayloadAction } from "@reduxjs/toolkit";

export type PlaybackState = {
    seedTrackId: string;
    currentTrackId: string;
    playRequestedId: string | null;
    playing: boolean;
    shuffleEnabled: boolean;
    shuffleOrder: string[];
    listeningPath: string[];
};

const initialState: PlaybackState = {
    seedTrackId: "demo-01",
    currentTrackId: "demo-01",
    playRequestedId: null,
    playing: false,
    shuffleEnabled: false,
    shuffleOrder: [],
    listeningPath: [],
};

const playbackSlice = createSlice({
    name: "playback",
    initialState,
    reducers: {
        selectSeedTrack(state, action: PayloadAction<string>) {
            state.seedTrackId = action.payload;
            if (state.currentTrackId === action.payload) return;
            state.currentTrackId = action.payload;
            state.playing = false;
            state.playRequestedId = null;
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
        recordPlayedTrack(state, action: PayloadAction<string>) {
            if (state.listeningPath.at(-1) !== action.payload) {
                state.listeningPath.push(action.payload);
            }
        },
        clearListeningPath(state) {
            state.listeningPath = [];
        },
    },
});

export const {
    clearListeningPath,
    clearPlayRequest,
    recordPlayedTrack,
    requestPlayback,
    selectPlaybackTrack,
    selectSeedTrack,
    setPlaying,
    setShuffle,
    setShuffleOrder,
} = playbackSlice.actions;

export const store = configureStore({
    reducer: { playback: playbackSlice.reducer },
});

export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;
