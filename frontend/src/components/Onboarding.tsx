import { useEffect, useState } from "react";
import { Pause, Shuffle, X } from "lucide-react";
import { getTracksById, searchTracks, type Track } from "../api/recommendations";
import type { Origin } from "../state/store";
import { PlayIcon, TastyLogo } from "./icons";
import SearchBox from "./SearchBox";

/** Starting points: one well-played track per main genre. */
const STARTER_IDS = ["13668", "24425", "12672", "23014", "55718", "65792"];
const GENRES = ["Hip-Hop", "Electronic", "Rock", "Pop", "Folk", "Jazz", "Experimental", "Instrumental",
    "International", "Classical", "Soul-RnB", "Blues", "Country", "Old-Time / Historic"];

/** First-visit (and "start somewhere new") picker for the seed track. */
export default function Onboarding({ firstVisit, defaultTitle, nowPlayingId, playing, onChoose, onDismiss, onPreview }: {
    firstVisit: boolean;
    defaultTitle: string;
    nowPlayingId: string;
    playing: boolean;
    onChoose: (track: Track, origin: Origin) => void;
    onDismiss: () => void;
    onPreview: (track: Track) => void;
}) {
    const [starters, setStarters] = useState<Track[]>([]);
    const [genre, setGenre] = useState<string | null>(null);
    const [genreTracks, setGenreTracks] = useState<Track[]>([]);
    const [loadingGenre, setLoadingGenre] = useState(false);

    useEffect(() => {
        getTracksById(STARTER_IDS).then(setStarters).catch(() => setStarters([]));
    }, []);

    useEffect(() => {
        const onKey = (event: KeyboardEvent) => event.key === "Escape" && onDismiss();
        window.addEventListener("keydown", onKey);
        return () => window.removeEventListener("keydown", onKey);
    }, [onDismiss]);

    /** A genre's tracks, best fit first (search ranks a genre by how well tracks fit it). */
    async function tracksFor(name: string): Promise<Track[]> {
        const found = await searchTracks(name);
        return found.filter((track) => track.matchedBy === "genre");
    }

    function pickGenre(name: string) {
        if (genre === name) {
            setGenre(null);
            return;
        }
        setGenre(name);
        setLoadingGenre(true);
        tracksFor(name)
            .then((found) => setGenreTracks(found.slice(0, 5)))
            .catch(() => setGenreTracks([]))
            .finally(() => setLoadingGenre(false));
    }

    async function surprise() {
        const name = GENRES[Math.floor(Math.random() * GENRES.length)];
        const found = await tracksFor(name).catch(() => []);
        const pool = found.length ? found : starters;
        if (pool.length) onChoose(pool[Math.floor(Math.random() * pool.length)], { via: "surprise" });
    }

    const row = (track: Track, origin: Origin) => {
        const isPlaying = playing && nowPlayingId === track.id;
        return (
            <li key={track.id} className="onboarding-track">
                <span className="play-box small">
                    <button className="row-play" onClick={() => onPreview(track)} disabled={!track.audioUrl}
                        aria-label={`${isPlaying ? "Pause" : "Preview"} ${track.title}`} title={isPlaying ? "Pause" : "Have a listen"}>
                        {isPlaying ? <Pause size={13} fill="currentColor" /> : <PlayIcon size={13} />}
                    </button>
                </span>
                <img src={track.coverUrl} alt="" />
                <span className="onboarding-track-copy"><strong>{track.title}</strong><small>{track.artist} · {track.genre}</small></span>
                <button className="onboarding-start" onClick={() => onChoose(track, origin)}>Start here</button>
            </li>
        );
    };

    return (
        <div className="onboarding-overlay" onMouseDown={(event) => event.target === event.currentTarget && onDismiss()}>
            <div className="onboarding-card" role="dialog" aria-modal="true" aria-labelledby={firstVisit ? "onboarding-brand-name" : "onboarding-title"}>
                <header className="onboarding-hero">
                    <div className="onboarding-brand">
                        <TastyLogo size={30} tone="white" />
                        <span id="onboarding-brand-name">tastey</span>
                    </div>
                    <div className="onboarding-intro">
                        {!firstVisit && <h2 id="onboarding-title">Start somewhere new</h2>}
                        <p>Pick a seed track to start your discovery journey. We'll find music that sounds like it, from close to home to far out</p>
                    </div>
                    {!firstVisit && <button className="onboarding-close" onClick={onDismiss} aria-label="Close"><X size={18} /></button>}
                </header>

                <div className="onboarding-body">
                    <SearchBox onChoose={(track, query) => onChoose(track, { via: "search", query })} autoFocus placeholder="Search a song or artist, or describe a sound" className="search-wrap onboarding-search" />

                    <section>
                        <h3>Or start from a genre</h3>
                        <div className="genre-chips">
                            {GENRES.map((name) => (
                                <button key={name} className={genre === name ? "active" : undefined} onClick={() => pickGenre(name)} aria-pressed={genre === name}>{name}</button>
                            ))}
                        </div>
                        {genre && (
                            loadingGenre
                                ? <p className="onboarding-note">Finding {genre} tracks…</p>
                                : genreTracks.length
                                    ? <ul className="onboarding-list">{genreTracks.map((track) => row(track, { via: "genre", genre: genre! }))}</ul>
                                    : <p className="onboarding-note">No {genre} tracks found.</p>
                        )}
                    </section>

                    {!genre && starters.length > 0 && (
                        <section>
                            <h3>Popular starting points</h3>
                            <ul className="onboarding-list">{starters.map((track) => row(track, { via: "starter" }))}</ul>
                        </section>
                    )}
                </div>

                <footer className="onboarding-footer">
                    <button className="onboarding-surprise" onClick={surprise}><Shuffle size={14} /> Surprise me</button>
                    <button className="text-button" onClick={onDismiss}>{firstVisit ? `Skip, start with ${defaultTitle}` : "Cancel"}</button>
                </footer>
            </div>
        </div>
    );
}
