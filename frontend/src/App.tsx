import { useEffect, useRef, useState } from "react";
import {
    AudioLines,
    Check,
    ChevronDown,
    Compass,
    Disc3,
    Heart,
    Library,
    ListMusic,
    Pause,
    Play,
    Plus,
    Search,
    SlidersHorizontal,
    Volume2,
} from "lucide-react";
import {
    getRecommendations,
    isPreview,
    previewCatalog,
    searchTracks,
    type Recommendation,
    type Track,
} from "./api/recommendations";

type Page = "discover" | "saved";

function App() {
    const [page, setPage] = useState<Page>("discover");
    const [search, setSearch] = useState("");
    const [searchOpen, setSearchOpen] = useState(false);
    const [tracks, setTracks] = useState<Track[]>(previewCatalog);
    const [selectedId, setSelectedId] = useState(previewCatalog[0].id);
    const [adventure, setAdventure] = useState(0.35);
    const [limit, setLimit] = useState(5);
    const [recommendations, setRecommendations] = useState<Recommendation[]>([]);
    const [saved, setSaved] = useState<string[]>(["demo-06"]);
    const [playing, setPlaying] = useState(false);
    const [error, setError] = useState("");
    const audioRef = useRef<HTMLAudioElement>(null);

    const selectedTrack = tracks.find((track) => track.id === selectedId)
        ?? recommendations.find((track) => track.id === selectedId)
        ?? previewCatalog.find((track) => track.id === selectedId)
        ?? previewCatalog[0];

    useEffect(() => {
        let active = true;
        searchTracks(search)
            .then((found) => {
                if (active) setTracks(found);
            })
            .catch((reason: unknown) => {
                if (active) setError(reason instanceof Error ? reason.message : "Could not load tracks.");
            });
        return () => {
            active = false;
        };
    }, [search]);

    useEffect(() => {
        let active = true;
        setError("");
        getRecommendations(selectedId, adventure, limit)
            .then((items) => {
                if (active) setRecommendations(items);
            })
            .catch((reason: unknown) => {
                if (active) setError(reason instanceof Error ? reason.message : "Could not get recommendations.");
            });
        return () => {
            active = false;
        };
    }, [selectedId, adventure, limit]);

    const visibleRecommendations = page === "saved"
        ? recommendations.filter((track) => saved.includes(track.id))
        : recommendations;

    const autoplayNext = useRef(false);
    useEffect(() => {
        const audio = audioRef.current;
        audio?.pause();
        if (audio && autoplayNext.current) {
            audio.play().catch(() => setPlaying(false));
        }
        autoplayNext.current = false;
    }, [selectedId]);

    function chooseTrack(track: Track) {
        setSelectedId(track.id);
        setSearch("");
        setSearchOpen(false);
        setPage("discover");
    }

    function togglePlay() {
        const audio = audioRef.current;
        if (!audio || !selectedTrack.previewUrl) return;
        if (audio.paused) {
            audio.play().catch(() => setPlaying(false));
        } else {
            audio.pause();
        }
    }

    function playTrack(trackId: string) {
        autoplayNext.current = true;
        setSelectedId(trackId);
    }

    function toggleSaved(trackId: string) {
        setSaved((current) => current.includes(trackId)
            ? current.filter((id) => id !== trackId)
            : [...current, trackId]);
    }

    return (
        <div className="app-shell">
            <aside className="sidebar">
                <a className="wordmark" href="#discover" onClick={() => setPage("discover")}>
                    <span className="wordmark-icon"><AudioLines size={19} strokeWidth={2.5} /></span>
                    <span>tastebridge</span>
                </a>

                <div className="side-label">Your space</div>
                <nav className="primary-nav" aria-label="Main navigation">
                    <button className={page === "discover" ? "nav-item active" : "nav-item"} onClick={() => setPage("discover")}>
                        <Compass size={18} /><span>Discover</span>
                    </button>
                    <button className={page === "saved" ? "nav-item active" : "nav-item"} onClick={() => setPage("saved")}>
                        <Library size={18} /><span>Your library</span>
                    </button>
                </nav>

                <div className="side-label playlist-label">Collections <button className="icon-button small" aria-label="Create collection" title="Create collection"><Plus size={16} /></button></div>
                <button className="collection-row" onClick={() => setPage("saved")}>
                    <span className="collection-art saved-art"><Heart size={15} fill="currentColor" /></span>
                    <span><strong>Saved tracks</strong><small>{saved.length} tracks</small></span>
                </button>
                <div className="sidebar-bottom">
                    <div className="mini-now-playing">
                        <div className="mini-cover"><img src={selectedTrack.coverUrl} alt="" /></div>
                        <div className="mini-track"><strong>{selectedTrack.title}</strong><span>{selectedTrack.artist}</span></div>
                        <button className={saved.includes(selectedTrack.id) ? "icon-button saved" : "icon-button"} onClick={() => toggleSaved(selectedTrack.id)} aria-label="Save starting track" title="Save track"><Heart size={16} fill={saved.includes(selectedTrack.id) ? "currentColor" : "none"} /></button>
                    </div>
                    <div className="sound-status"><span className={isPreview ? "status-dot preview" : "status-dot live"}></span>{isPreview ? "Preview catalogue" : "Recommender connected"}</div>
                </div>
            </aside>

            <main className="main-content">
                <header className="topbar">
                    <div className="history-controls" aria-hidden="true"><span>‹</span><span>›</span></div>
                    <div className="search-wrap">
                        <Search size={17} />
                        <input
                            aria-label="Search tracks"
                            value={search}
                            onChange={(event) => setSearch(event.target.value)}
                            onFocus={() => setSearchOpen(true)}
                            onBlur={() => window.setTimeout(() => setSearchOpen(false), 140)}
                            placeholder="Search a song to start from"
                        />
                        {searchOpen && search.trim() && tracks.length > 0 && (
                            <div className="search-results" role="listbox">
                                {tracks.slice(0, 5).map((track) => (
                                    <button key={track.id} onMouseDown={() => chooseTrack(track)} role="option" aria-selected={track.id === selectedId}>
                                        <img src={track.coverUrl} alt="" /><span><strong>{track.title}</strong><small>{track.artist} · {track.genre}</small></span>
                                    </button>
                                ))}
                            </div>
                        )}
                    </div>
                    <button className="profile-button"><span className="profile-avatar">T</span><span>Listener</span><ChevronDown size={14} /></button>
                </header>

                {isPreview && <div className="preview-banner"><span className="status-dot preview"></span><strong>Preview mode</strong><span>Sample tracks and responses are shown; they are not neural-network results.</span></div>}

                <div className="content-wrap">
                    <section className="page-intro">
                        <div>
                            <div className="eyebrow"><span className="eyebrow-mark"></span>Sound-led discovery</div>
                            <h1>{page === "saved" ? "Your saved tracks" : "Find a new way in."}</h1>
                            <p>{page === "saved" ? "The tracks you want to come back to." : "Start with a track you love. Let the sound take you somewhere new."}</p>
                        </div>
                        <div className="intro-index"><span>LISTENING SESSION</span><strong>01 <i>/</i> 08</strong></div>
                    </section>

                    {page === "discover" && (
                        <>
                            <section className="discovery-grid" aria-label="Discovery controls">
                                <div className="seed-feature">
                                    <img className="seed-cover" src={selectedTrack.coverUrl} alt={`Cover art for ${selectedTrack.title}`} />
                                    <div className="seed-shade"></div>
                                    <div className="seed-topline"><span className="overline">Starting track</span><span className="seed-id">{selectedTrack.id}</span></div>
                                    <div className="seed-details">
                                        <div><div className="seed-genre">{selectedTrack.genre} <span>·</span> {selectedTrack.duration}</div><h2>{selectedTrack.title}</h2><p>{selectedTrack.artist}</p></div>
                                        <button className="play-button" onClick={togglePlay} disabled={!selectedTrack.previewUrl} aria-label={playing ? "Pause" : "Play"} title={selectedTrack.previewUrl ? (playing ? "Pause" : "Play") : "No audio available"}>{playing ? <Pause size={20} fill="currentColor" /> : <Play size={20} fill="currentColor" />}</button>
                                    </div>
                                    <div className="cover-caption"><Disc3 size={14} /> SEED TRACK <span>01</span></div>
                                </div>

                                <div className="adventure-panel">
                                    <div className="panel-heading"><div><div className="eyebrow">Your listening radius</div><h2>Set the mood.</h2></div><SlidersHorizontal size={19} /></div>
                                    <p className="panel-copy">Stay close to the sound you know, or let the recommendations stretch further.</p>
                                    <div className="adventure-value"><span>ADVENTURE</span><strong>{Math.round(adventure * 100)}<small>%</small></strong></div>
                                    <input className="adventure-slider" type="range" min="0" max="100" step="1" value={Math.round(adventure * 100)} onChange={(event) => setAdventure(Number(event.target.value) / 100)} aria-label="Adventure level" style={{ "--slider-progress": `${adventure * 100}%` } as React.CSSProperties} />
                                    <div className="range-labels"><span>Familiar</span><span>Adventurous</span></div>
                                    <div className="panel-divider"></div>
                                    <div className="result-count"><div><span className="overline">Your next listens</span><strong>{limit} tracks</strong></div><select value={limit} onChange={(event) => setLimit(Number(event.target.value))} aria-label="Number of recommendations"><option value={5}>5 tracks</option><option value={7}>7 tracks</option><option value={10}>10 tracks</option></select></div>
                                </div>
                            </section>

                            <section className="recommendations-section">
                                <div className="section-heading"><div><div className="eyebrow"><span className="eyebrow-mark"></span>Made from your starting point</div><h2>Follow this thread</h2></div><button className="text-button" onClick={() => setAdventure(Math.min(1, Number((adventure + 0.2).toFixed(2))))}>Go further <span>↗</span></button></div>
                                <div className="list-heading"><span>#</span><span>Track</span><span>Style</span><span>Distance</span><span></span></div>
                                {error && <div className="error-state" role="alert">{error}</div>}
                                {!error && recommendations.map((track) => <RecommendationRow key={`${track.id}-${track.rank}`} track={track} saved={saved.includes(track.id)} onToggleSaved={() => toggleSaved(track.id)} onPlay={() => playTrack(track.id)} />)}
                                {!error && recommendations.length === 0 && <div className="empty-state">No recommendations for this selection yet.</div>}
                                <div className="list-footnote"><span>Cosine distance</span> is a model measure, not a score of how much you will like a track.</div>
                            </section>
                        </>
                    )}

                    {page === "saved" && (
                        <section className="recommendations-section saved-section">
                            <div className="section-heading"><div><div className="eyebrow"><span className="eyebrow-mark"></span>Your collection</div><h2>Kept for later</h2></div><span className="saved-count">{saved.length} SAVED</span></div>
                            <div className="list-heading"><span>#</span><span>Track</span><span>Style</span><span>Distance</span><span></span></div>
                            {visibleRecommendations.map((track) => <RecommendationRow key={`${track.id}-${track.rank}`} track={track} saved onToggleSaved={() => toggleSaved(track.id)} onPlay={() => playTrack(track.id)} />)}
                            {visibleRecommendations.length === 0 && <div className="empty-state">Save a recommendation with the heart button and it will appear here.</div>}
                        </section>
                    )}
                </div>

                <footer className="page-footer"><span>TASTEBRIDGE / MUSIC DISCOVERY</span><span>{isPreview ? "DEMO DATA" : "CONNECTED TO RECOMMENDER"}</span></footer>
            </main>

            <div className="player-bar">
                <audio
                    ref={audioRef}
                    src={selectedTrack.previewUrl}
                    onPlay={() => setPlaying(true)}
                    onPause={() => setPlaying(false)}
                    onEnded={() => setPlaying(false)}
                />
                <div className="player-track"><img src={selectedTrack.coverUrl} alt="" /><div><strong>{selectedTrack.title}</strong><span>{selectedTrack.artist}</span></div><button className="icon-button" onClick={() => toggleSaved(selectedTrack.id)} aria-label="Save current track"><Heart size={16} fill={saved.includes(selectedTrack.id) ? "currentColor" : "none"} /></button></div>
                <div className="player-center"><button className="player-play" onClick={togglePlay} disabled={!selectedTrack.previewUrl} aria-label={playing ? "Pause" : "Play"}>{playing ? <Pause size={15} fill="currentColor" /> : <Play size={15} fill="currentColor" />}</button><div className="fake-progress"><span></span></div>{isPreview && <small>PREVIEW</small>}</div>
                <div className="player-volume"><ListMusic size={16} /><Volume2 size={17} /><div className="volume-line"><span></span></div></div>
            </div>
        </div>
    );
}

function RecommendationRow({ track, saved, onToggleSaved, onPlay }: {
    track: Recommendation;
    saved: boolean;
    onToggleSaved: () => void;
    onPlay: () => void;
}) {
    return (
        <div className="track-row">
            <span className="track-rank">{String(track.rank).padStart(2, "0")}</span>
            <div className="track-cell"><img className="track-cover" src={track.coverUrl} alt="" /><div className="track-copy"><strong>{track.title}</strong><span>{track.artist}</span></div><button className="row-play" onClick={onPlay} disabled={!track.previewUrl} aria-label={`Play ${track.title}`} title={`Play ${track.title}`}><Play size={14} fill="currentColor" /></button></div>
            <span className="track-genre">{track.genre}</span>
            <span className="track-distance">{track.distance.toFixed(3)}</span>
            <button className={saved ? "icon-button heart-button is-saved" : "icon-button heart-button"} onClick={onToggleSaved} aria-label={saved ? `Remove ${track.title} from saved` : `Save ${track.title}`} title={saved ? "Saved" : "Save track"}>{saved ? <Check size={16} /> : <Heart size={16} />}</button>
        </div>
    );
}

export default App;