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
    Map as MapIcon,
    Pause,
    Play,
    Plus,
    Search,
    SlidersHorizontal,
    Shuffle,
    Volume2,
} from "lucide-react";
import { useDispatch, useSelector } from "react-redux";
import {
    getTasteMap,
    getRecommendations,
    isPreview,
    previewCatalog,
    searchTracks,
    type Recommendation,
    type TasteMapPoint,
    type Track,
} from "./api/recommendations";
import TasteMapCanvas from "./components/TasteMapCanvas";
import {
    clearListeningPath,
    clearPlayRequest,
    recordPlayedTrack,
    requestPlayback,
    selectSeedTrack,
    setPlaying,
    setShuffle,
    setShuffleOrder,
    type AppDispatch,
    type RootState,
} from "./state/store";

type Page = "discover" | "saved" | "map";

function App() {
    const dispatch = useDispatch<AppDispatch>();
    const { seedTrackId, currentTrackId, playRequestedId, playing, shuffleEnabled, shuffleOrder, listeningPath } =
        useSelector((state: RootState) => state.playback);
    const [page, setPage] = useState<Page>("discover");
    const [search, setSearch] = useState("");
    const [searchOpen, setSearchOpen] = useState(false);
    const [tracks, setTracks] = useState<Track[]>(previewCatalog);
    const [searchResults, setSearchResults] = useState<Track[]>([]);
    const [adventure, setAdventure] = useState(0.35);
    const [limit, setLimit] = useState(5);
    const [recommendations, setRecommendations] = useState<Recommendation[]>([]);
    const [recommendationsLoading, setRecommendationsLoading] = useState(false);
    const [tasteMapPoints, setTasteMapPoints] = useState<TasteMapPoint[]>([]);
    const [tasteMapLoading, setTasteMapLoading] = useState(false);
    const [tasteMapError, setTasteMapError] = useState("");
    const [mapSelectedId, setMapSelectedId] = useState<string | null>(null);
    const [saved, setSaved] = useState<string[]>(["demo-06"]);
    const [error, setError] = useState("");
    const [catalogReady, setCatalogReady] = useState(isPreview);
    const [audioDuration, setAudioDuration] = useState("");
    const audioRef = useRef<HTMLAudioElement>(null);

    const selectedTrack = tracks.find((track) => track.id === currentTrackId)
        ?? previewCatalog.find((track) => track.id === currentTrackId)
        ?? previewCatalog[0];
    const seedTrack = tracks.find((track) => track.id === seedTrackId)
        ?? previewCatalog.find((track) => track.id === seedTrackId)
        ?? previewCatalog[0];
    const mapSelectedTrack = tasteMapPoints.find((track) => track.id === mapSelectedId)
        ?? tasteMapPoints.find((track) => track.id === currentTrackId)
        ?? null;

    const togglePlayback = async () => {
        const audio = audioRef.current;
        if (!audio || !selectedTrack.audioUrl) return;

        if (audio.paused) {
            try {
                await audio.play();
            } catch {
                dispatch(setPlaying(false));
            }
        } else {
            audio.pause();
        }
    };

    useEffect(() => {
        audioRef.current?.pause();
        setAudioDuration("");
        dispatch(setPlaying(false));
    }, [currentTrackId, dispatch]);

    useEffect(() => {
        if (playRequestedId !== currentTrackId) return;
        const audio = audioRef.current;
        if (!audio) return;
        audio.play().catch(() => dispatch(setPlaying(false)));
        dispatch(clearPlayRequest());
    }, [playRequestedId, currentTrackId, dispatch]);

    useEffect(() => {
        if (isPreview) return;

        let active = true;
        searchTracks("")
            .then((found) => {
                if (!active) return;
                setTracks(found);
                if (found.length > 0) dispatch(selectSeedTrack(found[0].id));
                setCatalogReady(true);
            })
            .catch((reason: unknown) => {
                if (active) setError(reason instanceof Error ? reason.message : "Could not load tracks.");
            });
        return () => {
            active = false;
        };
    }, [dispatch]);

    useEffect(() => {
        if (page !== "map" || tasteMapPoints.length > 0) return;
        let active = true;
        setTasteMapLoading(true);
        getTasteMap()
            .then((points) => {
                if (!active) return;
                setTasteMapPoints(points);
                setMapSelectedId(currentTrackId);
                setTasteMapError("");
                setTasteMapLoading(false);
            })
            .catch((reason: unknown) => {
                if (!active) return;
                setTasteMapError(reason instanceof Error ? reason.message : "Could not load the Taste Map.");
                setTasteMapLoading(false);
            });
        return () => {
            active = false;
        };
    }, [page, tasteMapPoints.length, currentTrackId]);

    useEffect(() => {
        if (!search.trim()) {
            setSearchResults([]);
            return;
        }

        let active = true;
        searchTracks(search)
            .then((found) => {
                if (active) setSearchResults(found);
            })
            .catch((reason: unknown) => {
                if (active) setError(reason instanceof Error ? reason.message : "Could not search tracks.");
            });
        return () => {
            active = false;
        };
    }, [search]);

    useEffect(() => {
        let active = true;
        if (!catalogReady) return () => { active = false; };
        setError("");
        setRecommendationsLoading(true);
        getRecommendations(seedTrackId, adventure, limit)
            .then((items) => {
                if (!active) return;
                setRecommendations(items);
                setRecommendationsLoading(false);
            })
            .catch((reason: unknown) => {
                if (!active) return;
                setRecommendationsLoading(false);
                setError(reason instanceof Error ? reason.message : "Could not get recommendations.");
            });
        return () => {
            active = false;
        };
    }, [catalogReady, seedTrackId, adventure, limit, dispatch]);

    useEffect(() => {
        if (shuffleEnabled && recommendations.length > 1) {
            dispatch(setShuffleOrder(shuffleTrackIds(recommendations)));
        }
    }, [shuffleEnabled, recommendations, dispatch]);

    const filteredRecommendations = page === "saved"
        ? recommendations.filter((track) => saved.includes(track.id))
        : recommendations;
    const visibleRecommendations = shuffleEnabled
        ? orderTracks(filteredRecommendations, shuffleOrder)
        : filteredRecommendations;

    function chooseTrack(track: Track) {
        setTracks((current) => current.some((item) => item.id === track.id) ? current : [...current, track]);
        dispatch(selectSeedTrack(track.id));
        setSearch("");
        setSearchOpen(false);
        setPage("discover");
    }

    function playRecommendedTrack(track: Track) {
        setPage("discover");
        if (track.id === currentTrackId) {
            void togglePlayback();
            return;
        }
        setTracks((current) => current.some((item) => item.id === track.id) ? current : [...current, track]);
        dispatch(requestPlayback(track.id));
    }

    function playMapTrack(track: TasteMapPoint) {
        setTracks((current) => current.some((item) => item.id === track.id) ? current : [...current, track]);
        dispatch(requestPlayback(track.id));
    }

    function useMapTrackAsSeed(track: TasteMapPoint) {
        setTracks((current) => current.some((item) => item.id === track.id) ? current : [...current, track]);
        setMapSelectedId(track.id);
        dispatch(selectSeedTrack(track.id));
        setPage("discover");
    }

    function exploreFromCurrentTrack() {
        if (currentTrackId !== seedTrackId) dispatch(selectSeedTrack(currentTrackId));
    }

    function toggleRecommendationShuffle() {
        if (shuffleEnabled) {
            dispatch(setShuffle({ enabled: false, order: [] }));
            return;
        }
        dispatch(setShuffle({ enabled: true, order: shuffleTrackIds(visibleRecommendations) }));
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
                    <button className={page === "map" ? "nav-item active" : "nav-item"} onClick={() => setPage("map")}>
                        <MapIcon size={18} /><span>Taste Map</span>
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
                        {searchOpen && search.trim() && searchResults.length > 0 && (
                            <div className="search-results" role="listbox">
                                {searchResults.slice(0, 5).map((track) => (
                                    <button key={track.id} onMouseDown={() => chooseTrack(track)} role="option" aria-selected={track.id === currentTrackId}>
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
                            <div className="eyebrow"><span className="eyebrow-mark"></span>{page === "map" ? "Discogs-EffNet projection" : "Sound-led discovery"}</div>
                            <h1>{page === "map" ? "Taste Map" : page === "saved" ? "Your saved tracks" : "Find a new way in."}</h1>
                            <p>{page === "map" ? "Explore where tracks sit in the EffNet sound space and trace what you have played." : page === "saved" ? "The tracks you want to come back to." : "Start with a track you love. Let the sound take you somewhere new."}</p>
                        </div>
                    </section>

                    {page === "discover" && (
                        <>
                            <section className="discovery-grid" aria-label="Discovery controls">
                                <div className="seed-feature">
                                    <img className="seed-cover" src={selectedTrack.coverUrl} alt={`Cover art for ${selectedTrack.title}`} />
                                    <div className="seed-shade"></div>
                                    <div className="seed-topline"><span className="overline">{currentTrackId === seedTrackId ? "Starting track" : "Now playing"}</span><span className="seed-id">{selectedTrack.id}</span></div>
                                    <div className="seed-details">
                                        <div><div className="seed-genre">{selectedTrack.genre} <span>·</span> {audioDuration || selectedTrack.duration}</div><h2>{selectedTrack.title}</h2><p>{selectedTrack.artist}</p></div>
                                        <button className="play-button" onClick={togglePlayback} aria-label={playing ? "Pause audio" : isPreview ? "Play demo audio" : "Play track"} title={playing ? "Pause" : isPreview ? "Play demo audio" : "Play track"} disabled={!selectedTrack.audioUrl}>{playing ? <Pause size={20} fill="currentColor" /> : <Play size={20} fill="currentColor" />}</button>
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
                                <div className="section-heading"><div><div className="eyebrow"><span className="eyebrow-mark"></span>Similar to {seedTrack.title}</div><h2>Follow this thread</h2></div><div className="recommendation-actions"><button className={shuffleEnabled ? "icon-button shuffle-button active" : "icon-button shuffle-button"} onClick={toggleRecommendationShuffle} aria-label={shuffleEnabled ? "Turn off shuffle" : "Shuffle recommendations"} aria-pressed={shuffleEnabled} title={shuffleEnabled ? "Turn off shuffle" : "Shuffle recommendations"}><Shuffle size={16} /></button><button className="text-button" onClick={exploreFromCurrentTrack} disabled={currentTrackId === seedTrackId}>Explore this track <span>↗</span></button></div></div>
                                <div className="list-heading"><span>#</span><span>Track</span><span>Style</span><span>Distance</span><span></span></div>
                                {error && <div className="error-state" role="alert">{error}</div>}
                                {!error && recommendationsLoading && <div className="empty-state" role="status">Finding tracks similar to {seedTrack.title}...</div>}
                                {!error && !recommendationsLoading && visibleRecommendations.map((track) => <RecommendationRow key={`${track.id}-${track.rank}`} track={track} saved={saved.includes(track.id)} playing={playing && currentTrackId === track.id} onPlay={() => playRecommendedTrack(track)} onToggleSaved={() => toggleSaved(track.id)} />)}
                                {!error && !recommendationsLoading && recommendations.length === 0 && <div className="empty-state">No recommendations for this selection yet.</div>}
                                <div className="list-footnote"><span>Cosine distance</span> is a model measure, not a score of how much you will like a track.</div>
                            </section>
                        </>
                    )}

                    {page === "map" && (
                        <section className="taste-map-section" aria-label="Taste Map">
                            <div className="taste-map-heading">
                                <div>
                                    <div className="eyebrow"><span className="eyebrow-mark"></span>8,000-track sound space</div>
                                    <h2>Explore the sound space</h2>
                                </div>
                                <div className="map-legend">
                                    <span><i className="legend-dot current"></i>Now playing</span>
                                    <span><i className="legend-dot seed"></i>Recommendation seed</span>
                                    <span><i className="legend-line"></i>Listening path</span>
                                </div>
                            </div>
                            {tasteMapLoading && <div className="empty-state" role="status">Building Taste Map coordinates…</div>}
                            {tasteMapError && <div className="error-state" role="alert">{tasteMapError}</div>}
                            {!tasteMapLoading && !tasteMapError && tasteMapPoints.length > 0 && (
                                <div className="taste-map-layout">
                                    <div className="taste-map-main">
                                        <TasteMapCanvas
                                            points={tasteMapPoints}
                                            listeningPath={listeningPath}
                                            currentTrackId={currentTrackId}
                                            seedTrackId={seedTrackId}
                                            recommendationIds={recommendations.map((track) => track.id)}
                                            selectedId={mapSelectedId}
                                            onSelect={(point) => setMapSelectedId(point.id)}
                                        />
                                        <div className="taste-map-caption">
                                            <span>{tasteMapPoints.length.toLocaleString()} tracks plotted</span>
                                            <span>2D PCA projection for display; recommendations still use full EffNet embeddings.</span>
                                        </div>
                                    </div>
                                    <aside className="taste-map-detail">
                                        {mapSelectedTrack ? (
                                            <>
                                                <img src={mapSelectedTrack.coverUrl} alt="" />
                                                <span className="overline">Selected track</span>
                                                <h3>{mapSelectedTrack.title}</h3>
                                                <p>{mapSelectedTrack.artist} · {mapSelectedTrack.genre}</p>
                                                <button className="text-button" onClick={() => playMapTrack(mapSelectedTrack)} disabled={!mapSelectedTrack.audioUrl}>Play track <span>▶</span></button>
                                                <button className="explore-map-button" onClick={() => useMapTrackAsSeed(mapSelectedTrack)}>Use as recommendation seed</button>
                                            </>
                                        ) : <div className="empty-state">Select a point to inspect a track.</div>}
                                        <div className="trail-heading">
                                            <strong>Listening path</strong>
                                            <button className="text-button" onClick={() => dispatch(clearListeningPath())} disabled={listeningPath.length === 0}>Clear</button>
                                        </div>
                                        {listeningPath.length === 0 ? <p className="trail-empty">Play tracks and their order will appear here.</p> : (
                                            <ol className="trail-list">
                                                {listeningPath.map((trackId, index) => {
                                                    const track = tasteMapPoints.find((point) => point.id === trackId);
                                                    return <li key={`${trackId}-${index}`}><span>{String(index + 1).padStart(2, "0")}</span><div><strong>{track?.title ?? `Track ${trackId}`}</strong><small>{track?.artist ?? trackId}</small></div></li>;
                                                })}
                                            </ol>
                                        )}
                                    </aside>
                                </div>
                            )}
                        </section>
                    )}

                    {page === "saved" && (
                        <section className="recommendations-section saved-section">
                            <div className="section-heading"><div><div className="eyebrow"><span className="eyebrow-mark"></span>Your collection</div><h2>Kept for later</h2></div><span className="saved-count">{saved.length} SAVED</span></div>
                            <div className="list-heading"><span>#</span><span>Track</span><span>Style</span><span>Distance</span><span></span></div>
                            {visibleRecommendations.map((track) => <RecommendationRow key={`${track.id}-${track.rank}`} track={track} saved playing={playing && currentTrackId === track.id} onPlay={() => playRecommendedTrack(track)} onToggleSaved={() => toggleSaved(track.id)} />)}
                            {visibleRecommendations.length === 0 && <div className="empty-state">Save a recommendation with the heart button and it will appear here.</div>}
                        </section>
                    )}
                </div>

                <footer className="page-footer"><span>TASTEBRIDGE / MUSIC DISCOVERY</span><span>{isPreview ? "DEMO DATA" : "CONNECTED TO RECOMMENDER"}</span></footer>
            </main>

            <div className="player-bar">
                <div className="player-track"><img src={selectedTrack.coverUrl} alt="" /><div><strong>{selectedTrack.title}</strong><span>{selectedTrack.artist}</span></div><button className="icon-button" onClick={() => toggleSaved(selectedTrack.id)} aria-label="Save current track"><Heart size={16} fill={saved.includes(selectedTrack.id) ? "currentColor" : "none"} /></button></div>
                <div className="player-center"><button className="player-play" onClick={togglePlayback} aria-label={playing ? "Pause audio" : "Play track"} disabled={!selectedTrack.audioUrl}>{playing ? <Pause size={15} fill="currentColor" /> : <Play size={15} fill="currentColor" />}</button><div className="fake-progress"><span></span></div><small>{isPreview ? "DEMO AUDIO" : "FMA AUDIO"}</small></div>
                <div className="player-volume"><ListMusic size={16} /><Volume2 size={17} /><div className="volume-line"><span></span></div></div>
            </div>
            <audio ref={audioRef} src={selectedTrack.audioUrl} preload="metadata" onLoadedMetadata={(event) => { const seconds = Math.round(event.currentTarget.duration); setAudioDuration(`${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, "0")}`); }} onPlay={() => { dispatch(setPlaying(true)); dispatch(recordPlayedTrack(currentTrackId)); }} onPause={() => dispatch(setPlaying(false))} onEnded={() => dispatch(setPlaying(false))} />
        </div>
    );
}

function RecommendationRow({ track, saved, playing, onPlay, onToggleSaved }: {
    track: Recommendation;
    saved: boolean;
    playing: boolean;
    onPlay: () => void;
    onToggleSaved: () => void;
}) {
    return (
        <div className="track-row">
            <span className="track-rank">{String(track.rank).padStart(2, "0")}</span>
            <div className="track-cell"><button className="track-select" onClick={onPlay} aria-label={`Play ${track.title} by ${track.artist}`} disabled={!track.audioUrl}><img className="track-cover" src={track.coverUrl} alt="" /><span className="track-copy"><strong>{track.title}</strong><span>{track.artist}</span></span></button><button className="row-play" onClick={onPlay} aria-label={`${playing ? "Pause" : "Play"} ${track.title}`} title={playing ? "Pause" : "Play track"} disabled={!track.audioUrl}>{playing ? <Pause size={14} fill="currentColor" /> : <Play size={14} fill="currentColor" />}</button></div>
            <span className="track-genre">{track.genre}</span>
            <span className="track-distance">{track.distance.toFixed(3)}</span>
            <button className={saved ? "icon-button heart-button is-saved" : "icon-button heart-button"} onClick={onToggleSaved} aria-label={saved ? `Remove ${track.title} from saved` : `Save ${track.title}`} title={saved ? "Saved" : "Save track"}>{saved ? <Check size={16} /> : <Heart size={16} />}</button>
        </div>
    );
}

export default App;

function shuffleTrackIds(tracks: Track[]): string[] {
    const ids = tracks.map((track) => track.id);
    for (let index = ids.length - 1; index > 0; index -= 1) {
        const swapIndex = Math.floor(Math.random() * (index + 1));
        [ids[index], ids[swapIndex]] = [ids[swapIndex], ids[index]];
    }
    return ids;
}

function orderTracks<T extends Track>(tracks: T[], order: string[]): T[] {
    const byId = new Map(tracks.map((track) => [track.id, track]));
    const ordered = order.flatMap((id) => {
        const track = byId.get(id);
        return track ? [track] : [];
    });
    const orderedIds = new Set(ordered.map((track) => track.id));
    return [...ordered, ...tracks.filter((track) => !orderedIds.has(track.id))];
}