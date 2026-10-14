import { useEffect, useRef, useState } from "react";
import {
    ArrowDown,
    ArrowUp,
    Compass,
    GripVertical,
    Info,
    Bookmark,
    BookmarkCheck,
    Library,
    Pause,
    Shuffle,
    SkipBack,
    SkipForward,
    Map as MapIcon,
    Volume2,
} from "lucide-react";
import { useDispatch, useSelector } from "react-redux";
import {
    getRecommendations,
    getTracksById,
    isPreview,
    previewCatalog,
    searchTracks,
    type Recommendation,
    type Track,
} from "./api/recommendations";
import JourneyMap from "./components/JourneyMap";
import Onboarding from "./components/Onboarding";
import SearchBox from "./components/SearchBox";
import SpaceBackdrop from "./components/SpaceBackdrop";
import TasteTree from "./components/TasteTree";
import Waveform from "./components/Waveform";
import { NoteStars, PlayIcon, TastyLogo } from "./components/icons";
import {
    clearJourney,
    clearPlayRequest,
    moveSaved,
    plantSeed,
    requestPlayback,
    revisitSeed,
    selectSeedTrack,
    startJourney,
    store,
    toggleSaved as toggleSavedTrack,
    setPlaying,
    setShuffle,
    setShuffleOrder,
    type AppDispatch,
    type JourneyNode,
    type Link,
    type Origin,
    type RootState,
    type SavedTrack,
    type Tier,
} from "./state/store";

type Page = "discover" | "saved" | "tree";

function App() {
    const dispatch = useDispatch<AppDispatch>();
    const { seedTrackId, currentTrackId, playRequestedId, playing, shuffleEnabled, shuffleOrder } =
        useSelector((state: RootState) => state.playback);
    const journey = useSelector((state: RootState) => state.journey);
    const library = useSelector((state: RootState) => state.library);
    const savedIds = new Set(library.map((item) => item.track.id));
    const [page, setPage] = useState<Page>("discover");
    const [tracks, setTracks] = useState<Track[]>(previewCatalog);
    const [adventure, setAdventure] = useState(0.5);
    const [recommendations, setRecommendations] = useState<Recommendation[]>([]);
    const [recommendationsLoading, setRecommendationsLoading] = useState(false);
    const [error, setError] = useState("");
    const [catalogReady, setCatalogReady] = useState(isPreview);
    const [audioDuration, setAudioDuration] = useState("");
    const [position, setPosition] = useState(0);
    const [length, setLength] = useState(0);
    const [volume, setVolume] = useState(0.8);
    // the list next/previous walk through: whatever list playback started from
    // (empty = the current recommendations)
    // which list next/previous walk: always read live, so moving the slider (or
    // re-sorting the Library) takes effect straight away
    const [queue, setQueue] = useState<"recommendations" | "library">("recommendations");
    // the seed picker: "first" on a first visit, "again" from the sidebar link
    const [onboarding, setOnboarding] = useState<"first" | "again" | null>(null);
    const [mapView, setMapView] = useState<"tree" | "map">("tree");
    const [librarySort, setLibrarySort] = useState<LibrarySort>(loadLibrarySort);
    const [dragId, setDragId] = useState<string | null>(null);
    const [dragOverIndex, setDragOverIndex] = useState<number | null>(null);
    const audioRef = useRef<HTMLAudioElement>(null);

    const selectedTrack = tracks.find((track) => track.id === currentTrackId)
        ?? previewCatalog.find((track) => track.id === currentTrackId)
        ?? previewCatalog[0];
    const seedTrack = tracks.find((track) => track.id === seedTrackId)
        ?? previewCatalog.find((track) => track.id === seedTrackId)
        ?? previewCatalog[0];
    useEffect(() => {
        if (audioRef.current) audioRef.current.volume = volume;
    }, [volume]);

    function seek(seconds: number) {
        if (audioRef.current) audioRef.current.currentTime = seconds;
        setPosition(seconds);
    }

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
        setPosition(0);
        setLength(0);
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
        // the catalogue's first page, plus the default starting track (it isn't on that page)
        Promise.all([searchTracks(""), getTracksById([DEFAULT_SEED.id]).catch(() => [] as Track[])])
            .then(([catalogue, matches]) => {
                if (!active) return;
                const defaultSeed = matches.find((track) => track.id === DEFAULT_SEED.id);
                const found = defaultSeed ? [defaultSeed, ...catalogue.filter((track) => track.id !== defaultSeed.id)] : catalogue;
                const { nodes, currentKey } = store.getState().journey;
                const resumed = nodes.find((node) => node.key === currentKey);
                const journeyTracks = nodes.map((node) => node.track)
                    .filter((track) => !found.some((item) => item.id === track.id));
                setTracks([...found, ...journeyTracks]);
                if (resumed) dispatch(selectSeedTrack(resumed.track.id));  // pick up the saved journey
                else if (found.length > 0) {
                    dispatch(plantSeed({ track: trackOnly(found[0]), from: "root", tier: null, origin: { via: "default" } }));
                    dispatch(selectSeedTrack(found[0].id));
                    setOnboarding("first");
                }
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
        let active = true;
        if (!catalogReady) return () => { active = false; };
        setError("");
        setRecommendationsLoading(true);
        getRecommendations(seedTrackId, adventure)
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
    }, [catalogReady, seedTrackId, adventure, dispatch]);

    useEffect(() => {
        if (shuffleEnabled && recommendations.length > 1) {
            dispatch(setShuffleOrder(shuffleTrackIds(recommendations)));
        }
    }, [shuffleEnabled, recommendations, dispatch]);

    const visibleRecommendations = shuffleEnabled
        ? orderTracks(recommendations, shuffleOrder)
        : recommendations;
    const libraryTracks = sortLibrary(library.map((item) => item.track), librarySort);
    const playQueue = queue === "library" ? libraryTracks : visibleRecommendations;
    const manualOrder = librarySort.key === "manual";

    function chooseLibrarySort(key: SortKey) {
        const next = { key, desc: key === librarySort.key && key !== "manual" ? !librarySort.desc : false };
        setLibrarySort(next);
        try {
            localStorage.setItem(LIBRARY_SORT_KEY, JSON.stringify(next));
        } catch {
            // storage blocked: the choice just won't survive a reload
        }
    }

    function dropOnRow(index: number) {
        if (dragId) dispatch(moveSaved({ trackId: dragId, to: index }));
        setDragId(null);
        setDragOverIndex(null);
    }

    useEffect(() => {
        setQueue("recommendations");  // a new seed: next/previous follow its recommendations
    }, [seedTrackId]);

    function rememberTrack(track: Track) {
        setTracks((current) => current.some((item) => item.id === track.id) ? current : [...current, track]);
    }

    /** Make `track` the seed and record the step in the taste tree: a search jump
     *  starts a new tree, anything else grows a branch from the current seed. */
    function plant(track: Track, from: "root" | "branch", origin?: Origin) {
        rememberTrack(track);
        dispatch(plantSeed({ track: trackOnly(track), from, tier: tierFor(adventure), link: linkTo(track), origin }));
        dispatch(selectSeedTrack(track.id));
    }

    function revisit(node: JourneyNode) {
        rememberTrack(node.track);
        dispatch(revisitSeed(node.key));
        dispatch(selectSeedTrack(node.track.id));
        setPage("discover");
        window.scrollTo({ top: 0, behavior: "smooth" });
    }

    function playTreeTrack(track: Track) {
        if (track.id === currentTrackId) {
            void togglePlayback();
            return;
        }
        rememberTrack(track);
        dispatch(requestPlayback(track.id));
    }

    /** The picker's choice: a first visit replaces the default start, later picks start a new tree. */
    function chooseStart(track: Track, origin: Origin) {
        if (onboarding === "first") {
            rememberTrack(track);
            dispatch(startJourney({ track: trackOnly(track), origin }));
            dispatch(selectSeedTrack(track.id));
        } else {
            plant(track, "root", origin);
        }
        setOnboarding(null);
        setPage("discover");
        window.scrollTo({ top: 0 });
    }

    function chooseTrack(track: Track, query: string) {
        plant(track, "root", { via: "search", query });
        setPage("discover");
    }

    /** Play `track` from a list; next/previous then walk that list (as it is at the time). */
    function playFromList(track: Track, list: "recommendations" | "library") {
        setQueue(list);
        if (track.id === currentTrackId) {
            void togglePlayback();
            return;
        }
        rememberTrack(track);
        dispatch(requestPlayback(track.id));
    }

    /** Next (+1) or previous (-1) in the queue. Previous restarts the track if
     *  it's more than 3 s in, like most players. */
    function skip(direction: 1 | -1) {
        if (direction < 0 && position > 3) {
            seek(0);
            return;
        }
        const index = playQueue.findIndex((track) => track.id === currentTrackId);
        const target = index === -1 ? (direction > 0 ? playQueue[0] : undefined) : playQueue[index + direction];
        if (!target) {
            if (direction < 0) seek(0);
            return;
        }
        rememberTrack(target);
        dispatch(requestPlayback(target.id));
    }

    // the right panel always shows the seed, whatever is playing
    const seedIsPlaying = currentTrackId === seedTrackId;

    function playSeed() {
        if (seedIsPlaying) {
            void togglePlayback();
            return;
        }
        setQueue("recommendations");  // after the seed: its recommendations
        dispatch(requestPlayback(seedTrackId));
    }

    function seekSeed(fraction: number) {
        if (seedIsPlaying) seek(fraction * length);
        else playSeed();
    }

    const hasNext = playQueue.length > 0 && playQueue.findIndex((track) => track.id === currentTrackId) < playQueue.length - 1;

    function exploreFromCurrentTrack() {
        if (currentTrackId !== seedTrackId) plant(selectedTrack, "branch");
    }

    function exploreFromTrack(track: Track) {
        plant(track, "branch");
        setPage("discover");
        window.scrollTo({ top: 0, behavior: "smooth" });
    }

    function toggleRecommendationShuffle() {
        if (shuffleEnabled) {
            dispatch(setShuffle({ enabled: false, order: [] }));
            return;
        }
        dispatch(setShuffle({ enabled: true, order: shuffleTrackIds(visibleRecommendations) }));
    }

    /** Why `track` came up for the current seed, if it's in the recommendations. */
    function linkTo(track: Track): Link | undefined {
        const fromList = recommendations.find((item) => item.id === track.id);
        return fromList ? { factors: fromList.connectingFactors ?? [], distance: fromList.distance } : undefined;
    }

    /** Save or unsave a track. It's filed under the current seed (for the Taste
     *  Tree), with its distance from that seed if it came from the recommendations. */
    function toggleSaved(track: Track) {
        const fromList = recommendations.find((item) => item.id === track.id);
        dispatch(toggleSavedTrack({
            track: { ...trackOnly(track), distance: fromList?.distance },
            seedKey: journey.currentKey,
            link: linkTo(track),
        }));
    }

    return (
        <div className="app-shell">
            <aside className="sidebar">
                <a className="wordmark" href="#discover" onClick={() => setPage("discover")}>
                    <span className="wordmark-icon"><TastyLogo size={38} tone="white" /></span>
                    <span>tastey</span>
                </a>

                <button className="start-new-link" onClick={() => setOnboarding("again")}><NoteStars size={15} /> Start somewhere new</button>
                <div className="side-label">Your space</div>
                <nav className="primary-nav" aria-label="Main navigation">
                    <button className="nav-item start-new-nav" onClick={() => setOnboarding("again")} aria-label="Start somewhere new" title="Start somewhere new">
                        <NoteStars size={18} /><span>Start somewhere new</span>
                    </button>
                    <button className={page === "discover" ? "nav-item active" : "nav-item"} onClick={() => setPage("discover")}>
                        <Compass size={18} /><span>Discover</span>
                    </button>
                    <button className={page === "saved" ? "nav-item active" : "nav-item"} onClick={() => setPage("saved")}>
                        <Library size={18} /><span>Your library</span>
                    </button>
                    <button className={page === "tree" ? "nav-item active" : "nav-item"} onClick={() => setPage("tree")}>
                        <MapIcon size={18} /><span>Taste Map</span>
                    </button>
                </nav>

            </aside>

            <SpaceBackdrop page={page} />
            {onboarding && (
                <Onboarding firstVisit={onboarding === "first"} defaultTitle={seedTrack.title} nowPlayingId={currentTrackId} playing={playing}
                    onChoose={chooseStart} onDismiss={() => setOnboarding(null)} onPreview={playTreeTrack} />
            )}
            <main className="main-content">
                <header className="topbar">
                    <SearchBox onChoose={chooseTrack} currentTrackId={currentTrackId} placeholder="Search a song, artist or genre, or describe a sound" />
                    <div className="profile-button"><span className="profile-avatar">T</span><span>Listener</span></div>
                </header>

                {isPreview && <div className="preview-banner"><span className="status-dot preview"></span><strong>Preview mode</strong><span>Sample tracks and responses are shown; they are not neural-network results.</span></div>}

                <div className="content-wrap">
                    <section className="page-intro">
                        <div>
                            <div className="eyebrow">{page === "tree" ? "Your journey" : page === "saved" ? "Your library" : "Sound-led discovery"}</div>
                            <h1>{page === "tree" ? "Taste Map" : page === "saved" ? "Your saved tracks" : "Find a new way in"}</h1>
                            <p>{page === "tree" ? "Every seed you explored, the tracks you saved along the way, and where you went next. Pick any seed to branch off again" : page === "saved" ? "The tracks you want to come back to" : "Start with a seed track that you love. Let the sound take you somewhere new"}</p>
                        </div>
                    </section>

                    {page === "discover" && (
                        <>
                            <div className="discover-layout">
                                <section className="recommendations-section">
                                    <div className="section-heading"><div><div className="eyebrow">Similar to seed track: {seedTrack.title}</div></div><div className="recommendation-actions"><button className={shuffleEnabled ? "icon-button shuffle-button active" : "icon-button shuffle-button"} onClick={toggleRecommendationShuffle} aria-label={shuffleEnabled ? "Turn off shuffle" : "Shuffle recommendations"} aria-pressed={shuffleEnabled} title={shuffleEnabled ? "Turn off shuffle" : "Shuffle recommendations"}><Shuffle size={16} /></button><button className="text-button" onClick={exploreFromCurrentTrack} disabled={currentTrackId === seedTrackId}><NoteStars size={15} /> Seed this track</button><InfoTip label="Seed this track" align="right" text="Makes the track that's playing your new seed: the playlist refreshes with tracks that sound like it, and the step is added to your Taste Map. Each row's notes icon does the same for that track." /></div></div>
                                    <div className="list-heading"><span>Track</span><span>Genre</span><span className="distance-heading">Taste distance <InfoTip label="Taste distance" text="How far a track's sound is from your seed, in bridgons. The bigger the number, the bigger the leap: under 30 stays close to home, 40 and up takes you somewhere new. It's about sound, not a score of how much you'll like it." /></span><span></span><span></span></div>
                                    {error && <div className="error-state" role="alert">{error}</div>}
                                    {!error && recommendationsLoading && <div className="empty-state" role="status">Finding tracks similar to {seedTrack.title}...</div>}
                                    {!error && !recommendationsLoading && visibleRecommendations.map((track) => <RecommendationRow key={`${track.id}-${track.rank}`} track={track} saved={savedIds.has(track.id)} playing={playing && currentTrackId === track.id} onPlay={() => playFromList(track, "recommendations")} onToggleSaved={() => toggleSaved(track)} onExplore={() => exploreFromTrack(track)} isSeed={track.id === seedTrackId} />)}
                                    {!error && !recommendationsLoading && recommendations.length === 0 && <div className="empty-state">No recommendations for this selection yet.</div>}
                                </section>
                                <aside className="seed-panel" aria-label="Seed track and adventure level">
                                    <div className="seed-feature">
                                        <img className="seed-cover" src={seedTrack.coverUrl} alt={`Cover art for ${seedTrack.title}`} />
                                        <div className="seed-shade"></div>
                                        <div className="seed-heading">
                                            <span className="seed-label"><NoteStars size={12} /> Seed track</span>
                                            <h2>{seedTrack.title}</h2>
                                            <p>{seedTrack.artist}</p>
                                            <span className="seed-genre">{seedTrack.genre} · {seedIsPlaying && audioDuration ? audioDuration : seedTrack.duration}</span>
                                        </div>
                                        <div className="seed-controls">
                                            <span className="play-box"><button className="play-button" onClick={playSeed} aria-label={seedIsPlaying && playing ? "Pause the seed track" : "Play the seed track"} title={seedIsPlaying && playing ? "Pause" : "Play the seed track"} disabled={!seedTrack.audioUrl}>{seedIsPlaying && playing ? <Pause size={17} fill="currentColor" /> : <PlayIcon size={17} />}</button></span>
                                            <SearchBox onChoose={chooseTrack} currentTrackId={currentTrackId} placeholder="Search" className="search-wrap panel-search" />
                                        </div>
                                        <Waveform url={seedTrack.audioUrl} progress={seedIsPlaying && length ? position / length : 0} onSeek={seekSeed} className="panel-wave" />
                                    </div>

                                    <div className="adventure-panel">
                                        <div className="adventure-value"><span>Your discovery journey</span><strong>{Math.round(adventure * 100)}<small>%</small></strong></div>
                                        <input className="adventure-slider" type="range" min="0" max="100" step="50" value={Math.round(adventure * 100)} onChange={(event) => setAdventure(Number(event.target.value) / 100)} aria-label="Adventure level" style={{ "--slider-progress": `${adventure * 100}%` } as React.CSSProperties} />
                                        <div className="range-labels"><span>Home</span><span>Bridge</span><span>Adventure</span></div>
                                    </div>
                                </aside>
                            </div>
                        </>
                    )}

                    {page === "tree" && (
                        <section className="taste-tree-section" aria-label="Taste Map">
                            <div className="taste-tree-heading">
                                <div>
                                    <div className="eyebrow">{journeyStats(journey.nodes, library)}</div>
                                    <h2>Where your taste went</h2>
                                </div>
                                <div className="tree-legend">
                                    <span><i className="tier-familiar"></i>Home</span>
                                    <span><i className="tier-bridge"></i>Bridge</span>
                                    <span><i className="tier-adventurous"></i>Adventure</span>
                                    <button className="text-button" onClick={() => dispatch(clearJourney())} disabled={journey.nodes.length < 2}>Start over</button>
                                </div>
                            </div>
                            <div className="sort-control map-view-toggle" role="group" aria-label="Taste Map view">
                                <span>View</span>
                                <button className={mapView === "tree" ? "active" : undefined} aria-pressed={mapView === "tree"} onClick={() => setMapView("tree")}>Tree</button>
                                <button className={mapView === "map" ? "active" : undefined} aria-pressed={mapView === "map"} onClick={() => setMapView("map")}>Map</button>
                            </div>
                            {journey.nodes.length === 0
                                ? <div className="empty-state">Pick a starting track and your tree will grow from it.</div>
                                : mapView === "map"
                                ? <JourneyMap nodes={journey.nodes} saved={library} currentKey={journey.currentKey} nowPlayingId={currentTrackId} onRevisit={revisit} onPlay={playTreeTrack} />
                                : <TasteTree nodes={journey.nodes} saved={library} currentKey={journey.currentKey} nowPlayingId={currentTrackId} playing={playing} onRevisit={revisit} onPlay={playTreeTrack} onUnsave={toggleSaved} />}
                            {journey.nodes.length === 1 && <p className="tree-hint"><Bookmark size={13} /> Save tracks you like and <NoteStars size={13} /> seed them on Discover: your tree grows from both.</p>}
                        </section>
                    )}

                    {page === "saved" && (
                        <section className="recommendations-section saved-section">
                            <div className="library-toolbar">
                                <div className="sort-control" role="group" aria-label="Sort your Library">
                                    <span>Sort by</span>
                                    {SORT_OPTIONS.map(([key, label]) => (
                                        <button key={key} className={librarySort.key === key ? "active" : undefined} onClick={() => chooseLibrarySort(key)}
                                            aria-pressed={librarySort.key === key}
                                            title={key === "manual" ? "Your own order: drag the handles" : `Sort by ${label.toLowerCase()} (click again to reverse)`}>
                                            {label}
                                            {librarySort.key === key && key !== "manual" && (librarySort.desc ? <ArrowDown size={11} /> : <ArrowUp size={11} />)}
                                        </button>
                                    ))}
                                </div>
                                {manualOrder && library.length > 1 && <span className="library-hint">Drag <GripVertical size={11} /> to reorder</span>}
                            </div>
                            <div className="list-heading library-columns"><span>Track</span><span>Genre</span><span>Location</span><span>Year created</span><span className="distance-heading">BPM <InfoTip label="BPM" align="right" text="Beats per minute, estimated from the audio. Very fast or half-time tracks can read double or half." /></span><span></span><span></span></div>
                            {libraryTracks.map((track, index) => (
                                <RecommendationRow key={track.id} variant="library" track={track} saved playing={playing && currentTrackId === track.id}
                                    onPlay={() => playFromList(track, "library")} onToggleSaved={() => toggleSaved(track)} onExplore={() => exploreFromTrack(track)} isSeed={track.id === seedTrackId}
                                    reorder={manualOrder ? {
                                        dragging: dragId === track.id,
                                        over: dragOverIndex === index && dragId !== track.id,
                                        onDragStart: () => setDragId(track.id),
                                        onDragEnter: () => setDragOverIndex(index),
                                        onDrop: () => dropOnRow(index),
                                        onDragEnd: () => { setDragId(null); setDragOverIndex(null); },
                                        onMove: (delta) => dispatch(moveSaved({ trackId: track.id, to: index + delta })),
                                    } : undefined} />
                            ))}
                            {library.length === 0 && <div className="empty-state">Save a recommendation with the bookmark button and it will appear here.</div>}
                        </section>
                    )}
                </div>

                <footer className="page-footer"><span>TASTEY / MUSIC DISCOVERY</span></footer>
            </main>

            <div className="player-bar">
                <div className="player-track"><img src={selectedTrack.coverUrl} alt="" /><div><strong>{selectedTrack.title}</strong><span>{selectedTrack.artist}</span></div><button className="icon-button explore-button" onClick={exploreFromCurrentTrack} disabled={currentTrackId === seedTrackId} aria-label={`Seed ${selectedTrack.title}`} title="Seed this track"><NoteStars size={16} /></button><button className={savedIds.has(selectedTrack.id) ? "icon-button saved" : "icon-button"} onClick={() => toggleSaved(selectedTrack)} aria-label={savedIds.has(selectedTrack.id) ? "Remove from Library" : "Save to Library"} title={savedIds.has(selectedTrack.id) ? "In your Library" : "Save to Library"}>{savedIds.has(selectedTrack.id) ? <BookmarkCheck size={16} /> : <Bookmark size={16} />}</button></div>
                <div className="player-center">
                    <button className="icon-button skip-button" onClick={() => skip(-1)} aria-label="Previous track" title="Previous"><SkipBack size={16} fill="currentColor" /></button>
                    <span className="play-box"><button className="player-play" onClick={togglePlayback} aria-label={playing ? "Pause audio" : "Play track"} disabled={!selectedTrack.audioUrl}>{playing ? <Pause size={15} fill="currentColor" /> : <PlayIcon size={15} />}</button></span>
                    <button className="icon-button skip-button" onClick={() => skip(1)} disabled={!hasNext} aria-label="Next track" title="Next"><SkipForward size={16} fill="currentColor" /></button>
                    <Waveform url={selectedTrack.audioUrl} progress={length ? position / length : 0} onSeek={(fraction) => seek(fraction * length)} className="player-wave" />
                    <small>{formatTime(position)} / {length ? formatTime(length) : "--:--"}</small>
                </div>
                <div className="player-volume"><Volume2 size={17} /><input className="player-slider player-volume-slider" type="range" min="0" max="1" step="0.01" value={volume} onChange={(event) => setVolume(Number(event.target.value))} aria-label="Volume" style={{ "--slider-progress": `${volume * 100}%` } as React.CSSProperties} /></div>
            </div>
            <audio ref={audioRef} src={selectedTrack.audioUrl} preload="metadata" onLoadedMetadata={(event) => { event.currentTarget.volume = volume; setLength(event.currentTarget.duration); setAudioDuration(formatTime(event.currentTarget.duration)); }} onTimeUpdate={(event) => setPosition(event.currentTarget.currentTime)} onPlay={() => { dispatch(setPlaying(true)); }} onPause={() => dispatch(setPlaying(false))} onEnded={() => { dispatch(setPlaying(false)); skip(1); }} />
        </div>
    );
}

/** Where a first visit starts. Visitors with a saved journey resume their own seed. */
const DEFAULT_SEED = { id: "13668", title: "No Time For Lookin' Back" };  // by This One; used when the picker is skipped

function tierFor(adventure: number): Tier {
    // mirrors the server's tier_for_adventure
    return adventure < 1 / 3 ? "Familiar" : adventure < 2 / 3 ? "Bridge" : "Adventurous";
}

/** Just the Track fields -- a Recommendation also carries rank, distance, factors. */
function trackOnly({ id, title, artist, genre, duration, coverUrl, audioUrl, location, dateCreated, bpm }: Track): Track {
    return { id, title, artist, genre, duration, coverUrl, audioUrl, location, dateCreated, bpm };
}

function journeyStats(nodes: JourneyNode[], library: SavedTrack[]) {
    const keys = new Set(nodes.map((node) => node.key));
    const saved = library.filter((item) => item.seedKey && keys.has(item.seedKey)).length;
    return `${nodes.length} seed${nodes.length === 1 ? "" : "s"} · ${saved} saved`;
}

type SortKey = "manual" | "genre" | "bpm" | "year";
type LibrarySort = { key: SortKey; desc: boolean };

const SORT_OPTIONS: [SortKey, string][] = [["manual", "Manual"], ["genre", "Genre"], ["bpm", "BPM"], ["year", "Year"]];
const LIBRARY_SORT_KEY = "tasty.librarySort.v1";

function loadLibrarySort(): LibrarySort {
    try {
        const stored = JSON.parse(localStorage.getItem(LIBRARY_SORT_KEY) ?? "null");
        if (stored && SORT_OPTIONS.some(([key]) => key === stored.key)) return stored as LibrarySort;
    } catch {
        // fall through to the default
    }
    return { key: "manual", desc: false };
}

/** Library order: manual = as stored; otherwise by the field, missing values last either way. */
function sortLibrary(tracks: Track[], { key, desc }: LibrarySort): Track[] {
    if (key === "manual") return tracks;
    const value = (track: Track): string | number | null =>
        key === "genre" ? track.genre || null : key === "bpm" ? track.bpm ?? null : track.dateCreated?.slice(0, 4) ?? null;
    return [...tracks].sort((a, b) => {
        const va = value(a);
        const vb = value(b);
        if (va === null || vb === null) return va === vb ? 0 : va === null ? 1 : -1;
        const order = typeof va === "number" && typeof vb === "number" ? va - vb : String(va).localeCompare(String(vb));
        return (desc ? -order : order) || a.title.localeCompare(b.title);
    });
}

function formatTime(seconds: number) {
    const whole = Math.max(0, Math.round(seconds));
    return `${Math.floor(whole / 60)}:${String(whole % 60).padStart(2, "0")}`;
}

/** An "i" that explains something on hover or keyboard focus. */
function InfoTip({ label, text, align = "center" }: { label: string; text: string; align?: "center" | "right" }) {
    return (
        <span className={align === "right" ? "info-tip align-right" : "info-tip"} tabIndex={0} aria-label={`${label}: ${text}`} data-tip={text}>
            <Info size={12} />
        </span>
    );
}

type Reorder = {
    dragging: boolean;
    over: boolean;
    onDragStart: () => void;
    onDragEnter: () => void;
    onDrop: () => void;
    onDragEnd: () => void;
    onMove: (delta: number) => void;
};

function RecommendationRow({ track, saved, playing, isSeed, onPlay, onToggleSaved, onExplore, variant = "discover", reorder }: {
    variant?: "discover" | "library";
    reorder?: Reorder;
    track: Track & { distance?: number };
    saved: boolean;
    playing: boolean;
    isSeed: boolean;
    onPlay: () => void;
    onToggleSaved: () => void;
    onExplore: () => void;
}) {
    return (
        <div className={["track-row", variant === "library" && "library-columns", reorder?.dragging && "dragging", reorder?.over && "drag-over"].filter(Boolean).join(" ")}
            draggable={Boolean(reorder)}
            onDragStart={reorder ? (event) => { event.dataTransfer.effectAllowed = "move"; reorder.onDragStart(); } : undefined}
            onDragEnter={reorder?.onDragEnter}
            onDragOver={reorder ? (event) => event.preventDefault() : undefined}
            onDrop={reorder ? (event) => { event.preventDefault(); reorder.onDrop(); } : undefined}
            onDragEnd={reorder?.onDragEnd}>
            <div className="track-cell">{reorder && (
                <span className="drag-handle" tabIndex={0} role="button" aria-label={`Move ${track.title} (arrow keys)`} title="Drag to reorder"
                    onKeyDown={(event) => {
                        if (event.key === "ArrowUp" || event.key === "ArrowDown") {
                            event.preventDefault();
                            reorder.onMove(event.key === "ArrowUp" ? -1 : 1);
                        }
                    }}><GripVertical size={14} /></span>
            )}<span className="play-box small"><button className="row-play" onClick={onPlay} aria-label={`${playing ? "Pause" : "Play"} ${track.title}`} title={playing ? "Pause" : "Play track"} disabled={!track.audioUrl}>{playing ? <Pause size={13} fill="currentColor" /> : <PlayIcon size={13} />}</button></span><button className="track-select" onClick={onPlay} aria-label={`Play ${track.title} by ${track.artist}`} disabled={!track.audioUrl}><img className="track-cover" src={track.coverUrl} alt="" /><span className="track-copy"><strong>{track.title}</strong><span>{track.artist}</span></span></button></div>
            <span className="track-genre">{track.genre}</span>
            {variant === "library" ? <>
                {track.location
                    ? <span className="track-location">{track.location}</span>
                    : <span className="track-location uncharted" title="FMA has no location for this artist">Uncharted</span>}
                <span className="track-date">{track.dateCreated ? track.dateCreated.slice(0, 4) : "–"}</span>
                <span className="track-bpm">{track.bpm ?? "–"}</span>
            </> : <span className="track-distance">{track.distance === undefined ? "–" : <>{Math.round(track.distance * 100)} <small>bridgons away</small></>}</span>}
            <button className="icon-button explore-button" onClick={onExplore} disabled={isSeed} aria-label={`Seed ${track.title}`} title="Seed this track"><NoteStars size={17} /></button>
            <button className={saved ? "icon-button heart-button is-saved" : "icon-button heart-button"} onClick={onToggleSaved} aria-label={saved ? `Remove ${track.title} from your Library` : `Save ${track.title} to your Library`} title={saved ? "In your Library" : "Save to Library"}>{saved ? <BookmarkCheck size={16} /> : <Bookmark size={16} />}</button>
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