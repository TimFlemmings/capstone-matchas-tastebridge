import { useEffect, useState } from "react";
import { Search } from "lucide-react";
import { searchTracks, type Track } from "../api/recommendations";

/** A search field with its own results dropdown. Picking a result calls `onChoose`. */
export default function SearchBox({ onChoose, placeholder, className = "search-wrap", currentTrackId }: {
    onChoose: (track: Track) => void;
    placeholder: string;
    className?: string;
    currentTrackId?: string;
}) {
    const [query, setQuery] = useState("");
    const [open, setOpen] = useState(false);
    const [results, setResults] = useState<Track[]>([]);
    const [failed, setFailed] = useState(false);

    useEffect(() => {
        if (!query.trim()) {
            setResults([]);
            return;
        }
        let active = true;
        const timer = window.setTimeout(() => {
            searchTracks(query)
                .then((found) => {
                    if (!active) return;
                    setResults(found);
                    setFailed(false);
                })
                .catch(() => {
                    if (active) setFailed(true);
                });
        }, 180);
        return () => {
            active = false;
            window.clearTimeout(timer);
        };
    }, [query]);

    function choose(track: Track) {
        onChoose(track);
        setQuery("");
        setOpen(false);
    }

    return (
        <div className={className}>
            <Search size={15} />
            <input
                aria-label="Search tracks"
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                onFocus={() => setOpen(true)}
                onBlur={() => window.setTimeout(() => setOpen(false), 140)}
                placeholder={placeholder}
            />
            {open && query.trim() && (results.length > 0 || failed) && (
                <div className="search-results" role="listbox">
                    {failed
                        ? <div className="search-failed">Search isn't reachable right now.</div>
                        : results.slice(0, 5).map((track) => (
                            <button key={track.id} onMouseDown={() => choose(track)} role="option" aria-selected={track.id === currentTrackId}>
                                <img src={track.coverUrl} alt="" /><span><strong>{track.title}</strong><small>{track.artist} · {track.genre}{track.matchedBy === "sound" && " · sounds like your search"}</small></span>
                            </button>
                        ))}
                </div>
            )}
        </div>
    );
}
